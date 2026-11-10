from types import SimpleNamespace

import pytest
import torch

from sfllm.layers.sampler import Sampler, SamplingBatchInfo


@pytest.mark.parametrize("use_min_p", [False, True])
def test_bounded_top_k_retains_categorical_weights(monkeypatch, use_min_p):
    sampler = Sampler(SimpleNamespace(vocab_size=128))
    # Distinct probabilities make the selected token set unambiguous.
    logits = torch.linspace(-3, 3, 128).repeat(3, 1)
    logits[1] = logits[1].flip(0)
    probabilities = logits.softmax(-1)
    ks = torch.tensor([1, 7, 20])
    ps = torch.tensor([0.3, 0.8, 1.0])
    min_ps = torch.tensor([0.0, 0.5, 0.7])
    draws = []

    def draw(weights, num_samples):
        draws.append(weights.clone())
        return weights.argmax(-1, keepdim=True)

    monkeypatch.setattr(torch, "multinomial", draw)
    full = sampler.top_k_top_p_min_p_sampling_from_probs_torch(
        probabilities, ks, ps, min_ps, use_min_p,
    )
    bounded = sampler.top_k_top_p_min_p_sampling_from_probs_torch(
        probabilities, ks, ps, min_ps, use_min_p, max_top_k=20,
    )
    torch.testing.assert_close(draws[1], draws[0][:, :20], atol=0, rtol=0)
    assert torch.count_nonzero(draws[0][:, 20:]) == 0
    assert torch.equal(full, bounded)


@pytest.mark.parametrize("bound", [None, 0, 65, 128, 1000000])
def test_unbounded_sampling_preserves_full_vocabulary(monkeypatch, bound):
    sampler = Sampler(SimpleNamespace(vocab_size=128))
    weights = []

    def draw(probabilities, num_samples):
        weights.append(probabilities)
        return probabilities.argmax(-1, keepdim=True)

    monkeypatch.setattr(torch, "multinomial", draw)
    sampler.top_k_top_p_min_p_sampling_from_probs_torch(
        torch.ones(1, 128) / 128, torch.tensor([128]), torch.tensor([1.0]),
        torch.tensor([0.0]), False, max_top_k=bound,
    )
    assert weights[0].shape == (1, 128)


def test_forward_uses_batch_top_k_bound(monkeypatch):
    sampler = Sampler(SimpleNamespace(vocab_size=128))
    shapes = []

    def draw(weights, num_samples):
        shapes.append(weights.shape)
        return weights.argmax(-1, keepdim=True)

    monkeypatch.setattr(torch, "multinomial", draw)
    info = SamplingBatchInfo(
        temperatures=torch.ones(2), top_ps=torch.ones(2),
        top_ks=torch.tensor([1, 20]), min_ps=torch.zeros(2),
        is_all_greedy=False, max_top_k=20,
    )
    logits = torch.arange(256, dtype=torch.float32).reshape(2, 128)
    assert sampler(logits, info).tolist() == [127, 127]
    assert shapes == [(2, 20)]


def test_greedy_sampling_bypasses_selection(monkeypatch):
    sampler = Sampler(SimpleNamespace(vocab_size=3))

    def unexpected(*args, **kwargs):
        raise AssertionError("greedy decoding must not sample")

    monkeypatch.setattr(torch, "multinomial", unexpected)
    info = SamplingBatchInfo(None, None, None, None, True)
    assert sampler(torch.tensor([[0.0, 4.0, 1.0]]), info).tolist() == [1]


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="requires NVIDIA CUDA",
)
@pytest.mark.parametrize("use_min_p", [False, True])
def test_flashinfer_top_k_preserves_weights_and_graph_replay(monkeypatch, use_min_p):
    flashinfer = pytest.importorskip("flashinfer")
    sampler = Sampler(SimpleNamespace(vocab_size=65536))
    logits = torch.full((3, 65536), -100., device="cuda")
    logits[:, :32] = torch.linspace(-3, 3, 32, device="cuda")
    probabilities = logits.softmax(-1)
    ks = torch.tensor([1, 7, 20], device="cuda")
    ps = torch.tensor([0.3, 0.8, 1.0], device="cuda")
    min_ps = torch.tensor([0.0, 0.5, 0.7], device="cuda")
    draws, calls = [], []
    original_top_k = flashinfer.top_k

    def select(x, k, **kwargs):
        calls.append((k, kwargs))
        return original_top_k(x, k, **kwargs)

    def draw(weights, num_samples):
        draws.append(weights.clone())
        return weights.argmax(-1, keepdim=True)

    monkeypatch.setattr(flashinfer, "top_k", select)
    monkeypatch.setattr(torch, "multinomial", draw)

    def sample(bound):
        return sampler.top_k_top_p_min_p_sampling_from_probs_torch(
            probabilities, ks, ps, min_ps, use_min_p, max_top_k=bound,
        )

    expected = sample(None)
    actual = sample(20)
    assert calls == [(20, {"sorted": True, "deterministic": True})]
    torch.testing.assert_close(draws[1], draws[0][:, :20], atol=0, rtol=0)
    assert torch.count_nonzero(draws[0][:, 20:]) == 0
    assert torch.equal(actual, expected)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = sample(20)
    for shift in (19, 173, 1001):
        probabilities.copy_(probabilities.roll(shift, dims=-1))
        graph.replay()
        assert torch.equal(actual, sample(None))


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="requires NVIDIA CUDA",
)
@pytest.mark.parametrize("case", ["small_vocab", "float64", "strided", "missing_backend"])
def test_flashinfer_top_k_dispatch_falls_back_to_torch(monkeypatch, case):
    flashinfer = pytest.importorskip("flashinfer")
    import sfllm.kernels.topk as topk_module
    width = 128 if case == "small_vocab" else 65536
    dtype = torch.float64 if case == "float64" else torch.float32
    probabilities = torch.rand(2, width * (2 if case == "strided" else 1),
                               device="cuda", dtype=dtype)
    if case == "strided":
        probabilities = probabilities[:, ::2]

    def unexpected(*args, **kwargs):
        raise AssertionError("This tensor must use the Torch fallback")

    def draw(weights, num_samples):
        return weights.argmax(-1, keepdim=True)

    monkeypatch.setattr(flashinfer, "top_k", unexpected)
    if case == "missing_backend":
        monkeypatch.setattr(topk_module, "flashinfer", None)
    monkeypatch.setattr(torch, "multinomial", draw)
    sampler = Sampler(SimpleNamespace(vocab_size=width))
    actual = sampler.top_k_top_p_min_p_sampling_from_probs_torch(
        probabilities, torch.full((2,), 20, device="cuda"),
        torch.ones(2, device="cuda"), torch.zeros(2, device="cuda"),
        False, max_top_k=20,
    )
    assert torch.equal(actual, probabilities.argmax(-1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("ks,bound,width", [
    ([1, 7, 20], 20, 20),
    ([1, 1073741824], 1073741824, 65536),
    ([1, 1], None, None),
])
def test_scheduled_mixed_sampling_paths(monkeypatch, ks, bound, width):
    from sfllm.engine.sampling_params import SamplingParams
    from sfllm.engine.schedule_batch import ScheduleBatch

    batch = ScheduleBatch([
        SimpleNamespace(sampling_params=SamplingParams(top_k=k, top_p=1.0))
        for k in ks
    ], None)
    batch.prepare_sample()
    info = batch.forward_batch.sampling_batch_info
    assert info.max_top_k == bound
    widths = []

    def draw(weights, num_samples):
        widths.append(weights.shape[-1])
        return weights.argmax(-1, keepdim=True)

    monkeypatch.setattr(torch, "multinomial", draw)
    logits = torch.linspace(-3, 3, 65536, device="cuda").repeat(len(ks), 1)
    tokens = Sampler(SimpleNamespace(vocab_size=65536))(logits, info)
    assert tokens.tolist() == [65535] * len(ks)
    assert widths == ([] if width is None else [width])
