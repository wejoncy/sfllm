"""Exact Top-k scores, deterministic ties, and changing-input Graph replay."""

import pytest
import torch

flashinfer = pytest.importorskip("flashinfer")
from sfllm.kernels.topk import pruned_topk

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("m,n", [(1, 65536), (96, 248320), (192, 248320), (12, 250001)])
def test_topk_graph_candidates(m, n):
    x = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
    pruned_topk(x)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        values, ids = pruned_topk(x)
    for case in ("random", "ties", "equal", "signed_zero", "wide"):
        if case == "random":
            x.normal_()
        elif case == "ties":
            x.random_(-3, 4)
        elif case == "equal":
            x.fill_(1)
        elif case == "signed_zero":
            x.zero_()
            x[:, ::2] = -0.0
        else:
            x.fill_(-float("inf"))
            x[:, :3] = float("inf")
        expected = flashinfer.top_k(x, 16, sorted=True, deterministic=True)
        graph.replay()
        assert torch.equal(values, expected[0]), case
        assert torch.equal(x.gather(1, ids), values), case
        assert torch.all(ids.sort(dim=1).values.diff(dim=1) > 0), case
        ties = values[:, 1:] == values[:, :-1]
        assert torch.all(~ties | (ids[:, 1:] > ids[:, :-1])), case
        if case in ("equal", "signed_zero"):
            assert torch.equal(ids, torch.arange(16, device="cuda").expand(m, -1))


@pytest.mark.parametrize("case", ["large_m", "fp32", "small_vocab", "k8", "strided"])
def test_topk_fallback(case, monkeypatch):
    m, n, k = 12, 65536, 16
    dtype = torch.bfloat16
    if case == "large_m":
        m = 193
    elif case == "fp32":
        dtype = torch.float32
    elif case == "small_vocab":
        n = 16384
    elif case == "k8":
        k = 8
    x = torch.randn(m, n * (2 if case == "strided" else 1), device="cuda", dtype=dtype)
    if case == "strided":
        x = x[:, ::2]
    original = flashinfer.top_k
    calls = []

    def fallback(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(flashinfer, "top_k", fallback)
    if case == "strided":
        # FlashInfer rejects strided input; preserve its existing error.
        with pytest.raises(RuntimeError, match="contiguous"):
            pruned_topk(x, k)
        assert calls == [True]
        return
    values, ids = pruned_topk(x, k)
    expected = original(x, k, sorted=True, deterministic=True)
    assert calls == [True]
    assert torch.equal(values, expected[0])
    assert torch.equal(ids, expected[1])
