from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeSparseMoeBlock as ReferenceMoE,
)

from sfllm.kernels.moe import route_tokens
from sfllm.layers.moe import CutlassMoE, FusedMoE
from sfllm.model_loader.model_config import ModelConfig
from sfllm.model_loader.model_loader import TorchDefaultReset, get_model_architecture
from sfllm.models.qwen3_5_moe import (
    Qwen3_5MoeForConditionalGeneration, Qwen3_5SparseMoeBlock,
)


FIXTURE = Path(__file__).parent / "fixtures" / "qwen3_5_moe"
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")


def config(hidden=128, experts=8, top_k=2, intermediate=64, shared=64):
    return SimpleNamespace(
        hidden_size=hidden, num_experts=experts, num_experts_per_tok=top_k,
        moe_intermediate_size=intermediate, shared_expert_intermediate_size=shared,
        hidden_act="silu", _experts_implementation=None, tie_word_embeddings=False,
    )


def test_official_moe_architecture():
    cfg = ModelConfig(str(FIXTURE))
    cls, architecture = get_model_architecture(cfg.hf_config)
    assert architecture == "Qwen3_5MoeForConditionalGeneration"
    assert cls.__module__ == "sfllm.models.qwen3_5_moe"
    assert cfg.dtype == torch.bfloat16
    assert cfg.hf_config.text_config.num_experts == 256
    with pytest.raises(ValueError, match="unquantized"):
        cls(cfg.hf_config, quant_config=object())


def test_unsupported_routing_configuration_fails_at_construction():
    for top_k in (3, 64):
        with pytest.raises(ValueError, match="power-of-two top_k up to 32"):
            FusedMoE(128, 64, 256, top_k)


@pytest.mark.parametrize("backend", ["triton_kernel", "flashinfer_cutlass"])
def test_checkpoint_packed_experts_and_shared_weight_loading(backend):
    model = Qwen3_5MoeForConditionalGeneration.__new__(Qwen3_5MoeForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.config = config(experts=4, top_k=2)
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList([torch.nn.Module()])
    block = model.model.layers[0].mlp = (
        CutlassMoE(128, 64, 4, 2) if backend == "flashinfer_cutlass"
        else Qwen3_5SparseMoeBlock(model.config)
    )
    tensors = {
        "gate.weight": torch.randn(4, 128),
        "experts.gate_up_proj": torch.randn(4, 128, 128),
        "experts.down_proj": torch.randn(4, 128, 64),
        "shared_expert.gate_proj.weight": torch.randn(64, 128),
        "shared_expert.up_proj.weight": torch.randn(64, 128),
        "shared_expert.down_proj.weight": torch.randn(128, 64),
        "shared_expert_gate.weight": torch.randn(1, 128),
    }
    prefix = "model.language_model.layers.0.mlp."
    model.load_weights((prefix + name, value) for name, value in tensors.items())
    if backend == "flashinfer_cutlass":
        expected = {
            "gate.weight": F.pad(torch.cat([
                tensors["gate.weight"], tensors["shared_expert_gate.weight"],
            ]), (0, 0, 0, block.gate.out_features - 5)),
            "up_gate_proj": torch.cat([
                tensors["experts.gate_up_proj"].reshape(4, 2, 64, 128).flip(1).flatten(1, 2),
                torch.cat([tensors["shared_expert.up_proj.weight"], tensors["shared_expert.gate_proj.weight"]])[None],
            ]),
            "down_proj": torch.cat([tensors["experts.down_proj"], tensors["shared_expert.down_proj.weight"][None]]),
        }
    else:
        expected = {name: value for name, value in tensors.items() if not name.startswith("shared_expert.")}
        expected["shared_expert.gate_up_proj.weight"] = torch.cat([
            tensors["shared_expert.gate_proj.weight"], tensors["shared_expert.up_proj.weight"],
        ])
        expected["shared_expert.down_proj.weight"] = tensors["shared_expert.down_proj.weight"]
    for name, value in block.named_parameters():
        torch.testing.assert_close(value, expected[name], atol=0, rtol=0)
    # One packed MLP parameter must not hide a missing gate or up shard.
    with pytest.raises(RuntimeError, match="Missing Qwen3.5 shared expert"):
        model.load_weights((prefix + name, value) for name, value in tensors.items()
                           if name != "shared_expert.gate_proj.weight")


def routed_reference(x, layer):
    scores = F.linear(x, layer.gate.weight).float().softmax(-1)
    ids = torch.argsort(scores, descending=True, stable=True)[:, :layer.top_k]
    weights = scores.gather(1, ids)
    weights = (weights / weights.sum(-1, keepdim=True)).to(x.dtype)
    output = torch.zeros_like(x, dtype=torch.float32)
    for expert in range(layer.experts.num_experts):
        row, choice = torch.where(ids == expert)
        if row.numel():
            gate, up = F.linear(x[row], layer.experts.gate_up_proj[expert]).float().chunk(2, -1)
            activated = (F.silu(gate) * up).to(x.dtype)
            value = F.linear(activated, layer.experts.down_proj[expert])
            output.index_add_(0, row, value.float() * weights[row, choice, None].float())
    return output.to(x.dtype)


@cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("tokens,experts,top_k,hidden,intermediate", [
    (0, 4, 2, 128, 64),
    (1, 256, 8, 2048, 512),
    (37, 7, 2, 144, 80),
    (65, 4, 4, 256, 128),
    (4103, 7, 2, 128, 64),
])
def test_routed_experts_match_independent_gemms(dtype, tokens, experts, top_k, hidden, intermediate):
    torch.manual_seed(51)
    with TorchDefaultReset(dtype, device="cuda"):
        layer = FusedMoE(hidden, intermediate, experts, top_k)
        for param in layer.parameters():
            param.normal_(0, 0.03)
        x = torch.randn(tokens, hidden)
    torch.testing.assert_close(layer(x), routed_reference(x, layer), atol=0.003, rtol=0.02)


@cuda
@pytest.mark.parametrize("shared", [64, 96])
def test_router_and_shared_expert_match_transformers(shared):
    cfg = config(shared=shared)
    torch.manual_seed(53)
    with TorchDefaultReset(torch.bfloat16, device="cuda"):
        candidate = Qwen3_5SparseMoeBlock(cfg)
        reference = ReferenceMoE(cfg).requires_grad_(False)
        for param in candidate.parameters():
            param.normal_(0, 0.04)
        reference.gate.weight.copy_(candidate.gate.weight)
        reference.shared_expert_gate.weight.copy_(candidate.shared_expert_gate.weight)
        reference.experts.gate_up_proj.copy_(candidate.experts.gate_up_proj)
        reference.experts.down_proj.copy_(candidate.experts.down_proj)
        gate, up = candidate.shared_expert.gate_up_proj.weight.chunk(2, 0)
        reference.shared_expert.gate_proj.weight.copy_(gate)
        reference.shared_expert.up_proj.weight.copy_(up)
        reference.shared_expert.down_proj.weight.copy_(candidate.shared_expert.down_proj.weight)
        x = torch.randn(2, 11, 128)
    with torch.inference_mode():
        torch.testing.assert_close(candidate(x.view(-1, 128)).view_as(x), reference(x), atol=0.003, rtol=0.03)


@cuda
def test_routing_ties_and_normalization():
    x = torch.ones((3, 128), device="cuda", dtype=torch.bfloat16)
    router = torch.zeros((8, 128), device="cuda", dtype=torch.bfloat16)
    routing = route_tokens(x, router, 4)
    torch.testing.assert_close(routing.indx[:3].long(), torch.tensor([[0, 1, 2, 3]] * 3, device="cuda"))
    torch.testing.assert_close(routing.vals[:3], torch.full((3, 4), 0.25, device="cuda", dtype=x.dtype), atol=0, rtol=0)
    assert routing.mask_metadata.col_sum.sum().item() == 3 * 4


@cuda
def test_complete_moe_graph_replay_and_concurrent_streams():
    torch.manual_seed(52)
    with TorchDefaultReset(torch.bfloat16, device="cuda"):
        layer = Qwen3_5SparseMoeBlock(config())
        for param in layer.parameters():
            param.normal_(0, 0.03)
        x = torch.randn(17, 128)
        other = torch.randn_like(x)
    layer(x)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = layer(x)
    initial_ids = route_tokens(x, layer.gate.weight, layer.top_k).indx[:len(x)].clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    for _ in range(2):
        x.normal_()
        graph.replay()
        with torch.cuda.stream(stream):
            concurrent = layer(other)
        expected = routed_reference(x, layer)
        expected += layer.shared_expert(x) * torch.sigmoid(layer.shared_expert_gate(x))
        torch.testing.assert_close(actual, expected, atol=0.003, rtol=0.03)
        torch.cuda.current_stream().wait_stream(stream)
        torch.testing.assert_close(concurrent, layer(other), atol=0, rtol=0)
    assert not torch.equal(initial_ids, route_tokens(x, layer.gate.weight, layer.top_k).indx[:len(x)])


@cuda
def test_checkpoint_geometry_decode_graph_and_long_prefill():
    with TorchDefaultReset(torch.bfloat16, device="cuda"):
        torch.manual_seed(54)
        layer = FusedMoE(2048, 512, 256, 8)
        for param in layer.parameters():
            param.normal_(0, 0.02)
        for tokens in (16, 4103):
            x = torch.randn(tokens, 2048)
            actual = layer(x)
            if tokens == 16:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = layer(x)
                graph.replay()
            expected = routed_reference(x, layer)
            torch.testing.assert_close(actual, expected, atol=0.008, rtol=0.03)
            relative_error = (actual.float() - expected.float()).norm() / expected.float().norm()
            assert relative_error < 0.01


@pytest.mark.parametrize("backend,major,shared,expected", [
    ("auto", 9, 64, CutlassMoE),
    ("auto", 10, 64, Qwen3_5SparseMoeBlock),
    ("auto", 9, 96, Qwen3_5SparseMoeBlock),
    ("triton_kernel", 9, 64, Qwen3_5SparseMoeBlock),
    ("flashinfer_cutlass", 9, 64, CutlassMoE),
])
def test_moe_backend_selection(monkeypatch, backend, major, shared, expected):
    import sfllm.models.qwen3_5_moe as module

    monkeypatch.setattr(module, "get_global_server_args", lambda: SimpleNamespace(moe_runner_backend=backend))
    monkeypatch.setattr(module, "_cutlass_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (major, 0))
    mlp, quant_methods = module.Qwen3_5MoeDecoderLayer.build_mlp(None, config(shared=shared), None, "mlp")
    assert isinstance(mlp, expected)
    assert quant_methods == ()


def test_moe_auto_without_flashinfer(monkeypatch):
    import sfllm.models.qwen3_5_moe as module

    args = SimpleNamespace(moe_runner_backend="auto")
    monkeypatch.setattr(module, "get_global_server_args", lambda: args)
    monkeypatch.setattr(module, "_cutlass_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (9, 0))
    mlp, _ = module.Qwen3_5MoeDecoderLayer.build_mlp(None, config(), None, "mlp")
    assert isinstance(mlp, Qwen3_5SparseMoeBlock)
    args.moe_runner_backend = "flashinfer_cutlass"
    with pytest.raises(ImportError, match="sfllm\\[flashinfer\\]"):
        module.Qwen3_5MoeDecoderLayer.build_mlp(None, config(), None, "mlp")


def cutlass_reference(x, layer):
    # Use the packed router GEMM, independently followed by PyTorch selection
    # and one GEMM per expert. Separate E- and E+1-row BF16 router GEMMs can
    # round near-tied logits differently and thus select different experts.
    logits = F.linear(x, layer.gate.weight).float()
    ids = torch.argsort(logits[:, :layer.num_experts], descending=True, stable=True)[:, :layer.top_k]
    weights = logits[:, :layer.num_experts].gather(1, ids).softmax(-1).to(x.dtype).float()
    ids = torch.cat([ids, torch.full_like(ids[:, :1], layer.num_experts)], -1)
    shared_logits = logits[:, layer.num_experts:layer.num_experts + 1]
    weights = torch.cat([weights, shared_logits.sigmoid().to(x.dtype).float()], -1)
    output = torch.zeros_like(x, dtype=torch.float32)
    for expert in range(layer.num_experts + 1):
        row, choice = torch.where(ids == expert)
        if row.numel():
            up, gate = F.linear(x[row], layer.up_gate_proj[expert]).float().chunk(2, -1)
            value = F.linear((F.silu(gate) * up).to(x.dtype), layer.down_proj[expert])
            output.index_add_(0, row, value.float() * weights[row, choice, None])
    return output.to(x.dtype)


@cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("tokens,experts,top_k,hidden,intermediate", [
    (0, 4, 2, 128, 64),
    (37, 7, 2, 144, 80),
    (16, 256, 8, 2048, 512),
    (4103, 256, 8, 2048, 512),
])
def test_cutlass_shared_and_routed_match_independent_gemms(dtype, tokens, experts, top_k, hidden, intermediate):
    with TorchDefaultReset(dtype, device="cuda"):
        torch.manual_seed(57)
        layer = CutlassMoE(hidden, intermediate, experts, top_k)
        for param in layer.parameters():
            param.normal_(0, 0.02)
        x = torch.randn(tokens, hidden)
    actual = layer(x)
    expected = cutlass_reference(x, layer)
    torch.testing.assert_close(actual, expected, atol=0.008, rtol=0.03)
    if tokens:
        assert (actual.float() - expected.float()).norm() / expected.float().norm() < 0.01


@cuda
def test_cutlass_graph_changing_routes_and_stream_scratch():
    from sfllm.kernels.moe import route_with_shared

    with TorchDefaultReset(torch.bfloat16, device="cuda"):
        torch.manual_seed(58)
        layer = CutlassMoE(128, 64, 8, 2)
        for param in layer.parameters():
            param.normal_(0, 0.03)
        x = torch.randn(17, 128)
        other = torch.randn_like(x)
    layer(x)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = layer(x)
    initial_ids = route_with_shared(x, layer.gate.weight, layer.top_k, layer.num_experts)[0].clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    for _ in range(2):
        x.normal_()
        graph.replay()
        with torch.cuda.stream(stream):
            concurrent = layer(other)
        expected = cutlass_reference(x, layer)
        torch.testing.assert_close(actual, expected, atol=0.003, rtol=0.03)
        torch.cuda.current_stream().wait_stream(stream)
        torch.testing.assert_close(concurrent, layer(other), atol=0, rtol=0)
    assert not torch.equal(initial_ids, route_with_shared(x, layer.gate.weight, layer.top_k, layer.num_experts)[0])
    scratch_ptrs = [buffer.data_ptr() for buffer in layer.backend.scratch.values()]
    assert len(set(scratch_ptrs)) == len(scratch_ptrs) >= 2


@cuda
def test_cutlass_router_excludes_alignment_padding():
    from sfllm.kernels.moe import route_with_shared

    with TorchDefaultReset(torch.bfloat16, device="cuda"):
        layer = CutlassMoE(128, 64, 7, 2)
        layer.gate.weight.zero_()
        layer.gate.weight[layer.num_experts].fill_(0.125)
        # Even dominant padding logits must never enter top-k or its softmax.
        layer.gate.weight[layer.num_experts + 1:].fill_(100)
        x = torch.ones(3, 128)
    ids, weights = route_with_shared(x, layer.gate.weight, layer.top_k, layer.num_experts)
    torch.testing.assert_close(ids, torch.tensor([[0, 1, 7]] * 3, device=x.device, dtype=torch.int32), atol=0, rtol=0)
    expected_weights = torch.tensor([[0.5, 0.5, 0.0]] * 3, device=x.device)
    expected_weights[:, -1] = torch.sigmoid(torch.tensor(16.0, device=x.device))
    torch.testing.assert_close(weights, expected_weights, atol=0, rtol=0)
