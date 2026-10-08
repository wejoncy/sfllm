"""FlashInfer fast path and vendored Triton MoE primitives."""

import torch
from torch.nn import functional as F


def route_tokens(x, router_weight, top_k):
    from sfllm.kernels.moe_triton import topk

    # Softmax over the selected logits equals full softmax followed by top-k
    # renormalization. The library computes in FP32 and stores activation dtype.
    logits = F.linear(x, router_weight)
    tokens = x.shape[0]
    # The library specializes bitmatrix strides on its row capacity. Bucket
    # routing storage to avoid compiling a new kernel for every prefill length.
    # n_rows excludes padding from expert counts; model inputs remain unchanged.
    capacity = 1 << (tokens - 1).bit_length()
    logits = F.pad(logits, (0, 0, 0, capacity - tokens))
    return topk(logits, top_k, apply_softmax=True, n_rows=tokens)


def fused_experts(x, routing, gate_up, down):
    """Grouped expert GEMMs in Qwen's [gate, up] checkpoint weight layout."""
    from sfllm.kernels.moe_triton import make_ragged_tensor_metadata, reduce_forward
    from sfllm.kernels.moe_triton.matmul import matmul

    tokens, hidden = x.shape
    capacity = routing.indx.shape[0]
    top_k = routing.indx.shape[1]
    index = routing.mask_metadata.col_sorted_indx
    metadata = make_ragged_tensor_metadata(routing.mask_metadata.col_sum, capacity * top_k)
    projected = matmul(
        x, gate_up.transpose(1, 2), None,
        a_ragged_metadata=metadata, gather_indx=index // top_k,
    )
    activated = torch.empty(
        (capacity * top_k, down.shape[2]), dtype=x.dtype, device=x.device,
    )
    torch.ops.sfkernels.silu_and_mul(activated, projected)
    expert_output = matmul(
        activated, down.transpose(1, 2), None,
        a_ragged_metadata=metadata, scatter_indx=index,
    )
    output, _ = reduce_forward(
        expert_output.view(capacity, top_k, hidden)[:tokens], dim=1,
        # reduce_forward expresses broadcasting through zero strides.
        scale=routing.vals[:tokens].unsqueeze(-1).expand(tokens, top_k, hidden),
    )
    return output


@torch.compile(dynamic=True)
def _append_shared_expert(ids, weights, shared_logits, expert):
    """TorchInductor fuses packing and sigmoid; no handwritten routing kernel."""
    shared_ids = torch.full_like(ids[:, :1], expert)
    return (
        torch.cat((ids, shared_ids), dim=1).int(),
        torch.cat((weights, shared_logits.sigmoid()), dim=1).float(),
    )


def route_with_shared(x, router_weight, top_k, num_experts):
    from sfllm.kernels.moe_triton import topk_forward

    tokens = len(x)
    capacity = 1 << (tokens - 1).bit_length()
    # Write directly into bucketed storage. The library masks unused rows;
    # its strided input also lets us exclude the shared/padding columns by view.
    logits = torch.empty(
        (capacity, router_weight.shape[0]), device=x.device, dtype=x.dtype,
    )
    F.linear(x, router_weight, out=logits[:tokens])
    weights, ids, _ = topk_forward(logits[:, :num_experts], top_k, n_rows=tokens)
    return _append_shared_expert(
        ids[:tokens], weights[:tokens],
        logits[:tokens, num_experts:num_experts + 1], num_experts,
    )


class CutlassBackend:
    """One model owns library scratch, separately for each stream and capacity."""

    buckets = (1, 8, 16, 64, 256, 1024, 4096, 8192)

    def __init__(self):
        self.scratch = {}

    def workspace(self, x, up_gate, down, top_k):
        from flashinfer.fused_moe import cutlass_fused_moe_workspace_size

        capacity = 1 << (len(x) - 1).bit_length()
        key = (x.device, torch.cuda.current_stream(x.device).cuda_stream, capacity)
        if key not in self.scratch:
            size = cutlass_fused_moe_workspace_size(
                capacity, x.shape[1], down.shape[2], down.shape[0], top_k,
                x_dtype=x.dtype, weight_dtype=up_gate.dtype, output_dtype=x.dtype,
                use_fused_finalize=True, device=x.device,
            )
            self.scratch[key] = torch.empty(size, device=x.device, dtype=torch.uint8)
        return self.scratch[key]

    def prepare(self, up_gate, down, top_k):
        from pathlib import Path
        import flashinfer
        from flashinfer.autotuner import autotune
        from flashinfer.fused_moe import cutlass_fused_moe

        major, minor = torch.cuda.get_device_capability(up_gate.device)
        cache = Path.home() / ".cache" / "flashinfer" / (
            f"sfllm-moe-{flashinfer.__version__}-sm{major}{minor}-"
            f"{up_gate.dtype}-{down.shape[0]}-{down.shape[1]}-{down.shape[2]}-{top_k}.json"
        )
        cache.parent.mkdir(parents=True, exist_ok=True)
        x = torch.zeros((1, down.shape[1]), device=up_gate.device, dtype=up_gate.dtype)
        ids = torch.arange(top_k, device=x.device, dtype=torch.int32).view(1, -1)
        scales = torch.full((1, top_k), 1 / top_k, device=x.device, dtype=torch.float32)
        with autotune(True, cache=str(cache), tuning_buckets=self.buckets):
            cutlass_fused_moe(
                x, ids, scales, up_gate, down, x.dtype, [],
                use_fused_finalize=True, enable_pdl=False,
            )
        self.cache_path = cache

    def __call__(self, x, router_weight, up_gate, down, top_k):
        from flashinfer.autotuner import autotune
        from flashinfer.fused_moe import cutlass_fused_moe

        ids, weights = route_with_shared(x, router_weight, top_k, down.shape[0] - 1)
        output = torch.empty_like(x)
        scratch = self.workspace(x, up_gate, down, top_k + 1)
        with autotune(False, tuning_buckets=self.buckets):
            cutlass_fused_moe(
                x, ids, weights, up_gate, down, x.dtype, [], output=output,
                use_fused_finalize=True, enable_pdl=False, workspace_buffer=scratch,
            )
        return output
