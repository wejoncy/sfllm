import torch
import triton
import triton.language as tl
from sgl_kernel.flash_attn import flash_attn_with_kvcache

from sfllm.kernels.short_query_attention import _short_query_paged_attention_fwd
from sfllm.layers.quantization.fp8_kernel import sigmoid_mul_quant_fp8


@triton.jit
def _build_page_table_kernel(
    kv_indices,
    kv_indptr,
    out_cache_loc,
    qo_indptr,
    page_table,
    cache_seqlens,
    page_table_stride: tl.constexpr,
    page_table_capacity: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    APPEND_QUERY: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    prefix_start = tl.load(kv_indptr + batch_idx).to(tl.int64)
    prefix_end = tl.load(kv_indptr + batch_idx + 1).to(tl.int64)
    prefix_len = prefix_end - prefix_start
    if APPEND_QUERY:
        query_start = tl.load(qo_indptr + batch_idx).to(tl.int64)
        query_end = tl.load(qo_indptr + batch_idx + 1).to(tl.int64)
        query_len = query_end - query_start
        total_len = prefix_len + query_len
    else:
        total_len = prefix_len

    if block_idx == 0:
        tl.store(cache_seqlens + batch_idx, total_len)
    offset = block_idx * BLOCK_SIZE
    if offset >= total_len:
        return
    offsets = offset + tl.arange(0, BLOCK_SIZE)
    mask = (offsets < total_len) & (offsets < page_table_capacity)
    if APPEND_QUERY:
        from_prefix = offsets < prefix_len
        prefix_offsets = tl.minimum(offsets, prefix_len - 1)
        query_offsets = tl.maximum(offsets - prefix_len, 0)
        prefix_locs = tl.load(
            kv_indices + prefix_start + prefix_offsets,
            mask=mask & from_prefix,
            other=0,
        )
        query_locs = tl.load(
            out_cache_loc + query_start + query_offsets,
            mask=mask & ~from_prefix,
            other=0,
        )
        locs = tl.where(from_prefix, prefix_locs, query_locs)
    else:
        locs = tl.load(kv_indices + prefix_start + offsets, mask=mask, other=0)
    tl.store(
        page_table + batch_idx * page_table_stride + offsets,
        locs.to(tl.int32),
        mask=mask,
    )


def build_page_table(
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    out_cache_loc: torch.Tensor,
    qo_indptr: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    *,
    append_query: bool,
) -> None:
    _build_page_table_kernel[
        (page_table.shape[0], triton.cdiv(page_table.shape[1], 1024))
    ](
        kv_indices,
        kv_indptr,
        out_cache_loc,
        qo_indptr,
        page_table,
        cache_seqlens,
        page_table.stride(0),
        page_table_capacity=page_table.shape[1],
        BLOCK_SIZE=1024,
        APPEND_QUERY=append_query,
        num_warps=4,
    )


def fa3_attention_fwd(
    q: torch.Tensor,
    out: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    metadata,
    softmax_scale: float,
    causal: bool,
    *,
    layer_id: int,
    logit_cap: float = 0.0,
    return_softmax_lse: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    output_gate: torch.Tensor = None,
    output_scale: torch.Tensor = None,
) -> torch.Tensor:
    # Short-query GQA benefits from reusing KV across query heads.
    # Keep FA3's persistent scheduler for small batches and other layouts.
    if (
        1 <= metadata.max_query_len <= 8
        and metadata.cache_seqlens.numel() >= 8
        and causal
        and window_size == (-1, -1)
        and logit_cap == 0
        and not return_softmax_lse
        and q.dtype == k_buffer.dtype == v_buffer.dtype == torch.bfloat16
        and q.shape[2] == k_buffer.shape[2] == v_buffer.shape[2] == 256
        and q.shape[1] == k_buffer.shape[1] * 4
        and v_buffer.shape[1] == k_buffer.shape[1]
        and q.stride(-1) == k_buffer.stride(-1)
        == v_buffer.stride(-1) == out.stride(-1) == 1
        and torch.cuda.get_device_capability(q.device)[0] >= 9
    ):
        return _short_query_paged_attention_fwd(
            q, out, k_buffer, v_buffer, metadata, softmax_scale,
            output_gate=output_gate, output_scale=output_scale,
        )
    result = flash_attn_with_kvcache(
        q,
        k_buffer.view(k_buffer.shape[0], 1, k_buffer.shape[1], k_buffer.shape[2]),
        v_buffer.view(v_buffer.shape[0], 1, v_buffer.shape[1], v_buffer.shape[2]),
        cache_seqlens=metadata.cache_seqlens,
        page_table=metadata.page_table,
        cu_seqlens_q=metadata.qo_indptr,
        max_seqlen_q=metadata.max_query_len,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        softcap=logit_cap,
        num_splits=0,
        scheduler_metadata=metadata.scheduler_metadata[metadata.layer_index_mapping[layer_id]],
        return_softmax_lse=return_softmax_lse,
        out=out,
    )
    if return_softmax_lse:
        return result
    if output_gate is not None:
        return sigmoid_mul_quant_fp8(out.view(out.shape[0], -1), output_gate, output_scale)
    return out.view(out.shape[0], -1)
