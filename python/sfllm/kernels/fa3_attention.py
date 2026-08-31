import torch
import triton
import triton.language as tl
from sgl_kernel.flash_attn import flash_attn_with_kvcache


@triton.jit
def _build_page_table_kernel(
    kv_indices,
    kv_indptr,
    out_cache_loc,
    qo_indptr,
    page_table,
    cache_seqlens,
    page_table_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    APPEND_QUERY: tl.constexpr,
):
    batch_idx = tl.program_id(0)
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

    tl.store(cache_seqlens + batch_idx, total_len)
    for offset in range(0, total_len, BLOCK_SIZE):
        offsets = offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total_len
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
    _build_page_table_kernel[(page_table.shape[0],)](
        kv_indices,
        kv_indptr,
        out_cache_loc,
        qo_indptr,
        page_table,
        cache_seqlens,
        page_table.stride(0),
        BLOCK_SIZE=256,
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
) -> torch.Tensor:
    flash_attn_with_kvcache(
        q,
        k_buffer.view(k_buffer.shape[0], 1, k_buffer.shape[1], k_buffer.shape[2]),
        v_buffer.view(v_buffer.shape[0], 1, v_buffer.shape[1], v_buffer.shape[2]),
        cache_seqlens=metadata.cache_seqlens,
        page_table=metadata.page_table,
        cu_seqlens_q=metadata.qo_indptr,
        max_seqlen_q=metadata.max_query_len,
        softmax_scale=softmax_scale,
        causal=causal,
        num_splits=1,
        scheduler_metadata=metadata.scheduler_metadata,
        out=out,
    )
    return out.view(out.shape[0], -1)
