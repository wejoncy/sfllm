import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from sgl_kernel.flash_attn import flash_attn_with_kvcache
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


@triton.jit
def _short_paged_attention_tile(
    Q, K, V, Table, Lens, Indptr, Partial, Lse,
    q_stride: tl.constexpr,
    q_head_stride: tl.constexpr,
    k_stride: tl.constexpr,
    k_head_stride: tl.constexpr,
    v_stride: tl.constexpr,
    v_head_stride: tl.constexpr,
    table_stride: tl.constexpr,
    TOKENS: tl.constexpr,
    Q_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    DIM: tl.constexpr,
    MAX_Q: tl.constexpr,
    SPLITS: tl.constexpr,
    SCALE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    req = tl.program_id(0)
    group: tl.constexpr = Q_HEADS // KV_HEADS
    kh = tl.program_id(1)
    split = tl.program_id(2)
    q_begin = tl.load(Indptr + req)
    q_end = tl.load(Indptr + req + 1)
    q_len = q_end - q_begin
    kv_len = tl.load(Lens + req)
    if q_len <= 0:
        return
    m = tl.arange(0, BM)
    h = m // MAX_Q
    qi = m % MAX_Q
    dim = tl.arange(0, DIM)
    q = tl.load(
        Q + (q_begin + qi[:, None]) * q_stride
        + (kh * group + h[:, None]) * q_head_stride + dim[None, :],
        mask=(h[:, None] < group) & (qi[:, None] < q_len),
        other=0,
    )
    chunk = tl.cdiv(kv_len, SPLITS * BN) * BN
    begin = split * chunk
    end = tl.minimum(begin + chunk, kv_len)
    running_max = tl.full((BM,), -float("inf"), tl.float32)
    running_sum = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, DIM), tl.float32)
    n = tl.arange(0, BN)
    for offset in range(begin, end, BN):
        pos = offset + n
        loc = tl.load(
            Table + req * table_stride + pos, mask=pos < end, other=0
        ).to(tl.int64)
        k = tl.load(
            K + loc[None, :] * k_stride + kh * k_head_stride + dim[:, None],
            mask=pos[None, :] < end,
            other=0,
        )
        score = tl.dot(q, k) * SCALE
        valid = (
            (pos[None, :] < end)
            & (pos[None, :] <= kv_len - q_len + qi[:, None])
            & (qi[:, None] < q_len) & (h[:, None] < group)
        )
        score = tl.where(valid, score, -float("inf"))
        next_max = tl.maximum(running_max, tl.max(score, 1))
        safe_max = tl.where(next_max == -float("inf"), 0.0, next_max)
        prob = tl.exp(score - safe_max[:, None])
        factor = tl.exp(running_max - safe_max)
        v = tl.load(
            V + loc[:, None] * v_stride + kh * v_head_stride + dim[None, :],
            mask=pos[:, None] < end,
            other=0,
        )
        acc = acc * factor[:, None] + tl.dot(prob.to(v.dtype), v)
        running_sum = running_sum * factor + tl.sum(prob, 1)
        running_max = next_max
    token = q_begin + qi
    head = kh * group + h
    mask = (qi < q_len) & (h < group)
    output_base = (split * TOKENS + token) * Q_HEADS + head
    inv_sum = 1.0 / tl.where(running_sum > 0, running_sum, 1.0)
    tl.store(
        Partial + output_base[:, None] * DIM + dim[None, :],
        acc * inv_sum[:, None],
        mask=mask[:, None],
    )
    logsum = running_max + tl.log(running_sum)
    tl.store(Lse + output_base, logsum, mask=mask)


@triton.jit
def _short_paged_attention_kernel(
    Q, K, V, Table, Lens, Indptr, Partial, Lse,
    q_stride: tl.constexpr,
    q_head_stride: tl.constexpr,
    k_stride: tl.constexpr,
    k_head_stride: tl.constexpr,
    v_stride: tl.constexpr,
    v_head_stride: tl.constexpr,
    table_stride: tl.constexpr,
    TOKENS: tl.constexpr,
    Q_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    DIM: tl.constexpr,
    MAX_Q: tl.constexpr,
    SPLITS: tl.constexpr,
    SCALE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    ENABLE_PDL: tl.constexpr = False,
):
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    req = tl.program_id(0)
    length = tl.load(Indptr + req + 1) - tl.load(Indptr + req)
    if MAX_Q == 8 and Q_HEADS // KV_HEADS == 4:
        if length <= 4:
            _short_paged_attention_tile(Q, K, V, Table, Lens, Indptr, Partial, Lse, q_stride, q_head_stride, k_stride, k_head_stride, v_stride, v_head_stride, table_stride, TOKENS, Q_HEADS, KV_HEADS, DIM, 4, SPLITS, SCALE, 16, BN)
        else:
            _short_paged_attention_tile(Q, K, V, Table, Lens, Indptr, Partial, Lse, q_stride, q_head_stride, k_stride, k_head_stride, v_stride, v_head_stride, table_stride, TOKENS, Q_HEADS, KV_HEADS, DIM, MAX_Q, SPLITS, SCALE, BM, BN)
    else:
        _short_paged_attention_tile(Q, K, V, Table, Lens, Indptr, Partial, Lse, q_stride, q_head_stride, k_stride, k_head_stride, v_stride, v_head_stride, table_stride, TOKENS, Q_HEADS, KV_HEADS, DIM, MAX_Q, SPLITS, SCALE, BM, BN)

@triton.jit
def _combine_short_paged_attention_kernel(
    Partial, Lse, Out, Indptr,
    ROWS: tl.constexpr,
    Q_HEADS: tl.constexpr,
    REQS: tl.constexpr,
    DIM: tl.constexpr,
    O_STRIDE: tl.constexpr,
    O_HEAD_STRIDE: tl.constexpr,
    SPLITS: tl.constexpr,
    BS: tl.constexpr,
    ENABLE_PDL: tl.constexpr = False,
):
    if ENABLE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    row = tl.program_id(0)
    valid_row = row < tl.load(Indptr + REQS) * Q_HEADS
    split = tl.arange(0, BS)
    dim = tl.arange(0, DIM)
    lse = tl.load(
        Lse + split * ROWS + row,
        mask=(split < SPLITS) & valid_row,
        other=-float("inf"),
    )
    peak = tl.max(lse, 0)
    peak = tl.where(peak == -float("inf"), 0.0, peak)
    weights = tl.exp(lse - peak)
    total = tl.sum(weights, 0)
    weights = weights / tl.where(total > 0, total, 1.0)
    values = tl.load(
        Partial + (split[:, None] * ROWS + row) * DIM + dim[None, :],
        mask=(split[:, None] < SPLITS) & valid_row,
        other=0,
    )
    out = tl.sum(values * weights[:, None], 0)
    tl.store(Out + row // Q_HEADS * O_STRIDE + row % Q_HEADS * O_HEAD_STRIDE + dim, out)


@gluon.jit
def _combine_gate_quant_attention_kernel(
    partial,
    log_sums,
    gate_input,
    output,
    row_scale,
    cu_seqlens,
    rows: gl.constexpr,
    num_heads: gl.constexpr,
    num_requests: gl.constexpr,
    head_dim: gl.constexpr,
    stride_gate_token: gl.constexpr,
    stride_gate_head: gl.constexpr,
    stride_gate_dim: gl.constexpr,
    splits: gl.constexpr,
    block_splits: gl.constexpr,
    block_heads: gl.constexpr,
    layout_warps: gl.constexpr,
    enable_pdl: gl.constexpr,
):
    token = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout([1, 1, 4], [1, 4, 8], [layout_warps, 1, 1], [2, 1, 0])
    heads = gl.arange(0, block_heads, layout=gl.SliceLayout(1, gl.SliceLayout(2, layout)))
    split_offsets = gl.arange(0, block_splits, layout=gl.SliceLayout(0, gl.SliceLayout(2, layout)))
    dim_offsets = gl.arange(0, head_dim, layout=gl.SliceLayout(0, gl.SliceLayout(1, layout)))
    out_layout: gl.constexpr = gl.SliceLayout(1, layout)
    output_heads = gl.arange(0, block_heads, layout=gl.SliceLayout(1, out_layout))
    output_dims = gl.arange(0, head_dim, layout=gl.SliceLayout(0, out_layout))
    if enable_pdl:
        gl.inline_asm_elementwise(
            "griddepcontrol.wait; // $0", "=r", [], dtype=gl.int32, is_pure=False, pack=1
        )
        gl.inline_asm_elementwise(
            "griddepcontrol.launch_dependents; // $0",
            "=r",
            [],
            dtype=gl.int32,
            is_pure=False,
            pack=1,
        )
    valid_token = token < gl.load(cu_seqlens + num_requests)
    gate = gl.load(
        gate_input
        + token * stride_gate_token
        + output_heads[:, None] * stride_gate_head
        + output_dims[None, :] * stride_gate_dim,
        mask=(output_heads[:, None] < num_heads) & valid_token,
        other=0,
    ).to(gl.float32)
    row = token * num_heads + heads
    log_sums_value = gl.load(
        log_sums + split_offsets[None, :] * rows + row[:, None],
        mask=(split_offsets[None, :] < splits) & (heads[:, None] < num_heads) & valid_token,
        other=-float("inf"),
    )
    peak = gl.max(log_sums_value, 1)
    peak = gl.where(peak == -float("inf"), 0.0, peak)
    weights = gl.exp(log_sums_value - peak[:, None])
    total = gl.sum(weights, 1)
    weights = weights / gl.where(total > 0, total, 1.0)[:, None]
    values = gl.load(
        partial
        + (split_offsets[None, :, None] * rows + row[:, None, None]) * head_dim
        + dim_offsets[None, None, :],
        mask=(split_offsets[None, :, None] < splits)
        & (heads[:, None, None] < num_heads)
        & valid_token,
        other=0,
    )
    value = gl.sum(values * weights[:, :, None], 1).to(gl.bfloat16).to(gl.float32)
    value = value * (1.0 / (1.0 + gl.exp(-gate)))
    value = value.to(gl.bfloat16).to(gl.float32)
    value = gl.where(output_heads[:, None] < num_heads, value, 0.0)
    maximum = gl.max(gl.max(gl.abs(value), 1), 0)
    scale = gl.div_rn(maximum, 448.0)
    inverse_scale = gl.where(scale == 0, 0.0, gl.div_rn(1.0, scale))
    quant = gl.minimum(gl.maximum(value * inverse_scale, -448.0), 448.0)
    gl.store(row_scale + token, scale)
    gl.store(
        output
        + token * num_heads * head_dim
        + output_heads[:, None] * head_dim
        + output_dims[None, :],
        quant,
        mask=output_heads[:, None] < num_heads,
    )


def _short_paged_attention_fwd(
    q, out, k, v, metadata, scale, *, output_gate=None, output_scale=None
):
    """Short causal GQA, with optional combine/gate/row quantization fusion."""
    n, heads, dim = q.shape
    kv_heads = k.shape[1]
    max_q = metadata.max_query_len
    batch = metadata.cache_seqlens.numel()
    sm_count = torch.cuda.get_device_properties(q.device).multi_processor_count
    # About two resident CTAs per SM balances traversal work and split overhead.
    splits = min(
        8, max(4, triton.next_power_of_2(triton.cdiv(2 * sm_count, batch * kv_heads)))
    )
    partial = torch.empty((splits, n, heads, dim), device=q.device, dtype=torch.float32)
    lse = torch.empty((splits, n, heads), device=q.device, dtype=torch.float32)
    enable_pdl = 1 <= n <= 192
    _short_paged_attention_kernel[batch, kv_heads, splits](
        q,
        k,
        v,
        metadata.page_table,
        metadata.cache_seqlens,
        metadata.qo_indptr,
        partial,
        lse,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        metadata.page_table.stride(0),
        n,
        heads,
        kv_heads,
        dim,
        max_q,
        splits,
        scale,
        max(16, triton.next_power_of_2(max_q * (heads // kv_heads))),
        128,
        ENABLE_PDL=enable_pdl,
        num_warps=4,
        num_stages=3,
        launch_pdl=enable_pdl,
    )
    if output_gate is not None and output_scale is None and (heads <= 32):
        quant = torch.empty((n, heads * dim), device=q.device, dtype=torch.float8_e4m3fn)
        quant_scale = torch.empty((n, 1), device=q.device, dtype=torch.float32)
        gst = output_gate.stride(0)
        gsh = output_gate.stride(1) if output_gate.ndim == 3 else dim * output_gate.stride(-1)
        gsd = output_gate.stride(-1)
        warps = 16
        _combine_gate_quant_attention_kernel[n,](
            partial,
            lse,
            output_gate,
            quant,
            quant_scale,
            metadata.qo_indptr,
            n * heads,
            heads,
            metadata.cache_seqlens.numel(),
            dim,
            gst,
            gsh,
            gsd,
            splits,
            triton.next_power_of_2(splits),
            triton.next_power_of_2(heads),
            warps,
            enable_pdl,
            num_warps=warps,
            enable_fp_fusion=False,
            launch_pdl=enable_pdl,
        )
        return (quant, quant_scale)
    _combine_short_paged_attention_kernel[n * heads,](
        partial,
        lse,
        out,
        metadata.qo_indptr,
        n * heads,
        heads,
        metadata.cache_seqlens.numel(),
        dim,
        out.stride(0),
        out.stride(1),
        splits,
        triton.next_power_of_2(splits),
        ENABLE_PDL=enable_pdl,
        num_warps=4,
        launch_pdl=enable_pdl,
    )
    if output_gate is not None:
        return sigmoid_mul_quant_fp8(out.view(n, -1), output_gate, output_scale)
    return out.view(n, -1)

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
    # Hopper short-query GQA benefits from reusing KV across query heads.
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
        and torch.cuda.get_device_capability(q.device)[0] == 9
    ):
        return _short_paged_attention_fwd(
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
