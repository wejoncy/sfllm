import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from sfllm.layers.quantization.fp8_kernel import sigmoid_mul_quant_fp8


@triton.jit
def _short_query_paged_attention_tile(
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
def _short_query_paged_attention_kernel(
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
            _short_query_paged_attention_tile(Q, K, V, Table, Lens, Indptr, Partial, Lse, q_stride, q_head_stride, k_stride, k_head_stride, v_stride, v_head_stride, table_stride, TOKENS, Q_HEADS, KV_HEADS, DIM, 4, SPLITS, SCALE, 16, BN)
        else:
            _short_query_paged_attention_tile(Q, K, V, Table, Lens, Indptr, Partial, Lse, q_stride, q_head_stride, k_stride, k_head_stride, v_stride, v_head_stride, table_stride, TOKENS, Q_HEADS, KV_HEADS, DIM, MAX_Q, SPLITS, SCALE, BM, BN)
    else:
        _short_query_paged_attention_tile(Q, K, V, Table, Lens, Indptr, Partial, Lse, q_stride, q_head_stride, k_stride, k_head_stride, v_stride, v_head_stride, table_stride, TOKENS, Q_HEADS, KV_HEADS, DIM, MAX_Q, SPLITS, SCALE, BM, BN)

@triton.jit
def _combine_short_query_paged_attention_kernel(
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


def _short_query_paged_attention_fwd(
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
    _short_query_paged_attention_kernel[batch, kv_heads, splits](
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
    _combine_short_query_paged_attention_kernel[n * heads,](
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
