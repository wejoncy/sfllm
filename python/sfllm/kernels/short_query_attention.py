import torch
import triton
import triton.language as tl


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


def _short_query_paged_attention_fwd(
    q, out, k, v, metadata, scale
):
    """Short causal GQA using split KV traversal and output combine."""
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
    return out.view(n, -1)
