"""Exact pruned BF16 vocabulary top-k with ascending token IDs for ties."""

import flashinfer
import torch
import triton
import triton.language as tl


@triton.jit
def _block_max_kernel(
    X,
    C,
    N: tl.constexpr,
    NC: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1)
    i = tile * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + row * N + i, i < N, other=-float("inf"))
    bits = x.to(tl.uint16, bitcast=True).to(tl.int32)
    key = tl.where(bits & 32768 != 0, ~bits & 65535, bits ^ 32768)
    key = tl.where(bits & 32767 == 0, 32768, key)
    key = tl.where(i < N, key, -1)
    mx = tl.max(key, 0)
    first = tl.min(tl.where((i < N) & (key == mx), i, 2147483647), 0)
    code = (mx.to(tl.int64) << 32) + (4294967295 - first.to(tl.int64))
    tl.store(C + row * NC + tile, code)


@triton.jit
def _selected_top_kernel(
    X,
    Max,
    C,
    Field,
    N: tl.constexpr,
    RANK_BITS: tl.constexpr,
    NC: tl.constexpr,
    BLOCK: tl.constexpr,
    K: tl.constexpr,
    BMAX: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    selection = tl.program_id(1)
    offsets = tl.arange(0, BMAX)
    maxima = tl.load(Max + row * NC + offsets, offsets < NC, other=-1)
    # A discarded block has at least K distinct maxima ranked ahead of it;
    # none of its elements can enter the global Top-k.
    tops = tl.topk(maxima, K)
    kk = tl.arange(0, K)
    pivot = tl.min(tops >> 32, 0)
    peak = tl.max(tops >> 32, 0)
    narrow = peak - pivot < (1 << (32 - RANK_BITS))
    if selection == 0:
        tl.store(Field + row, narrow.to(tl.int32))
    chosen = tl.sum(tl.where(kk == selection, tops, 0), 0)
    rank = 4294967295 - (chosen & 4294967295)
    tile = rank // BLOCK
    i = tile * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + row * N + i, i < N, other=-float("inf"))
    bits = x.to(tl.uint16, bitcast=True).to(tl.int32)
    key = tl.where(bits & 32768 != 0, ~bits & 65535, bits ^ 32768)
    key = tl.where(bits & 32767 == 0, 32768, key)
    if narrow:
        word = ((key - pivot).to(tl.uint32) << RANK_BITS) + (
            (1 << RANK_BITS) - 1 - i
        ).to(tl.uint32)
        word = tl.where((i < N) & (key >= pivot), word, 0)
        top = tl.topk(word, K).to(tl.int64)
    else:
        code = (key.to(tl.int64) << 32) + (4294967295 - i.to(tl.int64))
        code = tl.where(i < N, code, -1)
        top = tl.topk(code, K)
    tl.store(C + (row * K + selection) * K + kk, top)


@triton.jit
def _merge_words_kernel(
    X,
    C,
    Field,
    O,
    I,
    N: tl.constexpr,
    RANK_BITS: tl.constexpr,
    K: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    i = tl.arange(0, K * K)
    codes = tl.load(C + row * K * K + i)
    top = tl.topk(codes, K)
    narrow = tl.load(Field + row)
    ids = tl.where(
        narrow != 0,
        (1 << RANK_BITS) - 1 - (top & ((1 << RANK_BITS) - 1)),
        4294967295 - (top & 4294967295),
    )
    vals = tl.load(X + row * N + ids)
    j = tl.arange(0, K)
    tl.store(I + row * K + j, ids)
    tl.store(O + row * K + j, vals)


def pruned_topk(x, k=16, *, sorted=True, deterministic=True):
    block = 1024
    if (
        not sorted
        or not deterministic
        or not x.is_cuda
        or x.dtype != torch.bfloat16
        or (k != 16)
        or (x.ndim != 2)
        or (not x.is_contiguous())
        # Keep at least four times as many blocks as retained candidates.
        or (x.shape[-1] < 65536)
        or (x.shape[0] < 1)
        or (x.shape[0] > 192)
    ):
        return flashinfer.top_k(x, k, sorted=sorted, deterministic=deterministic)
    n = x.shape[-1]
    m = x.shape[0]
    rank_bits = (n - 1).bit_length()
    if rank_bits >= 32:
        return flashinfer.top_k(x, k, sorted=True, deterministic=True)
    nc = triton.cdiv(n, block)
    mx = torch.empty(m, nc, device=x.device, dtype=torch.int64)
    scratch = torch.empty(m, k, k, device=x.device, dtype=torch.int64)
    field = torch.empty(m, device=x.device, dtype=torch.int32)
    out = torch.empty(m, k, device=x.device, dtype=x.dtype)
    ids = torch.empty(m, k, device=x.device, dtype=torch.int64)
    _block_max_kernel[m, nc](x, mx, n, nc, block, num_warps=4)
    _selected_top_kernel[m, k](
        x,
        mx,
        scratch,
        field,
        n,
        rank_bits,
        nc,
        block,
        k,
        triton.next_power_of_2(nc),
        num_warps=4,
    )
    _merge_words_kernel[m,](
        x,
        scratch,
        field,
        out,
        ids,
        n,
        rank_bits,
        k,
        num_warps=4,
    )
    return (out, ids)
