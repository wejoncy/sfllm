"""Fixed-budget verification from shared draft prefix scores."""

import torch
import triton
import triton.language as tl


@triton.jit
def _allocate_and_pack_kernel(
    prefix_logprobs, candidates, lengths, boundaries, source, packed,
    SCORE_ROW: tl.constexpr, SCORE_COL: tl.constexpr,
    TOKEN_ROW: tl.constexpr, TOKEN_COL: tl.constexpr,
    B: tl.constexpr, WIDTH: tl.constexpr, BUDGET: tl.constexpr,
    RB: tl.constexpr, CW: tl.constexpr,
):
    row = tl.arange(0, RB)
    col = tl.arange(0, CW)
    extra = BUDGET - B
    valid = (row[:, None] < B) & (col[None, :] < WIDTH)
    eligible = valid & (col[None, :] > 0)
    prefix = tl.load(
        prefix_logprobs + row[:, None] * SCORE_ROW
        + (col[None, :] - 1) * SCORE_COL,
        eligible, other=0,
    ).to(tl.float32)
    # Anchors are mandatory; additional prefix positions compete for the budget.
    score = tl.where(eligible, prefix, -float("inf"))
    flat = tl.reshape(score, (RB * CW,))
    order = tl.arange(0, RB * CW)
    if extra > 0:
        sorted_scores = tl.sort(flat, descending=True)
        cutoff = tl.sum(tl.where(order == extra - 1, sorted_scores, 0.0))
        above = flat > cutoff
        equal = (flat == cutoff) & tl.reshape(eligible, (RB * CW,))
        remaining = extra - tl.sum(above.to(tl.int32), axis=0)
        # Stable row-major ties preserve contiguous prefixes, including p=0/1.
        equal_rank = tl.cumsum(equal.to(tl.int32), axis=0)
        chosen = above | (equal & (equal_rank <= remaining))
        counts = tl.sum(tl.reshape(chosen.to(tl.int32), (RB, CW)), axis=1)
    else:
        counts = tl.full((RB,), 0, tl.int32)
    n = tl.where(row < B, counts + 1, 0)
    end = tl.cumsum(n, axis=0)
    begin = end - n
    tl.store(lengths + row, n, row < B)
    tl.store(boundaries + row, begin, row < B)
    tl.store(boundaries + B, BUDGET)
    dst = begin[:, None] + col[None, :]
    selected = valid & (col[None, :] < n[:, None])
    token = tl.load(
        candidates + row[:, None] * TOKEN_ROW + col[None, :] * TOKEN_COL,
        selected, other=0,
    )
    tl.store(packed + dst, token, selected)
    tl.store(source + dst, row[:, None] * WIDTH + col[None, :], selected)


def allocate_verify_budget(
    prefix_logprobs: torch.Tensor,
    candidates: torch.Tensor,
    budget: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Allocate and pack per-request prefixes in one graph-capturable launch.

    ``candidates[B, W]`` includes the anchor at column zero. Prefix log
    probabilities ``[B, W-1]`` must be non-increasing along each request.
    Returns lengths, cumulative boundaries, source indices into the flattened
    candidates, and packed tokens. Lengths sum to ``budget``; every request
    keeps its anchor. Output shapes stay fixed across CUDA Graph replays.
    """
    b, width = candidates.shape
    if b < 1 or width < 1 or prefix_logprobs.shape != (b, width - 1):
        raise ValueError("Expected candidates[B,W] and prefix log probabilities[B,W-1]")
    if not b <= budget <= b * width:
        raise ValueError("Budget must lie between one and W tokens per request")
    if prefix_logprobs.device != candidates.device or not prefix_logprobs.is_cuda:
        raise ValueError("Inputs must share a CUDA device")
    lengths = torch.empty(b, dtype=torch.int32, device=candidates.device)
    boundaries = torch.empty(b + 1, dtype=torch.int32, device=candidates.device)
    source = torch.empty(budget, dtype=torch.int64, device=candidates.device)
    packed = torch.empty(budget, dtype=candidates.dtype, device=candidates.device)
    _allocate_and_pack_kernel[(1,)](
        prefix_logprobs, candidates, lengths, boundaries, source, packed,
        prefix_logprobs.stride(0), prefix_logprobs.stride(1),
        candidates.stride(0), candidates.stride(1),
        b, width, budget,
        triton.next_power_of_2(b), triton.next_power_of_2(width),
        num_warps=4,
    )
    return lengths, boundaries, source, packed
