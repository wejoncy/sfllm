"""Capture-safe kernels for the DFlash2 block algorithm."""

import torch
import triton
import triton.language as tl


@triton.jit
def _prepare_block_kernel(
    anchor_tokens,
    anchor_positions,
    block_ids,
    positions,
    mask_token_id,
    block_size: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    valid = cols < block_size
    anchor = tl.load(anchor_tokens + row)
    position = tl.load(anchor_positions + row)
    offset = row * block_size + cols
    tl.store(
        block_ids + offset,
        tl.where(cols == 0, anchor, mask_token_id),
        mask=valid,
    )
    tl.store(positions + offset, position + cols, mask=valid)


def prepare_dflash2_block(
    *,
    anchor_tokens: torch.Tensor,
    anchor_positions: torch.Tensor,
    block_ids_out: torch.Tensor,
    positions_out: torch.Tensor,
    mask_token_id: int,
) -> None:
    batch_size, block_size = block_ids_out.shape
    _prepare_block_kernel[(batch_size,)](
        anchor_tokens,
        anchor_positions,
        block_ids_out,
        positions_out,
        int(mask_token_id),
        block_size=block_size,
        BLOCK=triton.next_power_of_2(block_size),
        num_warps=1,
    )


@triton.jit
def _selector_greedy_walk_kernel(
    candidate_ids,
    unary_logits,
    pairwise,
    proposals_out,
    slots: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    valid = offsets < top_k
    predecessor = 0
    for slot in range(slots):
        score_offset = ((row * slots + slot) * top_k + predecessor) * top_k
        values = tl.load(
            pairwise + score_offset + offsets,
            mask=valid,
            other=-float("inf"),
        ).to(tl.float32)
        unary = tl.load(
            unary_logits + (row * slots + slot) * top_k + offsets,
            mask=valid, other=0,
        ).to(tl.float32)
        values = unary + values
        best = tl.max(values, axis=0)
        selected = tl.min(tl.where(values == best, offsets, top_k), axis=0)
        token_offset = (row * slots + slot) * top_k + selected
        tl.store(
            proposals_out + row * slots + slot,
            tl.load(candidate_ids + token_offset),
        )
        predecessor = selected


def dflash2_selector_greedy_walk(
    candidate_ids: torch.Tensor,
    unary_logits: torch.Tensor,
    pairwise: torch.Tensor,
    proposals_out: torch.Tensor,
) -> None:
    batch_size, slots, top_k = candidate_ids.shape
    _selector_greedy_walk_kernel[(batch_size,)](
        candidate_ids,
        unary_logits,
        pairwise,
        proposals_out,
        slots=slots,
        top_k=top_k,
        BLOCK_K=triton.next_power_of_2(top_k),
        num_warps=1,
    )


__all__ = [
    "dflash2_selector_greedy_walk",
    "prepare_dflash2_block",
]
