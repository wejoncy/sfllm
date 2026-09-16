"""Computed-update journals for speculative Gated DeltaNet verification.

Verification leaves the persistent state untouched and records FP32 residual
updates, normalized keys and decay factors. Commit replays each request's own
accepted prefix. Both the persistent state and journal operands are FP32.
"""

from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


GDNJournal = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


# This separate specialization leaves the existing decode/verify kernels intact.
@gluon.jit
def _packed_gdn_journal_kernel(
    mixed_qkv,
    a,
    b,
    a_log,
    dt_bias,
    output,
    states,
    state_indices,
    updates,
    keys,
    decays,
    stride_qkv: gl.constexpr,
    stride_a: gl.constexpr,
    stride_b: gl.constexpr,
    stride_state: gl.constexpr,
    stride_indices: gl.constexpr,
    capacity: gl.constexpr,
    wide_offsets: gl.constexpr,
    scale,
    num_k_heads: gl.constexpr,
    num_v_heads: gl.constexpr,
    head_k_dim: gl.constexpr,
    head_v_dim: gl.constexpr,
    steps: gl.constexpr,
    block_k: gl.constexpr,
    block_v: gl.constexpr,
    layout_warps: gl.constexpr,
):
    value_tile = gl.program_id(0)
    batch_head = gl.program_id(1)
    batch = batch_head // num_v_heads
    if wide_offsets:
        batch = batch.to(gl.int64)
    value_head = batch_head % num_v_heads
    key_head = value_head // (num_v_heads // num_k_heads)

    # Map K reductions within groups of eight lanes; warps divide the V rows.
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [layout_warps, 1], [1, 0])
    key_offsets = gl.arange(0, block_k, layout=gl.SliceLayout(0, layout))
    value_offsets = value_tile * block_v + gl.arange(
        0, block_v, layout=gl.SliceLayout(1, layout)
    )
    key_mask = key_offsets < head_k_dim
    value_mask = value_offsets < head_v_dim
    state_mask = value_mask[:, None] & key_mask[None, :]

    slot = gl.load(state_indices + batch * stride_indices).to(gl.int64)
    if slot <= 0:
        for step in range(steps):
            token = batch * steps + step
            output_ptr = output + (token * num_v_heads + value_head) * head_v_dim
            gl.store(output_ptr + value_offsets, 0.0, mask=value_mask)
        return

    state_offsets = (
        value_head * head_v_dim * head_k_dim
        + value_offsets[:, None] * head_k_dim
        + key_offsets[None, :]
    )
    state_ptr = states + slot * stride_state + state_offsets
    state = gl.load(state_ptr, mask=state_mask, other=0.0).to(gl.float32)
    for step in range(steps):
        token = batch * steps + step
        qkv_ptr = mixed_qkv + token * stride_qkv
        q = gl.load(
            qkv_ptr + key_head * head_k_dim + key_offsets,
            mask=key_mask,
            other=0.0,
        ).to(gl.float32)
        k = gl.load(
            qkv_ptr + num_k_heads * head_k_dim + key_head * head_k_dim + key_offsets,
            mask=key_mask,
            other=0.0,
        ).to(gl.float32)
        v = gl.load(
            qkv_ptr
            + 2 * num_k_heads * head_k_dim
            + value_head * head_v_dim
            + value_offsets,
            mask=value_mask,
            other=0.0,
        ).to(gl.float32)

        q *= gl.rsqrt(gl.sum(q * q, axis=0) + 1e-6) * scale
        k *= gl.rsqrt(gl.sum(k * k, axis=0) + 1e-6)
        gate_a = gl.load(a + token * stride_a + value_head).to(gl.float32)
        gate_b = gl.load(b + token * stride_b + value_head).to(gl.float32)
        decay_log = gl.load(a_log + value_head).to(gl.float32)
        bias = gl.load(dt_bias + value_head).to(gl.float32)
        softplus_arg = gate_a + bias
        softplus = gl.where(
            softplus_arg <= 20.0,
            gl.log(1.0 + gl.exp(softplus_arg)),
            softplus_arg,
        )
        decay = gl.exp(-gl.exp(decay_log) * softplus)
        state *= decay
        prediction = gl.sum(state * k[None, :], axis=1)
        correction = (v - prediction) * tl.sigmoid(gate_b)
        state += correction[:, None] * k[None, :]
        result = gl.sum(state * q[None, :], axis=1)
        output_ptr = output + (token * num_v_heads + value_head) * head_v_dim
        gl.store(output_ptr + value_offsets, result, mask=value_mask)
        # These are computed FP32 operands, not raw projection inputs.
        row = (batch * num_v_heads + value_head) * capacity + step
        gl.store(updates + row * head_v_dim + value_offsets, correction, mask=value_mask)
        if value_tile == 0:
            gl.store(keys + row * head_k_dim + key_offsets, k, mask=key_mask)
            gl.store(decays + row, decay)


def packed_gdn_journal_verify(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    states: torch.Tensor,
    state_indices: torch.Tensor,
    num_k_heads: int,
    journal: GDNJournal,
) -> torch.Tensor:
    """Verify a linear block without writing full candidate state snapshots.

    Updates/keys are [batch capacity, value heads, block capacity, V/K];
    decays have no final dimension.
    They use batch rows, independent of the request's persistent state slot.
    Any positive block length up to the allocated capacity is supported.
    """
    if states.dtype != torch.float32:
        raise ValueError("GDN journals require FP32 recurrent states")
    updates, keys, decays = journal
    batch = state_indices.numel()
    num_tokens = mixed_qkv.shape[0]
    num_v_heads, head_v_dim, head_k_dim = states.shape[-3:]
    capacity = updates.shape[-2]
    if (batch == 0 or batch > updates.shape[0] or num_tokens % batch
            or not 0 < num_tokens // batch <= capacity):
        raise ValueError("GDN verification requires a nonempty, equal-width block per row")
    steps = num_tokens // batch
    output = mixed_qkv.new_empty((num_tokens, num_v_heads, head_v_dim))
    # Keep the register footprint small while distributing V rows over two warps.
    # The resulting FP32 reduction order can differ slightly from snapshot verify.
    block_v = min(triton.next_power_of_2(head_v_dim), 16)
    num_warps = 2
    # Keep common blocks' index arithmetic small without imposing a size limit.
    wide_offsets = max(
        num_tokens * max(mixed_qkv.stride(0), a.stride(0), b.stride(0),
                         num_v_heads * head_v_dim),
        updates.numel(), keys.numel(), decays.numel(),
    ) >= 2**31
    _packed_gdn_journal_kernel[(triton.cdiv(head_v_dim, block_v), batch * num_v_heads)](
        mixed_qkv, a, b, a_log, dt_bias, output, states, state_indices,
        updates, keys, decays,
        mixed_qkv.stride(0), a.stride(0), b.stride(0), states.stride(0),
        state_indices.stride(0), capacity, wide_offsets, head_k_dim**-0.5,
        num_k_heads, num_v_heads, head_k_dim, head_v_dim, steps,
        triton.next_power_of_2(head_k_dim), block_v, layout_warps=num_warps,
        num_warps=num_warps, num_stages=3,
    )
    return output


@triton.jit
def _replay_gdn_journal_kernel(
    states, updates, keys, decays, indices, accepted_steps,
    stride_state_layer: tl.constexpr, stride_state_slot: tl.constexpr,
    stride_update_layer: tl.constexpr, stride_key_layer: tl.constexpr,
    stride_decay_layer: tl.constexpr, stride_indices: tl.constexpr,
    stride_accepts: tl.constexpr,
    num_v_heads: tl.constexpr, head_k_dim: tl.constexpr,
    head_v_dim: tl.constexpr, capacity: tl.constexpr,
    wide_offsets: tl.constexpr,
    block_k: tl.constexpr, block_v: tl.constexpr,
):
    value_tile, batch, layer_head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if wide_offsets:
        batch = batch.to(tl.int64)
    layer = (layer_head // num_v_heads).to(tl.int64)
    head = layer_head % num_v_heads
    slot = tl.load(indices + batch * stride_indices).to(tl.int64)
    # accepted_steps is the last accepted index: zero commits just the anchor.
    count = tl.load(accepted_steps + batch * stride_accepts) + 1
    if slot <= 0 or count <= 0:
        return

    kk = tl.arange(0, block_k)
    vv = value_tile * block_v + tl.arange(0, block_v)
    mask = (vv[:, None] < head_v_dim) & (kk[None, :] < head_k_dim)
    state_ptr = states + layer * stride_state_layer + slot * stride_state_slot
    state_ptr += head * head_v_dim * head_k_dim + vv[:, None] * head_k_dim + kk[None, :]
    state = tl.load(state_ptr, mask, other=0).to(tl.float32)
    row = (batch * num_v_heads + head) * capacity
    for step in range(count):
        key = tl.load(keys + layer * stride_key_layer + (row + step) * head_k_dim + kk,
                      kk < head_k_dim, other=0)
        update = tl.load(updates + layer * stride_update_layer + (row + step) * head_v_dim + vv,
                         vv < head_v_dim, other=0)
        decay = tl.load(decays + layer * stride_decay_layer + row + step)
        # The verify kernel rounds the decay product before its rank-one FMA.
        # Prevent contraction/reassociation across these two operations.
        state = tl.inline_asm_elementwise(
            "mul.rn.f32 $0, $1, $2;", constraints="=f,f,f",
            args=[state, decay], dtype=tl.float32, is_pure=True, pack=1,
        )
        state = tl.fma(update[:, None], key[None, :], state)
    tl.store(state_ptr, state, mask)


def replay_gdn_journal(
    states: torch.Tensor,
    journal: GDNJournal,
    state_indices: torch.Tensor,
    accepted_steps: torch.Tensor,
) -> None:
    """Commit per-request accepted prefixes across all GDN layers in one launch.

    states is [layer, slot, value head, V, K]; journals have an outer layer axis.
    accepted_steps contains a last accepted index in [-1, verified steps - 1].
    A negative index commits nothing; state slots <= 0 denote padded rows.
    Neither acceptance lengths nor request indices are copied to the CPU.
    """
    if states.dtype != torch.float32:
        raise ValueError("GDN journals require FP32 recurrent states")
    updates, keys, decays = journal
    layers, _, num_v_heads, head_v_dim, head_k_dim = states.shape
    batch = state_indices.numel()
    if batch == 0 or layers == 0:
        return
    block_v = min(triton.next_power_of_2(head_v_dim), 64)
    _replay_gdn_journal_kernel[(triton.cdiv(head_v_dim, block_v), batch, layers * num_v_heads)](
        states, updates, keys, decays, state_indices, accepted_steps,
        states.stride(0), states.stride(1), updates.stride(0), keys.stride(0),
        decays.stride(0), state_indices.stride(0), accepted_steps.stride(0),
        num_v_heads, head_k_dim, head_v_dim, updates.shape[-2],
        max(updates.stride(0), keys.stride(0), decays.stride(0)) >= 2**31,
        triton.next_power_of_2(head_k_dim), block_v,
        num_warps=4, num_stages=3,
    )
