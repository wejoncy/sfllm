"""Computed-update journals for speculative Gated DeltaNet verification.

Verification records FP32 residual updates, normalized keys and decay factors.
In deferred mode, the next verification folds the accepted prefix into the
persistent state before reading it; ordinary decode flushes the same prefix.
Both the persistent state and journal operands are FP32.
"""

from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import NVMMASharedLayout
from triton.experimental.gluon.language.nvidia.ampere import async_copy
from triton.experimental.gluon.language.nvidia.hopper import (
    fence_async_shared,
    tma,
)
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

GDNJournal = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def gdn_journal_capacity(verify_bound: int) -> int:
    """Reserve a bounded append journal independently of the verify width."""
    return max(12, verify_bound)


@triton.jit
def _commit_conv_journal_kernel(
    raw,
    states,
    indices,
    accepted_steps,
    pending_steps,
    pending_epoch,
    raw_layer_stride: tl.constexpr,
    raw_row_stride: tl.constexpr,
    raw_step_stride: tl.constexpr,
    state_layer_stride: tl.constexpr,
    state_slot_stride: tl.constexpr,
    state_slots: tl.constexpr,
    indices_stride: tl.constexpr,
    accepts_stride: tl.constexpr,
    width: tl.constexpr,
    history: tl.constexpr,
    block: tl.constexpr,
    journal_capacity: tl.constexpr,
    verify_bound: tl.constexpr,
):
    row = tl.program_id(0)
    layer = tl.program_id(2).to(tl.int64)
    slot = tl.load(indices + row * indices_stride).to(tl.int64)
    step = tl.minimum(tl.load(accepted_steps + row * accepts_stride), verify_bound - 1)
    if slot <= 0 or slot >= state_slots:
        return
    if pending_steps is not None and tl.program_id(1) == 0 and layer == 0:
        previous_count = tl.load(pending_steps + slot) + 1
        retained = tl.where(
            previous_count <= journal_capacity - verify_bound, previous_count, 0
        )
        old_epoch = tl.load(pending_epoch + slot)
        tl.store(pending_steps + slot, retained + step)
        tl.store(pending_epoch + slot, tl.where(retained > 0, old_epoch, 1 - old_epoch))
    if step < 0:
        return

    dim = tl.program_id(1) * block + tl.arange(0, block)
    mask = dim < width
    old = states + layer * state_layer_stride + slot * state_slot_stride
    journal = raw + layer * raw_layer_stride + row * raw_row_stride
    # The oldest history entry is read before that position is overwritten.
    for tap in tl.static_range(history):
        source_step = step + tap - history + 1
        if source_step < 0:
            value = tl.load(old + dim * history + source_step + history,
                            mask=mask, other=0)
        else:
            value = tl.load(journal + source_step * raw_step_stride + dim,
                            mask=mask, other=0)
        tl.store(old + dim * history + tap, value, mask=mask)


def commit_conv_journal(
    raw: torch.Tensor,
    states: torch.Tensor,
    state_indices: torch.Tensor,
    accepted_steps: torch.Tensor,
    pending_steps: torch.Tensor = None,
    pending_epoch: torch.Tensor = None,
    *,
    journal_capacity: int = None,
) -> None:
    """Commit convolution history and publish accepted GDN journal metadata."""
    layers, slots, width, history = states.shape
    batch = state_indices.numel()
    if not batch or not layers:
        return
    block = 512
    verify_bound = raw.shape[2]
    # Direct callers retain the original flush-on-every-verify contract.
    journal_capacity = verify_bound if journal_capacity is None else journal_capacity
    if journal_capacity < verify_bound:
        raise ValueError(
            "GDN journal capacity must cover the convolution verification width"
        )
    _commit_conv_journal_kernel[(batch, triton.cdiv(width, block), layers)](
        raw,
        states,
        state_indices,
        accepted_steps,
        pending_steps,
        pending_epoch,
        raw.stride(0),
        raw.stride(1),
        raw.stride(2),
        states.stride(0),
        states.stride(1),
        (
            min(slots, pending_steps.numel(), pending_epoch.numel())
            if pending_steps is not None
            else slots
        ),
        state_indices.stride(0),
        accepted_steps.stride(0),
        width,
        history,
        block,
        journal_capacity,
        verify_bound,
        num_warps=4,
    )


@triton.jit
def _reset_journal_pending_kernel(
    indices,
    pending_steps,
    pending_epoch,
    stride_indices: tl.constexpr,
    state_slots: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.load(indices + row * stride_indices)
    if slot > 0 and slot < state_slots:
        tl.store(pending_steps + slot, -1)
        tl.store(pending_epoch + slot, 0)


def reset_journal_pending(indices, pending_steps, pending_epoch) -> None:
    """A newly admitted prefill owns a fresh recurrent state in this slot."""
    if pending_steps is not None and indices.numel():
        _reset_journal_pending_kernel[(indices.numel(),)](
            indices,
            pending_steps,
            pending_epoch,
            indices.stride(0),
            min(pending_steps.numel(), pending_epoch.numel()),
        )


# Keep journal verification separate from ordinary decode.
@gluon.jit(do_not_specialize=["num_tokens"])
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
    pending_steps,
    pending_epoch,
    stride_qkv: gl.constexpr,
    stride_a: gl.constexpr,
    stride_b: gl.constexpr,
    stride_state: gl.constexpr,
    stride_indices: gl.constexpr,
    cu_seqlens,
    request_order,
    capacity: gl.constexpr,
    wide_offsets: gl.constexpr,
    scale,
    num_k_heads: gl.constexpr,
    num_v_heads: gl.constexpr,
    head_k_dim: gl.constexpr,
    head_v_dim: gl.constexpr,
    steps: gl.constexpr,
    key_epoch_stride: gl.constexpr,
    decay_epoch_stride: gl.constexpr,
    block_k: gl.constexpr,
    block_v: gl.constexpr,
    layout_warps: gl.constexpr,
    state_descriptor=None,
    precomputed: gl.constexpr = False,
    verify_bound: gl.constexpr = 0,
    enable_pdl: gl.constexpr = False,
    flat_grid: gl.constexpr = False,
    num_tokens=0,
    state_slots: gl.constexpr = 0,
):
    reserve: gl.constexpr = capacity if verify_bound == 0 else verify_bound
    value_tile = gl.program_id(1) if flat_grid else gl.program_id(0)
    batch_head = gl.program_id(0) if flat_grid else gl.program_id(1)
    batch = batch_head // num_v_heads
    if request_order is not None:
        batch = gl.load(request_order + batch)
    if wide_offsets:
        batch = batch.to(gl.int64)
    value_head = batch_head % num_v_heads
    key_head = value_head // (num_v_heads // num_k_heads)
    if cu_seqlens is not None:
        begin = gl.load(cu_seqlens + batch)
        length = gl.load(cu_seqlens + batch + 1) - begin
        if wide_offsets:
            begin = begin.to(gl.int64)
    else:
        begin = batch * steps
        length = steps
    if num_tokens > 0:
        length = gl.minimum(length, gl.maximum(num_tokens - begin, 0))
    length = gl.minimum(length, reserve)

    # Map K reductions within groups of eight lanes; warps divide the V rows.
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [8, 4], [layout_warps, 1], [1, 0])
    key_offsets = gl.arange(0, block_k, layout=gl.SliceLayout(0, layout))
    value_offsets = value_tile * block_v + gl.arange(
        0, block_v, layout=gl.SliceLayout(1, layout)
    )
    key_mask = key_offsets < head_k_dim
    value_mask = value_offsets < head_v_dim
    state_mask = value_mask[:, None] & key_mask[None, :]

    if cu_seqlens is not None and num_tokens > 0:
        batches = gl.num_programs(0 if flat_grid else 1) // num_v_heads
        active_tokens = gl.load(cu_seqlens + batches)
        if active_tokens < num_tokens:
            # Cover captured padding in this launch without a separate fill.
            for pad_step in range(reserve):
                pad_token = batch * reserve + pad_step
                pad_mask = (pad_token >= active_tokens) & (pad_token < num_tokens)
                gl.store(
                    output
                    + (pad_token * num_v_heads + value_head) * head_v_dim
                    + value_offsets,
                    0.0,
                    mask=value_mask & pad_mask,
                )

    slot = gl.load(state_indices + batch * stride_indices).to(gl.int64)
    if slot <= 0 or (state_slots > 0 and slot >= state_slots):
        if enable_pdl:
            gl.inline_asm_elementwise(
                "griddepcontrol.wait; // $0",
                "=r",
                [],
                dtype=gl.int32,
                is_pure=False,
                pack=1,
            )
        for step in range(length):
            token = begin + step
            output_ptr = output + (token * num_v_heads + value_head) * head_v_dim
            gl.store(output_ptr + value_offsets, 0.0, mask=value_mask)
        return

    state_offsets = (
        value_head * head_v_dim * head_k_dim
        + value_offsets[:, None] * head_k_dim
        + key_offsets[None, :]
    )
    state_ptr = states + slot * stride_state + state_offsets
    if pending_steps is not None:
        previous_count = gl.load(pending_steps + slot) + 1
        old_epoch = gl.load(pending_epoch + slot)
        retained = gl.where(previous_count <= capacity - reserve, previous_count, 0)
        row = ((slot - 1) * num_v_heads + value_head) * capacity
        key_row = ((slot - 1) * num_k_heads + key_head) * capacity
        if precomputed and capacity <= 12:
            hist_layout: gl.constexpr = gl.BlockedLayout([4], [32], [2], [0])
            hist_k = gl.arange(
                0, triton.next_power_of_2(capacity) * block_k, layout=hist_layout
            )
            hist_v = gl.arange(
                0, triton.next_power_of_2(capacity) * block_v, layout=hist_layout
            )
            shared_layout: gl.constexpr = gl.SwizzledSharedLayout(4, 1, 1, [0])
            ks = gl.allocate_shared_memory(
                gl.float32, [triton.next_power_of_2(capacity) * block_k], shared_layout
            )
            us = gl.allocate_shared_memory(
                gl.float32, [triton.next_power_of_2(capacity) * block_v], shared_layout
            )
            async_copy.async_copy_global_to_shared(
                ks,
                keys
                + old_epoch * key_epoch_stride
                + (key_row + hist_k // block_k) * head_k_dim
                + hist_k % block_k,
                mask=(hist_k // block_k < previous_count)
                & (hist_k % block_k < head_k_dim),
            )
            async_copy.async_copy_global_to_shared(
                us,
                updates
                + (row + hist_v // block_v) * head_v_dim
                + value_tile * block_v
                + hist_v % block_v,
                mask=(hist_v // block_v < previous_count)
                & (value_tile * block_v + hist_v % block_v < head_v_dim),
            )
            ds = gl.allocate_shared_memory(
                gl.float32, [triton.next_power_of_2(capacity)], shared_layout
            )
            hist_d = gl.arange(0, triton.next_power_of_2(capacity), layout=hist_layout)
            async_copy.async_copy_global_to_shared(
                ds,
                decays + old_epoch * decay_epoch_stride + row + hist_d,
                mask=hist_d < previous_count,
            )
            async_copy.commit_group()
            state = gl.load(state_ptr, mask=state_mask, other=0.0).to(gl.float32)
            async_copy.wait_group(0)
            for old_step in range(previous_count):
                old_key = ks.gather(old_step * block_k + key_offsets, axis=0)
                old_update = us.gather(
                    old_step * block_v + value_offsets - value_tile * block_v, axis=0
                )
                old_decay = gl.sum(
                    ds.gather(
                        gl.full(
                            [1],
                            old_step,
                            gl.int32,
                            gl.BlockedLayout([1], [32], [2], [0]),
                        ),
                        axis=0,
                    ),
                    axis=0,
                )
                state = gl.inline_asm_elementwise(
                    "mul.rn.f32 $0, $1, $2;", constraints="=f,f,f",
                    args=[state, old_decay], dtype=gl.float32, is_pure=True, pack=1,
                )
                state = gl.fma(old_update[:, None], old_key[None, :], state)
        else:
            state = gl.load(state_ptr, mask=state_mask, other=0.0).to(gl.float32)
            for old_step in range(previous_count):
                old_key = gl.load(
                    keys
                    + old_epoch * key_epoch_stride
                    + (key_row + old_step) * head_k_dim
                    + key_offsets,
                    key_mask,
                    other=0,
                )
                old_update = gl.load(
                    updates + (row + old_step) * head_v_dim + value_offsets,
                    value_mask,
                    other=0,
                )
                old_decay = gl.load(
                    decays + old_epoch * decay_epoch_stride + row + old_step,
                )
                state = gl.inline_asm_elementwise(
                    "mul.rn.f32 $0, $1, $2;",
                    "=f,f,f",
                    [state, old_decay],
                    dtype=gl.float32,
                    is_pure=True,
                    pack=1,
                )
                state = gl.fma(old_update[:, None], old_key[None, :], state)
        if retained == 0 and previous_count > 0:
            if state_descriptor is not None:
                shared = gl.allocate_shared_memory(
                    gl.float32, [block_v, block_k], state_descriptor.layout
                )
                shared.store(state)
                fence_async_shared()
                tma.async_copy_shared_to_global(
                    state_descriptor,
                    [
                        (
                            slot * num_v_heads * head_v_dim
                            + value_head * head_v_dim
                            + value_tile * block_v
                        ).to(gl.int32),
                        0,
                    ],
                    shared,
                )
            else:
                gl.store(state_ptr, state, mask=state_mask)
    else:
        state = gl.load(state_ptr, mask=state_mask, other=0.0).to(gl.float32)
        row = (batch * num_v_heads + value_head) * capacity
        key_row = (batch * num_k_heads + key_head) * capacity
        retained = 0
    if enable_pdl:
        gl.inline_asm_elementwise(
            "griddepcontrol.wait; // $0",
            "=r",
            [],
            dtype=gl.int32,
            is_pure=False,
            pack=1,
        )
        gl.inline_asm_elementwise(
            "griddepcontrol.launch_dependents; // $0",
            "=r",
            [],
            dtype=gl.int32,
            is_pure=False,
            pack=1,
        )
    if not precomputed:
        decay_scale = gl.exp(gl.load(a_log + value_head).to(gl.float32))
        bias = gl.load(dt_bias + value_head).to(gl.float32)
    for step in range(length):
        token = begin + step
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

        if precomputed:
            decay = gl.load(a + token * stride_a + value_head)
            beta = gl.load(b + token * stride_b + value_head)
        else:
            q *= gl.rsqrt(gl.sum(q * q, axis=0) + 1e-6) * scale
            k *= gl.rsqrt(gl.sum(k * k, axis=0) + 1e-6)
            gate_a = gl.load(a + token * stride_a + value_head).to(gl.float32)
            gate_b = gl.load(b + token * stride_b + value_head).to(gl.float32)
            softplus_arg = gate_a + bias
            softplus = gl.where(
                softplus_arg <= 20.0,
                gl.log(1.0 + gl.exp(softplus_arg)),
                softplus_arg,
            )
            decay = gl.exp(-decay_scale * softplus)
            beta = tl.sigmoid(gate_b)
        state *= decay
        prediction = gl.sum(state * k[None, :], axis=1)
        correction = (v - prediction) * beta
        state += correction[:, None] * k[None, :]
        result = gl.sum(state * q[None, :], axis=1)
        output_ptr = output + (token * num_v_heads + value_head) * head_v_dim
        gl.store(output_ptr + value_offsets, result, mask=value_mask)
        # These are computed FP32 operands, not raw projection inputs.
        gl.store(
            updates + (row + retained + step) * head_v_dim + value_offsets,
            correction,
            mask=value_mask,
        )
        if value_tile == 0:
            decay_out = decays + row + retained + step
            key_out = keys + (key_row + retained + step) * head_k_dim + key_offsets
            if pending_steps is not None:
                decay_out += (
                    gl.where(retained > 0, old_epoch, 1 - old_epoch)
                    * decay_epoch_stride
                )
                key_out += (
                    gl.where(retained > 0, old_epoch, 1 - old_epoch) * key_epoch_stride
                )
            gl.store(decay_out, decay)
            if value_head % (num_v_heads // num_k_heads) == 0:
                gl.store(key_out, k, mask=key_mask)

    if pending_steps is not None and state_descriptor is not None:
        if retained == 0 and previous_count > 0:
            tma.store_wait(0)


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
    cu_seqlens: torch.Tensor = None,
    request_order: torch.Tensor = None,
    pending_steps: torch.Tensor = None,
    pending_epoch: torch.Tensor = None,
    *,
    precomputed: bool = False,
    verify_bound: int = None,
    output_dtype: torch.dtype = None,
) -> torch.Tensor:
    """Verify a linear block without writing full candidate state snapshots.

    In deferred mode, journals use persistent slot rows and key/decay double
    buffers. The previous accepted prefix is folded into state on its next read.
    cu_seqlens optionally delimits packed requests, each of length 0..capacity.
    Without it, all rows have the same positive verification length.
    """
    if states.dtype != torch.float32:
        raise ValueError("GDN journals require FP32 recurrent states")
    updates, keys, decays = journal
    batch = state_indices.numel()
    num_tokens = mixed_qkv.shape[0]
    num_v_heads, head_v_dim, head_k_dim = states.shape[-3:]
    capacity = updates.shape[-2]
    verify_bound = capacity if verify_bound is None else verify_bound
    if not 0 < verify_bound <= capacity:
        raise ValueError("GDN verification width must fit the journal capacity")
    journal_rows = updates.shape[0]
    if pending_steps is not None:
        if pending_epoch is None or keys.shape[0] != 2 or decays.shape[0] != 2:
            raise ValueError("Deferred GDN journals need pending steps and two epochs")
    if cu_seqlens is not None:
        if (
            cu_seqlens.ndim != 1
            or cu_seqlens.numel() != batch + 1
            or not cu_seqlens.is_contiguous()
            or cu_seqlens.dtype not in (torch.int32, torch.int64)
            or cu_seqlens.device != mixed_qkv.device
            or batch == 0
            or batch > journal_rows
            or num_tokens > batch * verify_bound
        ):
            raise ValueError("Invalid packed GDN verification metadata or capacity")
        steps = verify_bound
    elif (
        batch == 0
        or batch > journal_rows
        or num_tokens % batch
        or not 0 < num_tokens // batch <= verify_bound
    ):
        raise ValueError("GDN verification requires a nonempty, equal-width block per row")
    else:
        steps = num_tokens // batch
    output = mixed_qkv.new_empty(
        (num_tokens, num_v_heads, head_v_dim), dtype=output_dtype or mixed_qkv.dtype
    )
    # Keep the register footprint small while distributing V rows over two warps.
    # The resulting FP32 reduction order can differ slightly from snapshot verify.
    block_v = min(
        triton.next_power_of_2(head_v_dim),
        64 if cu_seqlens is not None or precomputed else 32,
    )
    num_warps = 2
    supports_pdl = (
        states.is_cuda
        and torch.version.hip is None
        and torch.cuda.get_device_capability(states.device)[0] >= 9
    )
    enable_pdl = supports_pdl and precomputed and 0 < num_tokens <= 192
    descriptor = None
    if (
        supports_pdl
        and precomputed
        and states.is_contiguous()
        and head_k_dim == head_v_dim == 128
    ):
        descriptor = TensorDescriptor(
            states,
            [states.shape[0] * num_v_heads * head_v_dim, head_k_dim],
            [head_k_dim, 1],
            [block_v, head_k_dim],
            NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=32, rank=2),
        )
    # Keep common blocks' index arithmetic small without imposing a size limit.
    wide_offsets = max(
        num_tokens * max(mixed_qkv.stride(0), a.stride(0), b.stride(0),
                         num_v_heads * head_v_dim),
        updates.numel(), keys.numel(), decays.numel(),
    ) >= 2**31
    grid = (
        (batch * num_v_heads, triton.cdiv(head_v_dim, block_v))
        if precomputed
        else (triton.cdiv(head_v_dim, block_v), batch * num_v_heads)
    )
    _packed_gdn_journal_kernel[grid](
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
        pending_steps,
        pending_epoch,
        mixed_qkv.stride(0),
        a.stride(0),
        b.stride(0),
        states.stride(0),
        state_indices.stride(0),
        cu_seqlens,
        request_order,
        capacity,
        wide_offsets,
        head_k_dim**-0.5,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        steps,
        keys.stride(0) if pending_steps is not None else 0,
        decays.stride(0) if pending_steps is not None else 0,
        triton.next_power_of_2(head_k_dim),
        block_v,
        layout_warps=num_warps,
        state_descriptor=descriptor,
        precomputed=precomputed,
        verify_bound=verify_bound,
        enable_pdl=enable_pdl,
        flat_grid=precomputed,
        num_tokens=num_tokens,
        state_slots=(
            min(
                states.shape[0],
                journal_rows + 1,
                pending_steps.numel(),
                pending_epoch.numel(),
            )
            if pending_steps is not None
            else states.shape[0]
        ),
        num_warps=num_warps,
        num_stages=3,
        launch_pdl=enable_pdl,
    )
    return output


@triton.jit
def _replay_gdn_journal_kernel(
    states, updates, keys, decays, indices, accepted_steps, pending_epoch,
    stride_state_layer: tl.constexpr, stride_state_slot: tl.constexpr,
    stride_update_layer: tl.constexpr, stride_key_layer: tl.constexpr,
    stride_decay_layer: tl.constexpr, stride_indices: tl.constexpr,
    stride_accepts: tl.constexpr,
    stride_key_epoch: tl.constexpr, stride_decay_epoch: tl.constexpr,
    num_v_heads: tl.constexpr, num_k_heads: tl.constexpr, head_k_dim: tl.constexpr,
    head_v_dim: tl.constexpr, capacity: tl.constexpr,
    wide_offsets: tl.constexpr,
    block_k: tl.constexpr, block_v: tl.constexpr,
    state_slots: tl.constexpr,
):
    value_tile, batch, layer_head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if wide_offsets:
        batch = batch.to(tl.int64)
    layer = (layer_head // num_v_heads).to(tl.int64)
    head = layer_head % num_v_heads
    slot = tl.load(indices + batch * stride_indices).to(tl.int64)
    if slot <= 0 or slot >= state_slots:
        return
    if pending_epoch is not None:
        count = tl.load(accepted_steps + slot) + 1
        epoch = tl.load(pending_epoch + slot)
        journal_row = slot - 1
    else:
        # Legacy direct replay is also the reference for deferred replay tests.
        count = tl.load(accepted_steps + batch * stride_accepts) + 1
        epoch = 0
        journal_row = batch
    count = tl.minimum(count, capacity)
    if count <= 0:
        return

    kk = tl.arange(0, block_k)
    vv = value_tile * block_v + tl.arange(0, block_v)
    mask = (vv[:, None] < head_v_dim) & (kk[None, :] < head_k_dim)
    state_ptr = states + layer * stride_state_layer + slot * stride_state_slot
    state_ptr += head * head_v_dim * head_k_dim + vv[:, None] * head_k_dim + kk[None, :]
    state = tl.load(state_ptr, mask, other=0).to(tl.float32)
    row = (journal_row * num_v_heads + head) * capacity
    key_row = (journal_row * num_k_heads + head // (num_v_heads // num_k_heads)) * capacity
    for step in range(count):
        key = tl.load(keys + layer * stride_key_layer + epoch * stride_key_epoch
                      + (key_row + step) * head_k_dim + kk,
                      kk < head_k_dim, other=0)
        update = tl.load(updates + layer * stride_update_layer + (row + step) * head_v_dim + vv,
                         vv < head_v_dim, other=0)
        decay = tl.load(decays + layer * stride_decay_layer
                        + epoch * stride_decay_epoch + row + step)
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
    pending_epoch: torch.Tensor = None,
) -> None:
    """Commit per-request accepted prefixes across all GDN layers in one launch.

    states is [layer, slot, value head, V, K]; journals have an outer layer axis.
    accepted_steps contains a last accepted index in [-1, verified steps - 1].
    With pending_epoch, it is indexed by persistent slot and flushes deferred
    state before an ordinary decode transition.
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
        states, updates, keys, decays, state_indices, accepted_steps, pending_epoch,
        states.stride(0), states.stride(1), updates.stride(0), keys.stride(0),
        decays.stride(0), state_indices.stride(0), accepted_steps.stride(0),
        keys.stride(1) if pending_epoch is not None else 0,
        decays.stride(1) if pending_epoch is not None else 0,
        num_v_heads, keys.shape[-3], head_k_dim, head_v_dim, updates.shape[-2],
        max(updates.stride(0), keys.stride(0), decays.stride(0)) >= 2**31,
        triton.next_power_of_2(head_k_dim), block_v,
        min(states.shape[1], updates.shape[1] + 1,
            accepted_steps.numel(), pending_epoch.numel())
        if pending_epoch is not None else states.shape[1],
        num_warps=4, num_stages=3,
    )
