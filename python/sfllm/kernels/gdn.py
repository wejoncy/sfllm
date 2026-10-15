"""Gated DeltaNet backends and fused preprocessing kernels for Qwen3.5."""

from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl
from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_fwd
try:
    from flashinfer import gdn_decode
    from flashinfer.gdn_prefill import chunk_gated_delta_rule
except ModuleNotFoundError as exc:
    if exc.name != "flashinfer":
        raise
    gdn_decode = chunk_gated_delta_rule = None
from triton.experimental import gluon as tg
from triton.experimental.gluon import language as gl
import sf_kernel

from sfllm.kernels.gdn_journal import GDNJournal, packed_gdn_journal_verify


@tg.jit(
    do_not_specialize=["num_tokens", "num_sequences"],
    do_not_specialize_on_alignment=["num_tokens", "num_sequences"],
)
def _split_l2norm_qkv_gates_kernel(
    mixed_qkv, q, k, v, g, beta, a, b, a_log, dt_bias,
    conv_weight, conv_states, query_start_loc, state_indices,
    stride_token: gl.constexpr, stride_dim: gl.constexpr, stride_state: gl.constexpr,
    stride_a_token: gl.constexpr, stride_a_head: gl.constexpr,
    stride_b_token: gl.constexpr, stride_b_head: gl.constexpr,
    num_tokens, num_sequences,
    num_k_heads: gl.constexpr, num_v_heads: gl.constexpr,
    head_k_dim: gl.constexpr, head_v_dim: gl.constexpr,
    kernel_width: gl.constexpr, block_t: gl.constexpr,
    block_k: gl.constexpr, log_g: gl.constexpr,
    initial_conv=None,
    source_indices=None,
):
    # Coalesce convolution loads and state writes along the head dimension.
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [1, 32], [4, 1], [1, 0])
    head = gl.program_id(0)
    d = gl.arange(0, block_k, layout=gl.SliceLayout(0, layout))
    if head < num_k_heads:
        dim, heads = head_k_dim, num_k_heads
        columns = head * dim + d
        input_columns = columns
        output = q
    elif head < 2 * num_k_heads:
        dim, heads = head_k_dim, num_k_heads
        columns = (head - num_k_heads) * dim + d
        input_columns = num_k_heads * head_k_dim + columns
        output = k
    else:
        dim, heads = head_v_dim, num_v_heads
        columns = (head - 2 * num_k_heads) * dim + d
        input_columns = 2 * num_k_heads * head_k_dim + columns
        output = v
    tile = gl.program_id(1)
    num_tiles = (num_tokens + block_t - 1) // block_t
    # A few extra CTAs save the final raw taps; convolution CTAs need no state addresses.
    if tile >= num_tiles:
        seq = tile - num_tiles
        state_seq_start = gl.load(query_start_loc + seq)
        state_seq_end = gl.load(query_start_loc + seq + 1)
        state_slot = gl.load(state_indices + seq)
        for tap_start in range(0, kernel_width - 1, block_t):
            state_tap = tap_start + gl.arange(0, block_t, layout=gl.SliceLayout(1, layout))
            state_token = state_seq_end - kernel_width + 1 + state_tap
            state_mask = (state_tap[:, None] < kernel_width - 1) & (d[None, :] < dim) & (state_slot >= 0)
            input_token = state_token
            if source_indices is not None:
                input_token = gl.load(source_indices + state_token,
                    (state_tap < kernel_width - 1) & (state_token >= state_seq_start), other=0)
            state_x = gl.load(
                mixed_qkv + input_token[:, None] * stride_token + input_columns[None, :] * stride_dim,
                mask=state_mask & (state_token[:, None] >= state_seq_start), other=0,
            )
            if initial_conv is not None:
                old_tap = state_token - state_seq_start + kernel_width - 1
                old_x = gl.load(
                    initial_conv + seq * stride_state + input_columns[None, :] * (kernel_width - 1)
                    + old_tap[:, None],
                    state_mask & (state_token[:, None] < state_seq_start) & (old_tap[:, None] >= 0), other=0,
                )
                state_x = gl.where(state_token[:, None] < state_seq_start, old_x, state_x)
                state_mask &= state_seq_end > state_seq_start
            gl.store(
                conv_states + state_slot * stride_state + input_columns[None, :] * (kernel_width - 1)
                + state_tap[:, None], state_x, mask=state_mask,
            )
    else:
        token = tile * block_t + gl.arange(0, block_t, layout=gl.SliceLayout(1, layout))
        token_mask = token < num_tokens
        # Find each token's packed sequence without constructing a token-to-row map.
        lo = gl.full((block_t,), 0, gl.int32, gl.SliceLayout(1, layout))
        remaining = num_sequences
        while remaining > 1:
            half = remaining // 2
            mid = lo + half
            start = gl.load(query_start_loc + mid)
            lo = gl.where(token >= start, mid, lo)
            remaining -= half
        seq_start = gl.load(query_start_loc + lo, mask=token_mask, other=0)
        if initial_conv is not None:
            # Only tiles touching a sequence start need the previous conv taps.
            needs_initial = gl.sum(
                (token < seq_start + kernel_width - 1) & token_mask, 0,
            ) > 0

        mask = token_mask[:, None] & (d[None, :] < dim)
        value = gl.full(mask.shape, 0.0, gl.float32, layout)
        # Historical taps precede current-input taps; preserve their FMA order
        # without keeping both input selections live through every convolution tap.
        if initial_conv is not None:
            if needs_initial:
                for tap in gl.static_range(kernel_width - 1):
                    source_token = token + tap - kernel_width + 1
                    old_tap = source_token - seq_start + kernel_width - 1
                    old_x = gl.load(
                        initial_conv + lo[:, None] * stride_state
                        + input_columns[None, :] * (kernel_width - 1) + old_tap[:, None],
                        mask & (source_token[:, None] < seq_start[:, None])
                        & (old_tap[:, None] >= 0), other=0,
                    ).to(gl.float32)
                    weight = gl.load(conv_weight + input_columns * kernel_width + tap,
                                     d < dim, other=0).to(gl.float32)
                    value = gl.fma(old_x, weight[None, :], value)
        for tap in gl.static_range(kernel_width):
            source_token = token + tap - kernel_width + 1
            input_token = source_token
            if source_indices is not None:
                input_token = gl.load(source_indices + source_token,
                    token_mask & (source_token >= seq_start), other=0)
            x = gl.load(
                mixed_qkv + input_token[:, None] * stride_token + input_columns[None, :] * stride_dim,
                mask=mask & (source_token[:, None] >= seq_start[:, None]), other=0,
            ).to(gl.float32)
            weight = gl.load(
                conv_weight + input_columns * kernel_width + tap,
                mask=d < dim, other=0,
            ).to(gl.float32)
            value = gl.fma(x.to(gl.float32), weight[None, :], value)
        value = (value / (1.0 + gl.exp(-value))).to(mixed_qkv.dtype.element_ty).to(gl.float32)
        if head < 2 * num_k_heads:
            value *= gl.rsqrt(gl.sum(value * value, axis=1)[:, None] + 1e-6)
        gl.store(output + token[:, None] * heads * dim + columns[None, :], value, mask=mask)

        if head >= 2 * num_k_heads:
            head -= 2 * num_k_heads
            gate_token = token
            if source_indices is not None:
                gate_token = gl.load(source_indices + token, token_mask, other=0)
            a_value = gl.load(a + gate_token * stride_a_token + head * stride_a_head,
                              mask=token_mask, other=0).to(gl.float32)
            b_value = gl.load(b + gate_token * stride_b_token + head * stride_b_head,
                              mask=token_mask, other=0).to(gl.float32)
            decay = gl.load(a_log + head).to(gl.float32)
            bias = gl.load(dt_bias + head).to(gl.float32)
            softplus_input = a_value + bias
            softplus = gl.where(softplus_input <= 20.0,
                                gl.log(1.0 + gl.exp(softplus_input)), softplus_input)
            gate_offset = token * num_v_heads + head
            gate = -gl.exp(decay) * softplus
            gl.store(g + gate_offset, gate if log_g else gl.exp(gate), mask=token_mask)
            gl.store(beta + gate_offset, (1.0 / (1.0 + gl.exp(-b_value))).to(b.dtype.element_ty), mask=token_mask)


def split_l2norm_qkv_gates(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    *,
    conv_weight: torch.Tensor,
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_indices: torch.Tensor,
    log_g: bool = False,
    initial_conv: torch.Tensor = None,
    source_indices: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse packed prefill convolution, QK normalization and GDN gates."""
    num_tokens = mixed_qkv.shape[0] if source_indices is None else source_indices.numel()
    q = mixed_qkv.new_empty((num_tokens, num_k_heads, head_k_dim))
    k = torch.empty_like(q)
    v = mixed_qkv.new_empty((num_tokens, num_v_heads, head_v_dim))
    if source_indices is None:
        g = torch.empty_like(a, dtype=torch.float32)
        beta = torch.empty_like(b, dtype=torch.float32)
    else:
        g = a.new_empty((num_tokens, num_v_heads), dtype=torch.float32)
        beta = b.new_empty((num_tokens, num_v_heads), dtype=torch.float32)
    # History operands need smaller tiles to avoid register-limited occupancy.
    block_t = 16 if initial_conv is not None else 32
    grid = (2 * num_k_heads + num_v_heads, triton.cdiv(num_tokens, block_t) + query_start_loc.shape[0] - 1)
    _split_l2norm_qkv_gates_kernel[grid](
        mixed_qkv,
        q,
        k,
        v,
        g,
        beta,
        a,
        b,
        a_log,
        dt_bias,
        conv_weight, conv_states, query_start_loc, state_indices,
        mixed_qkv.stride(0), mixed_qkv.stride(1), conv_states.stride(0),
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        num_tokens, query_start_loc.shape[0] - 1,
        kernel_width=conv_weight.shape[1],
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        block_t=block_t,
        block_k=triton.next_power_of_2(max(head_k_dim, head_v_dim)),
        log_g=log_g,
        initial_conv=initial_conv,
        source_indices=source_indices,
        num_warps=4,
        num_stages=3,
    )
    return q, k, v, g, beta


@triton.jit(
    do_not_specialize=["num_state_slots"],
    do_not_specialize_on_alignment=["num_state_slots"],
)
def _scatter_recurrent_state_kernel(
    source,
    destination,
    state_indices,
    accepted_steps,
    stride_source,
    stride_destination,
    stride_source_layer,
    stride_destination_layer,
    stride_step,
    state_elements: tl.constexpr,
    num_state_slots,
    block: tl.constexpr,
    source_indices=None,
):
    batch = tl.program_id(0)
    # Speculative state pools can exceed 2^31 elements across layers.
    layer = tl.program_id(2).to(tl.int64)
    offsets = tl.program_id(1) * block + tl.arange(0, block)
    slot = batch if state_indices is None else tl.load(state_indices + batch).to(tl.int64)
    mask = (offsets < state_elements) & (slot >= 0) & (slot < num_state_slots)
    source += layer * stride_source_layer
    destination += layer * stride_destination_layer
    if accepted_steps is not None:
        step = tl.load(accepted_steps + batch)
        source += step * stride_step
        mask &= step >= 0
    source_slot = batch
    source_mask = mask
    if source_indices is not None:
        source_slot = tl.load(source_indices + batch).to(tl.int64)
        source_mask &= source_slot >= 0
    value = tl.load(
        source + source_slot * stride_source + offsets,
        mask=source_mask,
        other=0.0,
    )
    tl.store(
        destination + slot * stride_destination + offsets,
        value,
        mask=mask,
    )


def scatter_recurrent_state(
    source: torch.Tensor,
    destination: torch.Tensor,
    state_indices: torch.Tensor | None,
    accepted_steps: torch.Tensor = None,
    *,
    source_indices: torch.Tensor = None,
    layered: bool = False,
) -> None:
    """Scatter [batch, ...] final states into [slot, ...] state rows.

    With accepted_steps, source is [layer, batch, step, ...] and destination
    is [layer, slot, ...]. Steps are zero-based: step 0 includes the anchor.
    With state_indices=None, gather source_indices into consecutive rows.
    """
    if source.numel() == 0:
        return
    verifying = accepted_steps is not None or layered
    state_elements = destination[0, 0].numel() if verifying else source[0].numel()
    block = 1024
    _scatter_recurrent_state_kernel[
        ((state_indices if state_indices is not None else source_indices).shape[0],
         triton.cdiv(state_elements, block),
         source.shape[0] if verifying else 1)
    ](
        source,
        destination,
        state_indices,
        accepted_steps,
        source.stride(1 if verifying else 0),
        destination.stride(1 if verifying else 0),
        source.stride(0) if verifying else 0,
        destination.stride(0) if verifying else 0,
        source.stride(2) if accepted_steps is not None else 0,
        state_elements=state_elements,
        num_state_slots=destination.shape[1 if verifying else 0],
        block=block,
        source_indices=source_indices,
        num_warps=4,
    )


@triton.jit
def _gather_gdn_initial_states_kernel(
    ssm, conv, ssm_slots, conv_slots, initial_ssm, initial_conv,
    ssm_stride, conv_stride,
    ssm_elements: tl.constexpr, conv_elements: tl.constexpr, block: tl.constexpr,
):
    row, tile = tl.program_id(0), tl.program_id(1)
    offsets = tl.arange(0, block)
    if tile < tl.cdiv(ssm_elements, block):
        offset = tile * block + offsets
        slot = tl.load(ssm_slots + row).to(tl.int64)
        value = tl.load(ssm + slot * ssm_stride + offset,
                        (slot >= 0) & (offset < ssm_elements), other=0)
        tl.store(initial_ssm + row * ssm_elements + offset, value, offset < ssm_elements)
    else:
        offset = (tile - tl.cdiv(ssm_elements, block)) * block + offsets
        slot = tl.load(conv_slots + row).to(tl.int64)
        conv_value = tl.load(conv + slot * conv_stride + offset,
                        (slot >= 0) & (offset < conv_elements), other=0)
        tl.store(initial_conv + row * conv_elements + offset, conv_value, offset < conv_elements)


def _gather_gdn_initial_states(states, slots):
    """Pack immutable SSM/conv history in batch order with one transfer."""
    ssm, conv = states
    if slots is None:
        return ssm.float(), conv
    count = slots[0].numel()
    initial_ssm = ssm.new_empty((count, *ssm.shape[1:]), dtype=torch.float32)
    initial_conv = conv.new_empty((count, *conv.shape[1:]))
    ssm_elements, conv_elements = ssm[0].numel(), conv[0].numel()
    block = 1024
    _gather_gdn_initial_states_kernel[
        (count, triton.cdiv(ssm_elements, block) + triton.cdiv(conv_elements, block))
    ](
        ssm, conv, *slots, initial_ssm, initial_conv, ssm.stride(0), conv.stride(0),
        ssm_elements, conv_elements, block,
    )
    return initial_ssm, initial_conv


@triton.jit(
    do_not_specialize=["num_requests", "num_states", "num_outputs"],
    do_not_specialize_on_alignment=["num_requests", "num_states", "num_outputs"],
)
def _finish_gdn_prefill_kernel(
    states, conv, live_states, live_conv, live_slots, conv_slots,
    checkpoint_states, checkpoint_conv, checkpoint_slots, query_start_loc,
    output, result, owners, num_requests, num_states, num_outputs,
    state_elements: tl.constexpr, conv_elements: tl.constexpr,
    output_elements: tl.constexpr, block: tl.constexpr,
):
    """Commit contiguous SSM/conv rows and select shared output rows once."""
    tile = tl.program_id(0)
    ssm_blocks = tl.cdiv(state_elements, block)
    state_blocks = ssm_blocks + tl.cdiv(conv_elements, block)
    output_start = num_states * state_blocks
    offsets = tl.arange(0, block)
    if tile < output_start:
        row = tile // state_blocks
        state_tile = tile % state_blocks
        for is_conv in tl.static_range(2):
            if (state_tile >= ssm_blocks) == is_conv:
                if is_conv:
                    source, destination, indices = conv, live_conv, conv_slots
                    checkpoint = checkpoint_conv
                    elements = conv_elements
                    offset = (state_tile - ssm_blocks) * block + offsets
                else:
                    source, destination, indices = states, live_states, live_slots
                    checkpoint = checkpoint_states
                    elements = state_elements
                    offset = state_tile * block + offsets
                slot = tl.load(indices + row, row < num_requests, other=-1).to(tl.int64)
                if checkpoint_slots is not None:
                    if row >= num_requests:
                        destination = checkpoint
                        slot = tl.load(checkpoint_slots + row - num_requests).to(tl.int64)
                valid = (offset < elements) & (slot >= 0)
                if is_conv and query_start_loc is not None:
                    valid &= tl.load(query_start_loc + row + 1) > tl.load(query_start_loc + row)
                value = tl.load(source + row * elements + offset, valid, other=0)
                tl.store(destination + slot * elements + offset, value, valid)
    elif owners is not None:
        offset = (tile - output_start) * block + offsets
        valid = offset < num_outputs * output_elements
        owner = tl.load(owners + offset // output_elements, valid, other=0).to(tl.int64)
        output_value = tl.load(output + owner * output_elements + offset % output_elements, valid, other=0)
        tl.store(result + offset, output_value, valid)


def _finish_gdn_prefill(states, live_states, live_slots, output, plan, checkpoint_states, use_initial_state):
    owners = plan.owners if isinstance(plan.owners, torch.Tensor) else None
    result = output.new_empty((owners.numel(), *output.shape[1:])) if owners is not None else output[plan.owners]
    ssm, conv = states
    state_elements, conv_elements, output_elements = ssm[0].numel(), conv[0].numel(), output[0].numel()
    block = 1024
    blocks = ssm.shape[0] * (triton.cdiv(state_elements, block) + triton.cdiv(conv_elements, block))
    if owners is not None:
        blocks += triton.cdiv(result.numel(), block)
    _finish_gdn_prefill_kernel[(blocks,)](
        *states, *live_states, *live_slots, *checkpoint_states,
        plan.internal_slots if plan.internal_slots.numel() else None,
        plan.indptr if use_initial_state else None, output, result, owners, live_slots[0].numel(), ssm.shape[0], result.shape[0],
        state_elements, conv_elements, output_elements, block,
    )
    return result


@triton.jit(
    do_not_specialize=["batch_size"],
    do_not_specialize_on_alignment=["batch_size"],
)
def _update_recurrent_state_indices_kernel(
    current_slots, request_slots, read_slots, write_slots, accepted_steps,
    batch_size, steps: tl.constexpr, prefill: tl.constexpr,
    block_b: tl.constexpr, block_t: tl.constexpr,
):
    batch = tl.arange(0, block_b)
    request = tl.load(request_slots + batch, batch < batch_size, other=-1)
    valid = (batch < batch_size) & (request > 0)
    if accepted_steps is not None:
        step = tl.load(accepted_steps + batch, batch < batch_size, other=-1)
        slot = tl.load(write_slots + batch * steps + step, valid & (step >= 0), other=-1)
        tl.store(current_slots + request, slot, valid & (step >= 0))
    else:
        base = (request - 1) * (steps + 1) + 1
        if prefill:
            slot = base
            tl.store(current_slots + request, slot, valid)
        else:
            slot = tl.load(current_slots + request, valid, other=-1)
            step = tl.arange(0, block_t)
            # Each request owns steps+1 slots. Never overwrite its input state.
            outputs = base[:, None] + (
                slot[:, None] - base[:, None] + 1 + step[None, :]
            ) % (steps + 1)
            tl.store(write_slots + batch[:, None] * steps + step[None, :],
                     tl.where(valid[:, None], outputs, -1),
                     (batch[:, None] < batch_size) & (step[None, :] < steps))
        tl.store(read_slots + batch, tl.where(valid, slot, -1), batch < batch_size)


def update_recurrent_state_indices(
    current_slots: torch.Tensor,
    request_slots: torch.Tensor,
    read_slots: torch.Tensor,
    write_slots: torch.Tensor,
    *,
    prefill: bool = False,
    accepted_steps: torch.Tensor = None,
) -> None:
    """Prepare or commit speculative SSM slots by index, without copying states."""
    batch_size = request_slots.shape[0]
    steps = write_slots.shape[1]
    _update_recurrent_state_indices_kernel[(1,)](
        current_slots, request_slots, read_slots, write_slots, accepted_steps,
        batch_size, steps, prefill,
        triton.next_power_of_2(current_slots.shape[0] - 1), triton.next_power_of_2(steps),
        num_warps=4,
    )


def gated_rmsnorm(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """RMSNorm followed by a SiLU output gate, fused into one launch."""
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    sf_kernel.gated_rmsnorm(out, x, gate, weight, eps)
    return out


@tg.jit
def _decode_convolve(
    projection,
    conv_states,
    conv_weight,
    slot,
    begin,
    step,
    columns,
    mask,
    stride_projection: gl.constexpr,
    stride_state: gl.constexpr,
    kernel_width: gl.constexpr,
    valid,
):
    x = gl.load(projection + (begin + step) * stride_projection + columns, mask, other=0)
    acc = gl.full(columns.shape, 0.0, gl.float32, columns.type.layout)
    for tap in gl.static_range(kernel_width - 1):
        hist = step + tap
        initial = gl.load(
            conv_states + slot * stride_state + columns * (kernel_width - 1) + hist,
            mask & valid & (hist < kernel_width - 1),
            other=0,
        )
        proposed = gl.load(
            projection + (begin + hist - kernel_width + 1) * stride_projection + columns,
            mask & valid & (hist >= kernel_width - 1),
            other=0,
        )
        old = gl.where(hist < kernel_width - 1, initial, proposed)
        weight = gl.load(conv_weight + columns * kernel_width + tap, mask, other=0)
        acc = acc + (old.to(gl.float32) * weight.to(gl.float32)).to(projection.dtype.element_ty).to(
            gl.float32
        )
    last = gl.load(conv_weight + columns * kernel_width + kernel_width - 1, mask, other=0)
    acc = acc + (x.to(gl.float32) * last.to(gl.float32)).to(projection.dtype.element_ty).to(
        gl.float32
    )
    silu = (acc / (1.0 + gl.exp(-acc))).to(projection.dtype.element_ty).to(gl.float32)
    return (gl.where(valid, silu, x.to(gl.float32)), x)


@tg.jit(do_not_specialize=["num_tokens"])
def _split_prepare_gdn_journal_kernel(
    projection,
    gates,
    conv_states,
    conv_weight,
    state_indices,
    cu_seqlens,
    a_log,
    dt_bias,
    mixed_qkv,
    output_gate,
    decays,
    betas,
    raw_journal,
    stride_projection: gl.constexpr,
    stride_gates: gl.constexpr,
    stride_state: gl.constexpr,
    state_slots: gl.constexpr,
    num_k_heads: gl.constexpr,
    num_v_heads: gl.constexpr,
    head_k_dim: gl.constexpr,
    head_v_dim: gl.constexpr,
    kernel_width: gl.constexpr,
    verify_bound: gl.constexpr,
    block_heads: gl.constexpr,
    layout_warps: gl.constexpr,
    num_tokens,
    stride_indices: gl.constexpr,
    equal_steps: gl.constexpr,
    ENABLE_PDL: gl.constexpr,
):
    if ENABLE_PDL:
        gl.inline_asm_elementwise(
            "griddepcontrol.wait; // $0", "=r", [], dtype=gl.int32, is_pure=False, pack=1
        )
    request = gl.program_id(0)
    head_tile = gl.program_id(1)
    step = gl.program_id(2)
    if cu_seqlens is not None:
        begin = gl.load(cu_seqlens + request)
        end = gl.minimum(gl.load(cu_seqlens + request + 1), num_tokens)
        active_tokens = gl.load(cu_seqlens + gl.num_programs(0))
    else:
        begin = request * equal_steps
        end = begin + equal_steps
        active_tokens = num_tokens
    token = begin + step
    slot = gl.load(state_indices + request * stride_indices).to(gl.int64)
    valid = (slot > 0) & (slot < state_slots)
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [layout_warps, 1], [1, 0])
    norm_layout: gl.constexpr = gl.BlockedLayout([1, 8], [8, 4], [layout_warps, 1], [1, 0])
    heads = head_tile * block_heads + gl.arange(0, block_heads, layout=gl.SliceLayout(1, layout))
    head_offsets = gl.arange(0, head_k_dim, layout=gl.SliceLayout(0, layout))
    columns = heads[:, None] * head_k_dim + head_offsets[None, :]
    mask = heads[:, None] < 2 * num_k_heads + num_v_heads
    qkv_dim: gl.constexpr = 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim
    if cu_seqlens is not None:
        pad_token = request * verify_bound + step
        if (pad_token >= active_tokens) & (pad_token < num_tokens):
            gl.store(mixed_qkv + pad_token * qkv_dim + columns, 0.0, mask)
            if head_tile * block_heads >= 2 * num_k_heads:
                value_heads = heads - 2 * num_k_heads
                gl.store(
                    output_gate
                    + (pad_token * num_v_heads + value_heads[:, None]) * head_v_dim
                    + head_offsets[None, :],
                    0.0,
                    mask,
                )
                gl.store(
                    decays + pad_token * num_v_heads + value_heads, 0.0, value_heads < num_v_heads
                )
                gl.store(
                    betas + pad_token * num_v_heads + value_heads, 0.0, value_heads < num_v_heads
                )
    if token >= end:
        return
    if ENABLE_PDL:
        gl.inline_asm_elementwise(
            "griddepcontrol.launch_dependents; // $0",
            "=r",
            [],
            dtype=gl.int32,
            is_pure=False,
            pack=1,
        )
    value, raw = _decode_convolve(
        projection,
        conv_states,
        conv_weight,
        slot,
        begin,
        step,
        columns,
        mask,
        stride_projection,
        stride_state,
        kernel_width,
        valid,
    )
    gl.store(raw_journal + (request * verify_bound + step) * qkv_dim + columns, raw, mask & valid)
    if head_tile * block_heads < 2 * num_k_heads:
        norm_heads = head_tile * block_heads + gl.arange(
            0, block_heads, layout=gl.SliceLayout(1, norm_layout)
        )
        norm_offsets = gl.arange(0, head_k_dim, layout=gl.SliceLayout(0, norm_layout))
        normalized = gl.convert_layout(value, norm_layout)
        factor = gl.rsqrt(gl.sum(normalized * normalized, 1) + 1e-06)
        if head_tile * block_heads < num_k_heads:
            factor = factor * head_k_dim ** (-0.5)
        normalized *= factor[:, None]
        gl.store(
            mixed_qkv + token * qkv_dim + norm_heads[:, None] * head_k_dim + norm_offsets[None, :],
            normalized,
            norm_heads[:, None] < 2 * num_k_heads,
        )
    else:
        gl.store(mixed_qkv + token * qkv_dim + columns, value, mask)
        value_heads = heads - 2 * num_k_heads
        z = gl.load(
            projection
            + token * stride_projection
            + qkv_dim
            + value_heads[:, None] * head_v_dim
            + head_offsets[None, :],
            mask,
            other=0,
        )
        gl.store(
            output_gate
            + (token * num_v_heads + value_heads[:, None]) * head_v_dim
            + head_offsets[None, :],
            z,
            mask,
        )
        gate_a = gl.load(
            gates + token * stride_gates + num_v_heads + value_heads,
            value_heads < num_v_heads,
            other=0,
        ).to(gl.float32)
        gate_b = gl.load(
            gates + token * stride_gates + value_heads, value_heads < num_v_heads, other=0
        ).to(gl.float32)
        decay_scale = gl.exp(
            gl.load(a_log + value_heads, value_heads < num_v_heads, other=0).to(gl.float32)
        )
        bias = gl.load(dt_bias + value_heads, value_heads < num_v_heads, other=0).to(gl.float32)
        softplus_input = gate_a + bias
        softplus = gl.where(
            softplus_input <= 20.0, gl.log(1.0 + gl.exp(softplus_input)), softplus_input
        )
        decay = gl.exp(-decay_scale * softplus)
        gl.store(decays + token * num_v_heads + value_heads, decay, value_heads < num_v_heads)
        gl.store(
            betas + token * num_v_heads + value_heads, tl.sigmoid(gate_b), value_heads < num_v_heads
        )


def _prepare_gdn_journal(
    x,
    ba,
    states,
    weight,
    indices,
    cu_seqlens,
    a_log,
    dt_bias,
    journal,
    num_k_heads,
    num_v_heads,
    head_k_dim,
    head_v_dim,
):
    tokens = x.shape[0]
    requests = indices.numel()
    bound = journal.shape[1]
    steps = tokens // requests if cu_seqlens is None else bound
    dim = 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim
    mixed = torch.empty((tokens, dim), dtype=torch.float32, device=x.device)
    gate = torch.empty((tokens, num_v_heads, head_v_dim), dtype=x.dtype, device=x.device)
    decay = torch.empty((tokens, num_v_heads), dtype=torch.float32, device=x.device)
    beta = torch.empty_like(decay)
    enable_pdl = torch.version.hip is None and torch.cuda.get_device_capability(x.device)[0] >= 9
    _split_prepare_gdn_journal_kernel[
        requests, triton.cdiv(2 * num_k_heads + num_v_heads, 4), bound
    ](
        x,
        ba,
        states,
        weight,
        indices,
        cu_seqlens,
        a_log,
        dt_bias,
        mixed,
        gate,
        decay,
        beta,
        journal,
        x.stride(0),
        ba.stride(0),
        states.stride(0),
        states.shape[0],
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        weight.shape[1],
        bound,
        4,
        2,
        tokens,
        indices.stride(0),
        steps,
        enable_pdl,
        num_warps=2,
        num_stages=2,
        launch_pdl=enable_pdl,
    )
    return (mixed, gate, decay, beta)

@triton.jit
def _fused_qkvzba_conv_decode_kernel(
    mixed_qkv,
    z,
    b,
    a,
    projected_qkvz,
    projected_ba,
    conv_states,
    conv_weight,
    state_indices,
    conv_journal,
    cu_seqlens,
    stride_qkvz,
    stride_ba,
    stride_state,
    stride_state_dim,
    stride_state_pos,
    stride_weight_dim,
    stride_indices,
    qkv_dim: tl.constexpr,
    value_dim: tl.constexpr,
    num_v_heads: tl.constexpr,
    num_state_slots: tl.constexpr,
    kernel_width: tl.constexpr,
    steps: tl.constexpr,
    cache_steps: tl.constexpr,
    block: tl.constexpr,
):
    if cu_seqlens is not None:
        batch = tl.program_id(0)
        step = tl.program_id(2)
        begin = tl.load(cu_seqlens + batch)
        end = tl.load(cu_seqlens + batch + 1)
        token = begin + step
        if token >= end:
            return
        cache_token = batch * cache_steps + step
    else:
        token = tl.program_id(0)
        batch = token // steps
        step = token % steps
        begin = batch * steps
        cache_token = batch * cache_steps + step
    offsets = tl.program_id(1) * block + tl.arange(0, block)
    qkv_mask = offsets < qkv_dim
    x = tl.load(
        projected_qkvz + token * stride_qkvz + offsets,
        mask=qkv_mask,
        other=0.0,
    )

    slot = tl.load(state_indices + batch * stride_indices).to(tl.int64)
    valid_slot = (slot > 0) & (slot < num_state_slots)
    state = conv_states + slot * stride_state + offsets * stride_state_dim
    acc = tl.zeros((block,), dtype=tl.float32)
    for pos in tl.static_range(kernel_width - 1):
        history_pos = step + pos
        state_value = tl.load(
            state + history_pos * stride_state_pos,
            mask=qkv_mask & valid_slot & (history_pos < kernel_width - 1),
            other=0.0,
        )
        if conv_journal is not None:
            proposed_value = tl.load(
                projected_qkvz + (begin + history_pos - kernel_width + 1)
                * stride_qkvz + offsets,
                mask=qkv_mask & valid_slot & (history_pos >= kernel_width - 1),
                other=0.0,
            )
            state_value = tl.where(history_pos < kernel_width - 1, state_value, proposed_value)
        weight_value = tl.load(
            conv_weight + offsets * stride_weight_dim + pos,
            mask=qkv_mask,
            other=0.0,
        )
        acc += state_value * weight_value
    last_weight = tl.load(
        conv_weight + offsets * stride_weight_dim + kernel_width - 1,
        mask=qkv_mask,
        other=0.0,
    )
    acc += x * last_weight
    conv_out = acc / (1.0 + tl.exp(-acc))
    tl.store(
        mixed_qkv + token * qkv_dim + offsets,
        tl.where(valid_slot, conv_out, x),
        mask=qkv_mask,
    )

    if conv_journal is not None:
        # Keep only raw inputs. Commit reconstructs the last history window.
        tl.store(conv_journal + cache_token * qkv_dim + offsets, x,
                 mask=qkv_mask & valid_slot)
    else:
        for pos in tl.static_range(kernel_width - 2):
            next_value = tl.load(
                state + (pos + 1) * stride_state_pos,
                mask=qkv_mask & valid_slot,
                other=0.0,
            )
            tl.store(
                state + pos * stride_state_pos,
                next_value,
                mask=qkv_mask & valid_slot,
            )
        if kernel_width > 1:
            tl.store(
                state + (kernel_width - 2) * stride_state_pos,
                x,
                mask=qkv_mask & valid_slot,
            )

    z_mask = offsets < value_dim
    z_value = tl.load(
        projected_qkvz + token * stride_qkvz + qkv_dim + offsets,
        mask=z_mask,
        other=0.0,
    )
    tl.store(z + token * value_dim + offsets, z_value, mask=z_mask)

    gate_mask = offsets < num_v_heads
    tl.store(
        b + token * num_v_heads + offsets,
        tl.load(
            projected_ba + token * stride_ba + offsets,
            mask=gate_mask,
            other=0.0,
        ),
        mask=gate_mask,
    )
    tl.store(
        a + token * num_v_heads + offsets,
        tl.load(
            projected_ba + token * stride_ba + num_v_heads + offsets,
            mask=gate_mask,
            other=0.0,
        ),
        mask=gate_mask,
    )


def fused_qkvzba_conv_decode(
    projected_qkvz: torch.Tensor,
    projected_ba: torch.Tensor,
    conv_states: torch.Tensor,
    conv_weight: torch.Tensor,
    state_indices: torch.Tensor,
    num_v_heads: int,
    head_v_dim: int,
    conv_journal: torch.Tensor = None,
    cu_seqlens: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse QKVZ/BA unpacking with indexed causal-Conv1D decode."""
    batch = projected_qkvz.shape[0]
    value_dim = num_v_heads * head_v_dim
    qkv_dim = projected_qkvz.shape[1] - value_dim
    mixed_qkv = torch.empty(
        (batch, qkv_dim), dtype=projected_qkvz.dtype, device=projected_qkvz.device
    )
    z = torch.empty(
        (batch, num_v_heads, head_v_dim),
        dtype=projected_qkvz.dtype,
        device=projected_qkvz.device,
    )
    b = torch.empty(
        (batch, num_v_heads), dtype=projected_ba.dtype, device=projected_ba.device
    )
    a = torch.empty_like(b)
    block = 256
    cache_steps = conv_journal.shape[1] if conv_journal is not None else 1
    equal_steps = 1
    if cu_seqlens is None and conv_journal is not None:
        requests = state_indices.numel()
        if not requests or batch % requests or not 0 < batch // requests <= cache_steps:
            raise ValueError("Equal-width convolution must fit its verification journal")
        equal_steps = batch // requests
    if cu_seqlens is not None:
        if conv_journal is None:
            raise ValueError("Packed varlen convolution requires a conv journal")
        grid = (state_indices.numel(), triton.cdiv(qkv_dim, block), cache_steps)
    else:
        grid = (batch, triton.cdiv(qkv_dim, block))
    _fused_qkvzba_conv_decode_kernel[
        grid
    ](
        mixed_qkv,
        z,
        b,
        a,
        projected_qkvz,
        projected_ba,
        conv_states,
        conv_weight,
        state_indices,
        conv_journal,
        cu_seqlens,
        projected_qkvz.stride(0),
        projected_ba.stride(0),
        conv_states.stride(0),
        conv_states.stride(1),
        conv_states.stride(2),
        conv_weight.stride(0),
        state_indices.stride(0),
        qkv_dim=qkv_dim,
        value_dim=value_dim,
        num_v_heads=num_v_heads,
        num_state_slots=conv_states.shape[0],
        kernel_width=conv_weight.shape[1],
        steps=equal_steps,
        cache_steps=cache_steps,
        block=block,
        num_warps=4 if cu_seqlens is not None else 8,
        num_stages=2,
    )
    return mixed_qkv, z, b, a


# Adapted from SGLang/FLA's packed recurrent GDN decode kernel.  Keeping the
# packed QKV input avoids three materialization kernels in every GDN layer.
@triton.jit
def _gdn_verify_request_order_kernel(cu_seqlens, order, batch: tl.constexpr, block: tl.constexpr):
    row = tl.arange(0, block)
    begin = tl.load(cu_seqlens + row, row < batch, other=0).to(tl.int64)
    end = tl.load(cu_seqlens + row + 1, row < batch, other=0).to(tl.int64)
    # Descending length, with original row order as the tie breaker.
    key = tl.where(row < batch, (end - begin) * block + block - 1 - row, -1)
    key = tl.sort(key, descending=True)
    tl.store(order + row, block - 1 - key % block, row < batch)


def gdn_verify_request_order(cu_seqlens: torch.Tensor) -> torch.Tensor:
    """Schedule long recurrences first; share this order across target layers."""
    batch = cu_seqlens.numel() - 1
    order = torch.empty(batch, dtype=torch.int32, device=cu_seqlens.device)
    if batch:
        _gdn_verify_request_order_kernel[(1,)](
            cu_seqlens, order, batch, triton.next_power_of_2(batch), num_warps=4,
        )
    return order


@triton.jit
def _packed_gdn_decode_kernel(
    mixed_qkv,
    a,
    b,
    a_log,
    dt_bias,
    output,
    states,
    state_indices,
    output_indices,
    cu_seqlens,
    request_order,
    stride_output_indices: tl.constexpr,
    stride_qkv: tl.constexpr,
    stride_a: tl.constexpr,
    stride_b: tl.constexpr,
    stride_state: tl.constexpr,
    stride_indices: tl.constexpr,
    scale,
    num_k_heads: tl.constexpr,
    num_v_heads: tl.constexpr,
    head_k_dim: tl.constexpr,
    head_v_dim: tl.constexpr,
    steps: tl.constexpr,
    block_k: tl.constexpr,
    block_v: tl.constexpr,
):
    value_tile = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // num_v_heads
    if request_order is not None:
        batch = tl.load(request_order + batch)
    value_head = batch_head % num_v_heads
    key_head = value_head // (num_v_heads // num_k_heads)
    if cu_seqlens is not None:
        begin = tl.load(cu_seqlens + batch)
        length = tl.load(cu_seqlens + batch + 1) - begin
    else:
        begin = batch * steps
        length = steps

    key_offsets = tl.arange(0, block_k)
    value_offsets = value_tile * block_v + tl.arange(0, block_v)
    key_mask = key_offsets < head_k_dim
    value_mask = value_offsets < head_v_dim
    state_mask = value_mask[:, None] & key_mask[None, :]

    slot = tl.load(state_indices + batch * stride_indices).to(tl.int64)
    if slot <= 0:
        for step in range(length):
            token = begin + step
            output_ptr = output + (token * num_v_heads + value_head) * head_v_dim
            tl.store(output_ptr + value_offsets, 0.0, mask=value_mask)
        return

    state_offsets = (
        value_head * head_v_dim * head_k_dim
        + value_offsets[:, None] * head_k_dim
        + key_offsets[None, :]
    )
    state_ptr = states + slot * stride_state + state_offsets
    state = tl.load(state_ptr, mask=state_mask, other=0.0).to(tl.float32)
    for step in range(length):
        token = begin + step
        qkv_ptr = mixed_qkv + token * stride_qkv
        q = tl.load(
            qkv_ptr + key_head * head_k_dim + key_offsets,
            mask=key_mask,
            other=0.0,
        ).to(tl.float32)
        k = tl.load(
            qkv_ptr + num_k_heads * head_k_dim + key_head * head_k_dim + key_offsets,
            mask=key_mask,
            other=0.0,
        ).to(tl.float32)
        v = tl.load(
            qkv_ptr
            + 2 * num_k_heads * head_k_dim
            + value_head * head_v_dim
            + value_offsets,
            mask=value_mask,
            other=0.0,
        ).to(tl.float32)

        q *= tl.rsqrt(tl.sum(q * q, axis=0) + 1e-6) * scale
        k *= tl.rsqrt(tl.sum(k * k, axis=0) + 1e-6)
        gate_a = tl.load(a + token * stride_a + value_head).to(tl.float32)
        gate_b = tl.load(b + token * stride_b + value_head).to(tl.float32)
        decay_log = tl.load(a_log + value_head).to(tl.float32)
        bias = tl.load(dt_bias + value_head).to(tl.float32)
        softplus_arg = gate_a + bias
        softplus = tl.where(
            softplus_arg <= 20.0,
            tl.log(1.0 + tl.exp(softplus_arg)),
            softplus_arg,
        )
        state *= tl.exp(-tl.exp(decay_log) * softplus)
        prediction = tl.sum(state * k[None, :], axis=1)
        correction = (v - prediction) * tl.sigmoid(gate_b)
        state += correction[:, None] * k[None, :]
        result = tl.sum(state * q[None, :], axis=1)
        output_ptr = output + (token * num_v_heads + value_head) * head_v_dim
        tl.store(output_ptr + value_offsets, result, mask=value_mask)
        if output_indices is not None:
            write_offset = batch * stride_output_indices + step if cu_seqlens is not None else token
            write_slot = tl.load(output_indices + write_offset).to(tl.int64)
            state_ptr = states + write_slot * stride_state + state_offsets
        # Verification writes several large snapshots; only the accepted one
        # will be read again. Evict these streaming writes before model weights.
        tl.store(
            state_ptr, state, mask=state_mask,
            cache_modifier=".cs" if steps > 1 and states.dtype.element_ty == tl.bfloat16 else "",
        )
        # Match the state precision of consecutive single-token decode calls.
        state = state.to(states.dtype.element_ty).to(tl.float32)


def packed_gdn_decode(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    states: torch.Tensor,
    state_indices: torch.Tensor,
    num_k_heads: int,
    ssm_output_indices: torch.Tensor = None,
    cu_seqlens: torch.Tensor = None,
    request_order: torch.Tensor = None,
) -> torch.Tensor:
    num_tokens = mixed_qkv.shape[0]
    steps = ssm_output_indices.shape[1] if ssm_output_indices is not None else 1
    if cu_seqlens is not None:
        if ssm_output_indices is None:
            raise ValueError("Packed varlen GDN requires verification output slots")
        batch = state_indices.numel()
    else:
        batch = num_tokens // steps
    num_v_heads, head_v_dim, head_k_dim = states.shape[-3:]
    output = torch.empty(
        (num_tokens, num_v_heads, head_v_dim),
        dtype=mixed_qkv.dtype,
        device=mixed_qkv.device,
    )
    block_k = triton.next_power_of_2(head_k_dim)
    # Smaller tiles reduce register pressure for BF16 multi-token verification.
    tile_v = 16 if steps > 1 and states.dtype == torch.bfloat16 else 32
    block_v = min(triton.next_power_of_2(head_v_dim), tile_v)
    _packed_gdn_decode_kernel[(triton.cdiv(head_v_dim, block_v), batch * num_v_heads)](
        mixed_qkv,
        a,
        b,
        a_log,
        dt_bias,
        output,
        states,
        state_indices,
        ssm_output_indices,
        cu_seqlens,
        request_order,
        ssm_output_indices.stride(0) if ssm_output_indices is not None else 0,
        mixed_qkv.stride(0),
        a.stride(0),
        b.stride(0),
        states.stride(0),
        state_indices.stride(0),
        head_k_dim**-0.5,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        steps=steps,
        block_k=block_k,
        block_v=block_v,
        num_warps=1,
        num_stages=3,
    )
    return output


@triton.jit
def _packed_gdn_prefill_reference_kernel(
    q,
    k,
    v,
    g,
    beta,
    output,
    states,
    state_indices,
    cu_seqlens,
    stride_state,
    scale,
    num_k_heads: tl.constexpr,
    num_v_heads: tl.constexpr,
    head_k_dim: tl.constexpr,
    head_v_dim: tl.constexpr,
    block_k: tl.constexpr,
    block_v: tl.constexpr,
):
    value_tile = tl.program_id(0)
    seq_head = tl.program_id(1)
    seq = seq_head // num_v_heads
    value_head = seq_head % num_v_heads
    key_head = value_head // (num_v_heads // num_k_heads)
    begin = tl.load(cu_seqlens + seq).to(tl.int64)
    end = tl.load(cu_seqlens + seq + 1).to(tl.int64)
    slot = tl.load(state_indices + seq).to(tl.int64)

    ko = tl.arange(0, block_k)
    vo = value_tile * block_v + tl.arange(0, block_v)
    km = ko < head_k_dim
    vm = vo < head_v_dim
    sm = vm[:, None] & km[None, :]
    state_ptr = states + slot * stride_state
    state_ptr += (
        value_head * head_v_dim * head_k_dim
        + vo[:, None] * head_k_dim
        + ko[None, :]
    )
    state = tl.zeros((block_v, block_k), dtype=tl.float32)

    token = begin
    while token < end:
        q_ptr = q + (token * num_k_heads + key_head) * head_k_dim
        k_ptr = k + (token * num_k_heads + key_head) * head_k_dim
        v_ptr = v + (token * num_v_heads + value_head) * head_v_dim
        qv = tl.load(q_ptr + ko, mask=km, other=0.0).to(tl.float32) * scale
        kv = tl.load(k_ptr + ko, mask=km, other=0.0).to(tl.float32)
        vv = tl.load(v_ptr + vo, mask=vm, other=0.0).to(tl.float32)
        state *= tl.load(g + token * num_v_heads + value_head).to(tl.float32)
        vv -= tl.sum(state * kv[None, :], axis=1)
        vv *= tl.load(beta + token * num_v_heads + value_head).to(tl.float32)
        state += vv[:, None] * kv[None, :]
        out = tl.sum(state * qv[None, :], axis=1)
        out_ptr = output + (token * num_v_heads + value_head) * head_v_dim
        tl.store(out_ptr + vo, out, mask=vm)
        token += 1
    tl.store(state_ptr, state, mask=sm)


def packed_gdn_prefill_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    states: torch.Tensor,
    state_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    """Serial FP32 recurrence for accuracy checks; g contains decay factors.

    Pass FP32 states to retain the reference's unrounded final state.
    """
    num_k_heads, head_k_dim = q.shape[-2:]
    num_v_heads, head_v_dim = v.shape[-2:]
    output = torch.empty_like(v)
    block_k = triton.next_power_of_2(head_k_dim)
    block_v = min(triton.next_power_of_2(head_v_dim), 32)
    grid = (triton.cdiv(head_v_dim, block_v), state_indices.shape[0] * num_v_heads)
    _packed_gdn_prefill_reference_kernel[grid](
        q,
        k,
        v,
        g,
        beta,
        output,
        states,
        state_indices,
        cu_seqlens,
        states.stride(0),
        head_k_dim**-0.5,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        block_k=block_k,
        block_v=block_v,
        num_warps=1,
        num_stages=3,
    )
    return output


def packed_gdn_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    initial_state: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """FLA chunked prefill with optional initial state; gates are natural logarithms.

    Returns output and FP32 final state in V-first layout; the caller commits it.
    cu_seqlens must be immutable for the forward; FLA caches its chunk indices.
    """
    _, output, _, final_state, _, _ = chunk_gated_delta_rule_fwd(
        q=q.unsqueeze(0), k=k.unsqueeze(0), v=v.unsqueeze(0),
        g=log_g.unsqueeze(0), beta=beta.unsqueeze(0), scale=q.shape[-1] ** -0.5,
        initial_state=initial_state, output_final_state=True,
        state_v_first=True, cu_seqlens=cu_seqlens,
    )
    return output.squeeze(0), final_state


class GatedDeltaNetBackend:
    """Fused GDN execution over model-owned recurrent state buffers."""

    def __init__(self, prefill_backend: str, decode_backend: str, *, use_initial_state: bool = False):
        supported = {"flashinfer", "triton"}
        if prefill_backend not in supported or decode_backend not in supported:
            raise ValueError("GDN backends must be one of: flashinfer, triton")
        if "flashinfer" in (prefill_backend, decode_backend) and gdn_decode is None:
            raise ImportError(
                "FlashInfer GDN requires sfllm[flashinfer]; without FlashInfer, "
                "select --linear-attn-backend triton."
            )
        self.prefill_backend = prefill_backend
        self.decode_backend = decode_backend
        self.use_initial_state = use_initial_state
        self.prefill_plan = None
        self.prefill_initial_states = None
        self.checkpoint_states = None

    @staticmethod
    def prepare_verify(forward_batch) -> None:
        # Rebuild once per forward. Graph replay recomputes the order from the
        # current GPU boundaries; all GDN layers share this captured operation.
        forward_batch._gdn_verify_order = gdn_verify_request_order(forward_batch.qo_indptr)

    def prefill(
        self,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        conv_weight: torch.Tensor,
        conv_states: torch.Tensor,
        ssm_states: torch.Tensor,
        state_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        query_start_loc_i64: torch.Tensor,
        a_log: torch.Tensor,
        dt_bias: torch.Tensor,
        num_k_heads: int,
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        ssm_state_indices: torch.Tensor = None,
    ) -> torch.Tensor:
        use_flashinfer = self.prefill_backend == "flashinfer"
        if use_flashinfer and not (
            head_k_dim == head_v_dim == 128
            and ssm_states.dtype in (torch.float32, torch.bfloat16)
            and mixed_qkv.dtype in (torch.float16, torch.bfloat16)
        ):
            raise ValueError("FlashInfer GDN prefill requires K=V=128, FP16/BF16 inputs and FP32/BF16 states.")
        plan = self.prefill_plan
        initial_ssm = initial_conv = None
        if self.use_initial_state:
            if self.prefill_initial_states is not None:
                ssm, conv, slots = self.prefill_initial_states
                initial_ssm, initial_conv = _gather_gdn_initial_states(
                    (ssm, conv), (slots, slots) if slots is not None else None,
                )
            else:
                slots = (ssm_state_indices if ssm_state_indices is not None else state_indices,
                         state_indices)
                if plan is not None:
                    slots = tuple(indices[plan.rows] for indices in slots)
                initial_ssm, initial_conv = _gather_gdn_initial_states((ssm_states, conv_states), slots)
        if plan is not None:
            original_states = (ssm_states, conv_states)
            original_slots = (ssm_state_indices if ssm_state_indices is not None else state_indices,
                              state_indices)
            conv_states = conv_states.new_empty((plan.dense_rows.numel(), *conv_states.shape[1:]))
            state_indices = plan.dense_rows
            ssm_state_indices = None
            query_start_loc = query_start_loc_i64 = plan.indptr

        q, k, v, g, beta = split_l2norm_qkv_gates(
            mixed_qkv,
            a,
            b,
            a_log,
            dt_bias,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            conv_weight=conv_weight,
            conv_states=conv_states,
            query_start_loc=query_start_loc,
            state_indices=state_indices,
            log_g=not use_flashinfer,
            initial_conv=initial_conv,
            source_indices=plan.expanded if plan is not None else None,
        )
        if ssm_state_indices is not None:
            state_indices = ssm_state_indices
        if not use_flashinfer:
            output, output_state = packed_gdn_prefill(
                q,
                k,
                v,
                g,
                beta,
                query_start_loc_i64,
                initial_state=initial_ssm,
            )

        else:
            # CP accumulates in FP32; the final scatter rounds to the cache dtype.
            output_state = torch.empty(
                (state_indices.shape[0], *ssm_states.shape[1:]),
                dtype=torch.float32, device=ssm_states.device,
            )
            output, output_state = chunk_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=initial_ssm,
                output_final_state=True,
                cu_seqlens=query_start_loc_i64,
                use_qk_l2norm_in_kernel=False,
                output_state=output_state,
                use_cp=True,
            )
        if plan is not None:
            return _finish_gdn_prefill(
                (output_state, conv_states), original_states, original_slots, output, plan,
                self.checkpoint_states if plan.internal_slots.numel() else (None, None), self.use_initial_state,
            )
        scatter_recurrent_state(output_state, ssm_states, state_indices)
        return output

    def decode(
        self,
        projected_qkvz: torch.Tensor,
        projected_ba: torch.Tensor,
        *,
        conv_weight: torch.Tensor,
        conv_states: torch.Tensor,
        ssm_states: torch.Tensor,
        state_indices: torch.Tensor,
        a_log: torch.Tensor,
        dt_bias: torch.Tensor,
        num_k_heads: int,
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        conv_journal: torch.Tensor = None,
        ssm_state_indices: torch.Tensor = None,
        ssm_output_indices: torch.Tensor = None,
        ssm_journal: GDNJournal = None,
        cu_seqlens: torch.Tensor = None,
        forward_batch=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        request_order = None
        if cu_seqlens is not None:
            request_order = (forward_batch._gdn_verify_order if forward_batch is not None
                             else gdn_verify_request_order(cu_seqlens))
        verify_bound = conv_journal.shape[1] if conv_journal is not None else None
        requests = state_indices.numel()
        fast_journal = (
            ssm_journal is not None and conv_journal is not None
            and projected_qkvz.is_cuda and torch.version.hip is None
            and 0 < projected_qkvz.shape[0] <= 192
            and projected_qkvz.dtype == torch.bfloat16
            and torch.cuda.get_device_capability(projected_qkvz.device)[0] >= 9
            and head_k_dim == head_v_dim == 128
            and num_k_heads > 0
            and num_k_heads % 4 == num_v_heads % 4 == 0
            and num_v_heads % num_k_heads == 0
            and conv_weight.ndim == 2
            and conv_weight.shape[0] == 2 * num_k_heads * head_k_dim
                                        + num_v_heads * head_v_dim
            and conv_weight.shape[1] == conv_states.shape[-1] + 1
            and conv_weight.is_contiguous() and conv_states.is_contiguous()
            and conv_journal.is_contiguous() and ssm_states.is_contiguous()
            and ssm_states.dtype == torch.float32 and requests > 0
            and conv_journal.shape[0] >= requests
            and projected_qkvz.stride(-1) == projected_ba.stride(-1) == 1
            and projected_qkvz.shape[1] == 2 * num_k_heads * head_k_dim
                                          + 2 * num_v_heads * head_v_dim
            and projected_ba.shape == (projected_qkvz.shape[0], 2 * num_v_heads)
            and (cu_seqlens is None or (cu_seqlens.numel() == requests + 1
                                       and cu_seqlens.is_contiguous()))
            and projected_qkvz.shape[0] <= requests * verify_bound
            and (cu_seqlens is not None or projected_qkvz.shape[0] % requests == 0)
        )
        if fast_journal:
            mixed_qkv, z, decay, beta = _prepare_gdn_journal(
                projected_qkvz, projected_ba, conv_states, conv_weight,
                state_indices, cu_seqlens, a_log, dt_bias, conv_journal,
                num_k_heads, num_v_heads, head_k_dim, head_v_dim,
            )
            core = packed_gdn_journal_verify(
                mixed_qkv, decay, beta, a_log, dt_bias, ssm_states,
                ssm_state_indices if ssm_state_indices is not None else state_indices,
                num_k_heads, ssm_journal, cu_seqlens, request_order,
                precomputed=True, output_dtype=projected_qkvz.dtype,
            )
            return core, z
        mixed_qkv, z, b, a = fused_qkvzba_conv_decode(
            projected_qkvz,
            projected_ba,
            conv_states,
            conv_weight,
            state_indices,
            num_v_heads,
            head_v_dim,
            conv_journal,
            cu_seqlens,
        )
        if ssm_state_indices is not None:
            state_indices = ssm_state_indices
        steps = ssm_output_indices.shape[1] if ssm_output_indices is not None else 1
        if ssm_journal is not None:
            core = packed_gdn_journal_verify(
                mixed_qkv, a, b, a_log, dt_bias, ssm_states, state_indices,
                num_k_heads, ssm_journal, cu_seqlens, request_order,
            )
        # BF16 verification has one implementation, independent of ordinary decode.
        elif cu_seqlens is not None or self.decode_backend == "triton" or (
            ssm_output_indices is not None and ssm_states.dtype == torch.bfloat16
        ):
            core = packed_gdn_decode(
                mixed_qkv=mixed_qkv,
                a=a,
                b=b,
                a_log=a_log,
                dt_bias=dt_bias,
                states=ssm_states,
                state_indices=state_indices,
                num_k_heads=num_k_heads,
                ssm_output_indices=ssm_output_indices,
                cu_seqlens=cu_seqlens,
                request_order=request_order,
            )
        else:
            recurrent = (
                gdn_decode.gated_delta_rule_mtp
                if ssm_output_indices is not None
                else gdn_decode.gated_delta_rule_decode_pretranspose
            )
            state_options = (
                dict(ssm_state_indices=ssm_output_indices, disable_state_update=False)
                if ssm_output_indices is not None else dict(state=None)
            )
            key_dim = num_k_heads * head_k_dim
            value_dim = num_v_heads * head_v_dim
            q, k, v = mixed_qkv.split((key_dim, key_dim, value_dim), dim=-1)
            core, _ = recurrent(
                q=q.view(-1, steps, num_k_heads, head_k_dim),
                k=k.view(-1, steps, num_k_heads, head_k_dim),
                v=v.view(-1, steps, num_v_heads, head_v_dim),
                output=v.new_empty((v.shape[0] // steps, steps, num_v_heads, head_v_dim)),
                A_log=a_log,
                a=a.view(-1, steps, num_v_heads),
                dt_bias=dt_bias,
                b=b.view(-1, steps, num_v_heads),
                use_qk_l2norm=True,
                initial_state=ssm_states,
                initial_state_indices=state_indices,
                **state_options,
            )
            core = core.view(-1, num_v_heads, head_v_dim)
        return core, z
