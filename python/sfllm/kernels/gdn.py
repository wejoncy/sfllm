"""Gated DeltaNet backends and fused preprocessing kernels for Qwen3.5."""

from __future__ import annotations

from inspect import signature
from typing import Tuple

import torch
import triton
import triton.language as tl
from triton.experimental import gluon as tg
from triton.experimental.gluon import language as gl
import sf_kernel


def _load_gdn_kernels(prefill_backend: str, decode_backend: str):
    chunk_gated_delta_rule = None
    gdn_decode = None
    if prefill_backend == decode_backend == "triton":
        return chunk_gated_delta_rule, gdn_decode
    if torch.version.cuda is None:
        return chunk_gated_delta_rule, gdn_decode
    arch = torch.cuda.get_device_capability()[0]
    if arch < 9:
        return chunk_gated_delta_rule, gdn_decode
    if prefill_backend == "flashinfer":
        try:
            from flashinfer import gdn_prefill

            prefill = getattr(gdn_prefill, "chunk_gated_delta_rule", None)
            if (
                prefill is not None
                and {"use_cp", "output_state"} <= signature(prefill).parameters.keys()
                and getattr(gdn_prefill, f"cp_delta_rule_dsl_sm{arch}0", None) is not None
                # FlashInfer's SM100 CP kernel requires CUDA 13.
                and (arch != 10 or int(torch.version.cuda.split(".")[0]) >= 13)
            ):
                chunk_gated_delta_rule = prefill
        except (ImportError, RuntimeError):
            pass
    if decode_backend == "flashinfer":
        try:
            from flashinfer import gdn_decode
        except (ImportError, RuntimeError):
            pass
    return chunk_gated_delta_rule, gdn_decode


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
    block_k: gl.constexpr,
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
            state_x = gl.load(
                mixed_qkv + state_token[:, None] * stride_token + input_columns[None, :] * stride_dim,
                mask=state_mask & (state_token[:, None] >= state_seq_start), other=0,
            )
            gl.store(
                conv_states + state_slot * stride_state + input_columns[None, :] * (kernel_width - 1)
                + state_tap[:, None], state_x, mask=state_mask,
            )
    else:
        token = tile * block_t + gl.arange(0, block_t, layout=gl.SliceLayout(1, layout))
        token_mask = token < num_tokens
        # Find each token's packed sequence without constructing a token-to-row map.
        lo = gl.full((block_t,), 0, gl.int32, gl.SliceLayout(1, layout))
        hi = gl.full((block_t,), num_sequences, gl.int32, gl.SliceLayout(1, layout))
        remaining = num_sequences
        while remaining > 0:
            remaining = remaining // 2
            mid = (lo + hi) // 2
            end = gl.load(query_start_loc + mid + 1, mask=mid < num_sequences, other=num_tokens)
            right = token >= end
            lo = gl.where((lo < hi) & right, mid + 1, lo)
            hi = gl.where((lo < hi) & ~right, mid, hi)
        seq_start = gl.load(query_start_loc + lo, mask=token_mask, other=0)

        mask = token_mask[:, None] & (d[None, :] < dim)
        value = gl.full(mask.shape, 0.0, gl.float32, layout)
        for tap in gl.static_range(kernel_width):
            source_token = token + tap - kernel_width + 1
            x = gl.load(
                mixed_qkv + source_token[:, None] * stride_token + input_columns[None, :] * stride_dim,
                mask=mask & (source_token[:, None] >= seq_start[:, None]), other=0,
            ).to(gl.float32)
            weight = gl.load(
                conv_weight + input_columns * kernel_width + tap,
                mask=d < dim, other=0,
            ).to(gl.float32)
            value = gl.fma(x, weight[None, :], value)
        value = (value / (1.0 + gl.exp(-value))).to(mixed_qkv.dtype.element_ty).to(gl.float32)
        if head < 2 * num_k_heads:
            value *= gl.rsqrt(gl.sum(value * value, axis=1)[:, None] + 1e-6)
        gl.store(output + token[:, None] * heads * dim + columns[None, :], value, mask=mask)

        if head >= 2 * num_k_heads:
            head -= 2 * num_k_heads
            a_value = gl.load(a + token * stride_a_token + head * stride_a_head,
                              mask=token_mask, other=0).to(gl.float32)
            b_value = gl.load(b + token * stride_b_token + head * stride_b_head,
                              mask=token_mask, other=0).to(gl.float32)
            decay = gl.load(a_log + head).to(gl.float32)
            bias = gl.load(dt_bias + head).to(gl.float32)
            softplus_input = a_value + bias
            softplus = gl.where(softplus_input <= 20.0,
                                gl.log(1.0 + gl.exp(softplus_input)), softplus_input)
            gate_offset = token * num_v_heads + head
            gl.store(g + gate_offset, gl.exp(-gl.exp(decay) * softplus), mask=token_mask)
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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse packed prefill convolution, QK normalization and GDN gates."""
    num_tokens = mixed_qkv.shape[0]
    q = mixed_qkv.new_empty((num_tokens, num_k_heads, head_k_dim))
    k = torch.empty_like(q)
    v = mixed_qkv.new_empty((num_tokens, num_v_heads, head_v_dim))
    g = torch.empty_like(a, dtype=torch.float32)
    beta = torch.empty_like(b, dtype=torch.float32)
    block_t = 32
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
        num_warps=4,
        num_stages=3,
    )
    return q, k, v, g, beta


@triton.jit
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
    num_state_slots: tl.constexpr,
    block: tl.constexpr,
):
    batch = tl.program_id(0)
    # Speculative state pools can exceed 2^31 elements across layers.
    layer = tl.program_id(2).to(tl.int64)
    offsets = tl.program_id(1) * block + tl.arange(0, block)
    slot = tl.load(state_indices + batch).to(tl.int64)
    mask = (offsets < state_elements) & (slot >= 0) & (slot < num_state_slots)
    source += layer * stride_source_layer
    destination += layer * stride_destination_layer
    if accepted_steps is not None:
        step = tl.load(accepted_steps + batch)
        source += step * stride_step
        mask &= step >= 0
    value = tl.load(
        source + batch * stride_source + offsets,
        mask=mask,
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
    state_indices: torch.Tensor,
    accepted_steps: torch.Tensor = None,
) -> None:
    """Scatter [batch, ...] final states into [slot, ...] state rows.

    With accepted_steps, source is [layer, batch, step, ...] and destination
    is [layer, slot, ...]. Steps are zero-based: step 0 includes the anchor.
    """
    if source.numel() == 0:
        return
    verifying = accepted_steps is not None
    state_elements = destination[0, 0].numel() if verifying else source[0].numel()
    block = 1024
    _scatter_recurrent_state_kernel[
        (state_indices.shape[0], triton.cdiv(state_elements, block),
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
        source.stride(2) if verifying else 0,
        state_elements=state_elements,
        num_state_slots=destination.shape[1 if verifying else 0],
        block=block,
        num_warps=4,
    )


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
    intermediate_conv,
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
    block: tl.constexpr,
):
    token = tl.program_id(0)
    batch = token // steps
    step = token % steps
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
        if intermediate_conv is not None:
            proposed_value = tl.load(
                projected_qkvz + (batch * steps + history_pos - kernel_width + 1)
                * stride_qkvz + offsets,
                mask=qkv_mask & valid_slot & (history_pos >= kernel_width - 1),
                other=0.0,
            )
            state_value = tl.where(history_pos < kernel_width - 1, state_value, proposed_value)
            if pos > 0:
                tl.store(intermediate_conv + (token * qkv_dim + offsets) * (kernel_width - 1) + pos - 1,
                         state_value, mask=qkv_mask & valid_slot)
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

    if intermediate_conv is not None:
        state = intermediate_conv + (token * qkv_dim + offsets) * (kernel_width - 1)
        stride_state_pos = 1
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
    intermediate_conv: torch.Tensor = None,
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
    _fused_qkvzba_conv_decode_kernel[
        (batch, triton.cdiv(qkv_dim, block))
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
        intermediate_conv,
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
        steps=intermediate_conv.shape[1] if intermediate_conv is not None else 1,
        block=block,
        num_warps=8,
        num_stages=2,
    )
    return mixed_qkv, z, b, a


# Adapted from SGLang/FLA's packed recurrent GDN decode kernel.  Keeping the
# packed QKV input avoids three materialization kernels in every GDN layer.
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
    stride_qkv,
    stride_a,
    stride_b,
    stride_state,
    stride_indices,
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
    value_head = batch_head % num_v_heads
    key_head = value_head // (num_v_heads // num_k_heads)

    key_offsets = tl.arange(0, block_k)
    value_offsets = value_tile * block_v + tl.arange(0, block_v)
    key_mask = key_offsets < head_k_dim
    value_mask = value_offsets < head_v_dim
    state_mask = value_mask[:, None] & key_mask[None, :]

    slot = tl.load(state_indices + batch * stride_indices).to(tl.int64)
    if slot <= 0:
        for step in range(steps):
            token = batch * steps + step
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
    for step in range(steps):
        token = batch * steps + step
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
            write_slot = tl.load(output_indices + token).to(tl.int64)
            state_ptr = states + write_slot * stride_state + state_offsets
        tl.store(state_ptr, state, mask=state_mask)
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
) -> torch.Tensor:
    num_tokens = mixed_qkv.shape[0]
    steps = ssm_output_indices.shape[1] if ssm_output_indices is not None else 1
    batch = num_tokens // steps
    num_v_heads, head_v_dim, head_k_dim = states.shape[-3:]
    output = torch.empty(
        (num_tokens, num_v_heads, head_v_dim),
        dtype=mixed_qkv.dtype,
        device=mixed_qkv.device,
    )
    block_k = triton.next_power_of_2(head_k_dim)
    block_v = min(triton.next_power_of_2(head_v_dim), 32)
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
def _packed_gdn_prefill_kernel(
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


def packed_gdn_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    states: torch.Tensor,
    state_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    num_k_heads, head_k_dim = q.shape[-2:]
    num_v_heads, head_v_dim = v.shape[-2:]
    output = torch.empty_like(v)
    block_k = triton.next_power_of_2(head_k_dim)
    block_v = min(triton.next_power_of_2(head_v_dim), 32)
    grid = (triton.cdiv(head_v_dim, block_v), state_indices.shape[0] * num_v_heads)
    _packed_gdn_prefill_kernel[grid](
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


class GatedDeltaNetBackend:
    """Fused GDN execution over model-owned recurrent state buffers."""

    def __init__(self, prefill_backend: str, decode_backend: str):
        supported = {"flashinfer", "triton"}
        if prefill_backend not in supported or decode_backend not in supported:
            raise ValueError("GDN backends must be one of: flashinfer, triton")
        self.prefill_backend = prefill_backend
        self.decode_backend = decode_backend
        (
            self._gdn_prefill,
            self._gdn_decode,
        ) = _load_gdn_kernels(prefill_backend, decode_backend)
        self._gdn_mtp = getattr(self._gdn_decode, "gated_delta_rule_mtp", None)
        if self._gdn_mtp is not None and (
            getattr(self._gdn_decode, "get_tile_v_mtp", None) is None
            or "ssm_state_indices" not in signature(self._gdn_mtp).parameters
        ):
            self._gdn_mtp = None

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
            conv_weight=conv_weight, conv_states=conv_states,
            query_start_loc=query_start_loc, state_indices=state_indices,
        )
        if ssm_state_indices is not None:
            state_indices = ssm_state_indices

        # The FlashInfer CP prefill API uses 128x128 FP32 states.
        if (
            self._gdn_prefill is None
            or head_k_dim != 128 or head_v_dim != 128
            or ssm_states.dtype != torch.float32
            or q.dtype not in (torch.float16, torch.bfloat16)
        ):
            return packed_gdn_prefill(
                q,
                k,
                v,
                g,
                beta,
                ssm_states,
                state_indices,
                query_start_loc,
            )

        # SFLLM has no prefix cache or chunked prefill, so EXTEND starts from
        # zero state.  Only the final state is materialized for later decode.
        output_state = ssm_states.new_empty(
            (state_indices.shape[0], *ssm_states.shape[1:]),
        )
        output, output_state = self._gdn_prefill(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=True,
            cu_seqlens=query_start_loc_i64,
            use_qk_l2norm_in_kernel=False,
            output_state=output_state,
            use_cp=True,
        )
        scatter_recurrent_state(output_state, ssm_states, state_indices)
        return output

    def _decode_kernel(self, qkv, states, dt_bias, state_indices, output_indices):
        """Select an available implementation using tensor metadata only."""
        if (
            self._gdn_decode is None or states.dtype != torch.float32
            or qkv.dtype not in (torch.float16, torch.bfloat16)
            or dt_bias.dtype not in (torch.bfloat16, torch.float32)
        ):
            return None
        num_v_heads, head_v_dim, head_k_dim = states.shape[-3:]
        if head_k_dim < 128 or head_v_dim < 128:
            return None
        if output_indices is None:
            if (
                getattr(self._gdn_decode, "run_pretranspose_decode", None) is not None
                and head_v_dim % self._gdn_decode.TILE_V == 0
            ):
                return self._gdn_decode.gated_delta_rule_decode_pretranspose
        elif self._gdn_mtp is not None and output_indices.shape[1] >= 2:
            tile_v = self._gdn_decode.get_tile_v_mtp(
                state_indices.shape[0], output_indices.shape[1],
                num_v_heads=num_v_heads, v_dim=head_v_dim,
            )
            if head_v_dim % tile_v == 0:
                return self._gdn_mtp
        return None

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
        intermediate_conv: torch.Tensor = None,
        ssm_state_indices: torch.Tensor = None,
        ssm_output_indices: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mixed_qkv, z, b, a = fused_qkvzba_conv_decode(
            projected_qkvz,
            projected_ba,
            conv_states,
            conv_weight,
            state_indices,
            num_v_heads,
            head_v_dim,
            intermediate_conv,
        )
        if ssm_state_indices is not None:
            state_indices = ssm_state_indices
        steps = ssm_output_indices.shape[1] if ssm_output_indices is not None else 1
        recurrent = self._decode_kernel(
            mixed_qkv, ssm_states, dt_bias, state_indices, ssm_output_indices
        )
        if recurrent is None:
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
            )
        else:
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
