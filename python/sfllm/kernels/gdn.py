"""Gated DeltaNet backends and fused preprocessing kernels for Qwen3.5."""

from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl
import sf_kernel

def _load_gdn_kernels(prefill_backend: str, decode_backend: str):
    try:
        from sgl_kernel import causal_conv1d_fwd
    except ImportError as exc:
        raise ImportError(
            "Qwen3.5 requires sgl-kernel with the causal-conv1d operators"
        ) from exc
    chunk_gated_delta_rule = None
    gated_delta_rule_decode_pretranspose = None
    if prefill_backend == "flashinfer":
        from flashinfer.gdn_prefill import chunk_gated_delta_rule
    if decode_backend == "flashinfer":
        from flashinfer.gdn_decode import gated_delta_rule_decode_pretranspose
    return (
        causal_conv1d_fwd,
        chunk_gated_delta_rule,
        gated_delta_rule_decode_pretranspose,
    )


@triton.jit(
    do_not_specialize=["num_tokens", "stride_dim"],
    do_not_specialize_on_alignment=["num_tokens", "stride_dim"],
)
def _split_l2norm_qkv_gates_kernel(
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
    stride_token,
    stride_dim,
    stride_a_token,
    stride_a_head,
    stride_b_token,
    stride_b_head,
    num_tokens,
    num_k_heads: tl.constexpr,
    num_v_heads: tl.constexpr,
    head_k_dim: tl.constexpr,
    head_v_dim: tl.constexpr,
    block_t: tl.constexpr,
    block_k: tl.constexpr,
    block_v: tl.constexpr,
):
    token = tl.program_id(0) * block_t + tl.arange(0, block_t)
    head = tl.program_id(1)
    token_mask = token < num_tokens

    kd = tl.arange(0, block_k)
    key_mask = (head < num_k_heads) & (kd[:, None] < head_k_dim)
    key_mask &= token_mask[None, :]
    q_offset = head * head_k_dim + kd[:, None]
    q_value = tl.load(
        mixed_qkv + q_offset * stride_dim + token[None, :] * stride_token,
        mask=key_mask,
        other=0.0,
    ).to(tl.float32)
    q_value *= tl.rsqrt(tl.sum(q_value * q_value, axis=0)[None, :] + 1e-6)
    tl.store(
        q + token[:, None] * (num_k_heads * head_k_dim) + q_offset.trans(),
        q_value.trans(),
        mask=key_mask.trans(),
    )

    k_offset = num_k_heads * head_k_dim + q_offset
    k_value = tl.load(
        mixed_qkv + k_offset * stride_dim + token[None, :] * stride_token,
        mask=key_mask,
        other=0.0,
    ).to(tl.float32)
    k_value *= tl.rsqrt(tl.sum(k_value * k_value, axis=0)[None, :] + 1e-6)
    tl.store(
        k + token[:, None] * (num_k_heads * head_k_dim) + q_offset.trans(),
        k_value.trans(),
        mask=key_mask.trans(),
    )

    vd = tl.arange(0, block_v)
    value_mask = (head < num_v_heads) & (vd[:, None] < head_v_dim)
    value_mask &= token_mask[None, :]
    v_head_offset = head * head_v_dim + vd[:, None]
    v_offset = 2 * num_k_heads * head_k_dim + v_head_offset
    v_value = tl.load(
        mixed_qkv + v_offset * stride_dim + token[None, :] * stride_token,
        mask=value_mask,
        other=0.0,
    )
    tl.store(
        v
        + token[:, None] * (num_v_heads * head_v_dim)
        + v_head_offset.trans(),
        v_value.trans(),
        mask=value_mask.trans(),
    )

    gate_mask = token_mask & (head < num_v_heads)
    a_value = tl.load(
        a + token * stride_a_token + head * stride_a_head,
        mask=gate_mask,
        other=0.0,
    ).to(tl.float32)
    b_value = tl.load(
        b + token * stride_b_token + head * stride_b_head,
        mask=gate_mask,
        other=0.0,
    ).to(tl.float32)
    head_mask = head < num_v_heads
    decay = tl.load(a_log + head, mask=head_mask, other=0.0).to(tl.float32)
    bias = tl.load(dt_bias + head, mask=head_mask, other=0.0).to(tl.float32)
    softplus_input = a_value + bias
    softplus = tl.where(
        softplus_input <= 20.0,
        tl.log(1.0 + tl.exp(softplus_input)),
        softplus_input,
    )
    gate_offset = token * num_v_heads + head
    tl.store(g + gate_offset, tl.exp(-tl.exp(decay) * softplus), mask=gate_mask)
    beta_value = tl.sigmoid(b_value).to(b.dtype.element_ty)
    tl.store(beta + gate_offset, beta_value, mask=gate_mask)


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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare contiguous normalized QKV and GDN gates in one launch."""
    num_tokens = mixed_qkv.shape[0]
    q = mixed_qkv.new_empty((num_tokens, num_k_heads, head_k_dim))
    k = torch.empty_like(q)
    v = mixed_qkv.new_empty((num_tokens, num_v_heads, head_v_dim))
    g = torch.empty_like(a, dtype=torch.float32)
    beta = torch.empty_like(b, dtype=torch.float32)
    block_t = 16
    grid = (triton.cdiv(num_tokens, block_t), max(num_k_heads, num_v_heads))
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
        mixed_qkv.stride(0),
        mixed_qkv.stride(1),
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        num_tokens,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        block_t=block_t,
        block_k=triton.next_power_of_2(head_k_dim),
        block_v=triton.next_power_of_2(head_v_dim),
        num_warps=2,
        num_stages=3,
    )
    return q, k, v, g, beta


@triton.jit
def _scatter_recurrent_state_kernel(
    source,
    destination,
    state_indices,
    stride_source,
    stride_destination,
    state_elements: tl.constexpr,
    num_state_slots: tl.constexpr,
    block: tl.constexpr,
):
    batch = tl.program_id(0)
    offsets = tl.program_id(1) * block + tl.arange(0, block)
    slot = tl.load(state_indices + batch).to(tl.int64)
    mask = (offsets < state_elements) & (slot >= 0) & (slot < num_state_slots)
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
) -> None:
    """Cast and scatter packed final states into model-local state rows."""
    state_elements = source[0].numel()
    block = 1024
    _scatter_recurrent_state_kernel[
        (source.shape[0], triton.cdiv(state_elements, block))
    ](
        source,
        destination,
        state_indices,
        source.stride(0),
        destination.stride(0),
        state_elements=state_elements,
        num_state_slots=destination.shape[0],
        block=block,
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
    block: tl.constexpr,
):
    batch = tl.program_id(0)
    offsets = tl.program_id(1) * block + tl.arange(0, block)
    qkv_mask = offsets < qkv_dim
    x = tl.load(
        projected_qkvz + batch * stride_qkvz + offsets,
        mask=qkv_mask,
        other=0.0,
    )

    slot = tl.load(state_indices + batch * stride_indices).to(tl.int64)
    valid_slot = (slot > 0) & (slot < num_state_slots)
    state = conv_states + slot * stride_state + offsets * stride_state_dim
    acc = tl.zeros((block,), dtype=tl.float32)
    for pos in tl.static_range(kernel_width - 1):
        state_value = tl.load(
            state + pos * stride_state_pos,
            mask=qkv_mask & valid_slot,
            other=0.0,
        )
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
        mixed_qkv + batch * qkv_dim + offsets,
        tl.where(valid_slot, conv_out, x),
        mask=qkv_mask,
    )

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
    tl.store(
        state + (kernel_width - 2) * stride_state_pos,
        x,
        mask=qkv_mask & valid_slot,
    )

    z_mask = offsets < value_dim
    z_value = tl.load(
        projected_qkvz + batch * stride_qkvz + qkv_dim + offsets,
        mask=z_mask,
        other=0.0,
    )
    tl.store(z + batch * value_dim + offsets, z_value, mask=z_mask)

    gate_mask = offsets < num_v_heads
    tl.store(
        b + batch * num_v_heads + offsets,
        tl.load(
            projected_ba + batch * stride_ba + offsets,
            mask=gate_mask,
            other=0.0,
        ),
        mask=gate_mask,
    )
    tl.store(
        a + batch * num_v_heads + offsets,
        tl.load(
            projected_ba + batch * stride_ba + num_v_heads + offsets,
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
    output_ptr = output + (batch * num_v_heads + value_head) * head_v_dim
    if slot <= 0:
        tl.store(output_ptr + value_offsets, 0.0, mask=value_mask)
        return

    state_ptr = states + slot * stride_state
    state_ptr += (
        value_head * head_v_dim * head_k_dim
        + value_offsets[:, None] * head_k_dim
        + key_offsets[None, :]
    )
    state = tl.load(state_ptr, mask=state_mask, other=0.0).to(tl.float32)

    qkv_ptr = mixed_qkv + batch * stride_qkv
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
    gate_a = tl.load(a + batch * stride_a + value_head).to(tl.float32)
    gate_b = tl.load(b + batch * stride_b + value_head).to(tl.float32)
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
    tl.store(output_ptr + value_offsets, result, mask=value_mask)
    tl.store(state_ptr, state, mask=state_mask)


def packed_gdn_decode(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    states: torch.Tensor,
    state_indices: torch.Tensor,
    num_k_heads: int,
) -> torch.Tensor:
    batch = mixed_qkv.shape[0]
    num_v_heads, head_v_dim, head_k_dim = states.shape[-3:]
    output = torch.empty(
        (batch, num_v_heads, head_v_dim),
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
            self._causal_conv1d_fwd,
            self._gdn_prefill,
            self._gdn_decode,
        ) = _load_gdn_kernels(prefill_backend, decode_backend)

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
    ) -> torch.Tensor:
        conv_input = mixed_qkv.transpose(0, 1).contiguous()
        self._causal_conv1d_fwd(
            conv_input,
            conv_weight,
            None,
            conv_states,
            query_start_loc,
            state_indices,
            None,
            True,
            -1,
        )
        mixed_qkv = conv_input.transpose(0, 1)

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
        )

        if self.prefill_backend == "triton":
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mixed_qkv, z, b, a = fused_qkvzba_conv_decode(
            projected_qkvz,
            projected_ba,
            conv_states,
            conv_weight,
            state_indices,
            num_v_heads,
            head_v_dim,
        )
        if self.decode_backend == "triton":
            core = packed_gdn_decode(
                mixed_qkv=mixed_qkv,
                a=a,
                b=b,
                a_log=a_log,
                dt_bias=dt_bias,
                states=ssm_states,
                state_indices=state_indices,
                num_k_heads=num_k_heads,
            )
        else:
            key_dim = num_k_heads * head_k_dim
            value_dim = num_v_heads * head_v_dim
            q, k, v = mixed_qkv.split((key_dim, key_dim, value_dim), dim=-1)
            core, _ = self._gdn_decode(
                q=q.view(-1, 1, num_k_heads, head_k_dim),
                k=k.view(-1, 1, num_k_heads, head_k_dim),
                v=v.view(-1, 1, num_v_heads, head_v_dim),
                output=v.new_empty((v.shape[0], 1, num_v_heads, head_v_dim)),
                state=None,
                A_log=a_log,
                a=a.view(-1, 1, num_v_heads),
                dt_bias=dt_bias,
                b=b.view(-1, 1, num_v_heads),
                use_qk_l2norm=True,
                initial_state=ssm_states,
                initial_state_indices=state_indices,
            )
            core = core.view(-1, num_v_heads, head_v_dim)
        return core, z
