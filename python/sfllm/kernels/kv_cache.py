"""Indexed norm/RoPE/cache writes to two independent KV destinations."""

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def _shuffle_xor(value, offset):
    return gl.inline_asm_elementwise(
        "shfl.sync.bfly.b32 $0, $1, $2, 31, -1;", "=r,r,r",
        (value.to(gl.int32, bitcast=True), offset), dtype=gl.int32,
        is_pure=True, pack=1,
    ).to(gl.float32, bitcast=True)


@gluon.jit
def _indexed_kv_norm_rope(raw, weight, rope, positions, rows, locations,
                         key, value, other_key, other_value,
                         count, split, epsilon,
                         RT: gl.constexpr, RL: gl.constexpr, RH: gl.constexpr,
                         WL: gl.constexpr, KL: gl.constexpr, VL: gl.constexpr,
                         OKL: gl.constexpr, OVL: gl.constexpr,
                         H: gl.constexpr, D: gl.constexpr, RD: gl.constexpr,
                         INDEXED: gl.constexpr, INTERLEAVE: gl.constexpr,
                         VEC: gl.constexpr, BT: gl.constexpr):
    # Match the native writer: a lane owns VEC adjacent elements, a subgroup one head.
    LANES: gl.constexpr = D // VEC
    layout: gl.constexpr = gl.BlockedLayout([1, 1, VEC], [32 // LANES, LANES, 1], [4, 1, 1], [2, 1, 0])
    reduced: gl.constexpr = gl.SliceLayout(2, layout)
    row = gl.program_id(0) * BT + gl.arange(0, BT, layout=gl.SliceLayout(1, reduced))
    lane = gl.arange(0, LANES, layout=gl.SliceLayout(1, gl.SliceLayout(0, layout)))
    element = gl.arange(0, VEC, layout=gl.SliceLayout(0, gl.SliceLayout(0, layout)))
    head, layer = gl.program_id(1), gl.program_id(2).to(gl.int64)
    source = row.to(gl.int64)
    if INDEXED:
        source = gl.load(rows + row, row < count, other=0).to(gl.int64)
    d = lane[:, None] * VEC + element[None, :]
    base = source[:, None, None] * RT + layer * RL + head * RH + d[None, :, :]
    k = gl.load(raw + base, row[:, None, None] < count, other=0).to(gl.float32)
    v = gl.load(raw + base + H * RH, row[:, None, None] < count, other=0)
    w = gl.load(weight + layer * WL + d).to(gl.float32)
    variance = gl.full((BT, LANES), 0, gl.float32, layout=reduced)
    # Preserve the native FP32 accumulation order, including the order across lanes.
    for i in gl.static_range(VEC):
        index = gl.full((BT, LANES, 1), i, gl.int32, layout=layout)
        x = gl.convert_layout(gl.gather(k, index, 2).reshape(BT, LANES), reduced)
        variance = gl.fma(x, x, variance)
    for shift in gl.static_range(LANES.bit_length() - 2, -1, -1):
        variance += _shuffle_xor(variance, 1 << shift)
    inv = gl.rsqrt(variance / D + epsilon)
    k = (k * inv[:, :, None] * w[None, :, :]).to(raw.dtype.element_ty).to(gl.float32)
    if INTERLEAVE:
        index = gl.broadcast_to((element ^ 1)[None, :][None, :, :], (BT, LANES, VEC))
        paired = gl.gather(k, index, 2)
        rotary_index = d // 2
        negative = d % 2 == 0
    else:
        paired = _shuffle_xor(k, RD // (2 * VEC))
        rotary_index = d % (RD // 2)
        negative = d < RD // 2
    pos = gl.load(positions + source, row < count, other=0).to(gl.int64)
    mask = (row[:, None, None] < count) & (d[None, :, :] < RD)
    cos = gl.load(rope + pos[:, None, None] * RD + rotary_index[None, :, :], mask, other=1)
    sin = gl.load(rope + pos[:, None, None] * RD + RD // 2 + rotary_index[None, :, :], mask, other=0)
    rotated = gl.fma(k, cos, gl.where(negative[None, :, :], -paired, paired) * sin)
    k = gl.where(d[None, :, :] < RD, rotated, k)
    dst = gl.load(locations + row, row < count, other=0).to(gl.int64)
    kp = gl.where(row < split, key + layer * KL, other_key + layer * OKL)
    vp = gl.where(row < split, value + layer * VL, other_value + layer * OVL)
    offset = dst[:, None, None] * H * D + head * D + d[None, :, :]
    gl.store(kp[:, None, None] + offset, k, row[:, None, None] < count)
    gl.store(vp[:, None, None] + offset, v, row[:, None, None] < count)


def indexed_kv_norm_rope(raw, weight, rope, positions, interleave,
                         key, value, locations, rows, secondary, epsilon):
    if not locations.numel():
        return
    if isinstance(rows, slice):
        raw, positions, rows = raw[rows], positions[rows], None
    other_key, other_value, split = secondary
    dim = key.shape[3]
    vec = max(16 // raw.element_size(), dim // 32)
    tile_tokens = 128 // (dim // vec)
    _indexed_kv_norm_rope[(triton.cdiv(locations.numel(), tile_tokens), key.shape[2], key.shape[0])](
        raw, weight, rope, positions, rows, locations, key, value, other_key, other_value,
        locations.numel(), split, epsilon,
        RT=raw.stride(0), RL=raw.stride(1), RH=raw.stride(2), WL=weight.stride(0),
        KL=key.stride(0), VL=value.stride(0), OKL=other_key.stride(0), OVL=other_value.stride(0),
        H=key.shape[2], D=dim, RD=rope.shape[1], INDEXED=rows is not None,
        INTERLEAVE=interleave, VEC=vec, BT=tile_tokens, num_warps=4,
    )


def write_context_kv(raw_kv, norm_weights, rope, positions, interleave,
                     kv_buffers, cache_locs, epsilon):
    writer = torch.ops.sfkernels.kv_norm_rope_and_cache
    # Contiguous layer runs share one launch. Token slices are views;
    # disjoint retained windows are indexed directly by the writer.
    for layers, key, value, locations, rows, secondary in kv_buffers:
        projected = raw_kv[:, layers]
        locs, pos = cache_locs if locations is None else locations, positions
        if secondary is not None or (locations is not None and isinstance(rows, torch.Tensor)):
            indexed_kv_norm_rope(projected, norm_weights[layers], rope, pos, interleave,
                                 key, value, locs, rows,
                                 secondary if secondary is not None else (key, value, locs.numel()), epsilon)
            continue
        if rows is not None:
            projected, pos = projected[rows], pos[rows]
            if locations is None:
                locs = locs[rows]
        if not projected.shape[0]:
            continue
        writer(projected, norm_weights[layers], rope, pos,
               interleave, key, value, locs, epsilon)
