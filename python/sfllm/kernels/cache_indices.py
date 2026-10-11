"""Resolve a retained SWA window once for all layers in its resource group."""

import torch
import triton
import triton.language as tl


@triton.jit
def _window_indices(indices, indptr, locations, qo_indptr, requests,
                    resolved, resolved_indptr, resolved_locations,
                    B: tl.constexpr, ROWS: tl.constexpr, WINDOW: tl.constexpr,
                    DECODE: tl.constexpr, BLOCK: tl.constexpr,
                    BASE: tl.constexpr, STRIDE: tl.constexpr, prefill_mapping):
    row = tl.program_id(0)
    request = tl.load(requests + row)
    rows = tl.arange(0, ROWS)
    ends = tl.load(indptr + rows + 1, rows < B, other=0)
    starts = tl.load(indptr + rows, rows < B, other=0)
    counts = tl.minimum(ends - starts, WINDOW + (1 if DECODE else 0))
    base = tl.sum(tl.where(rows < row, counts, 0)).to(tl.int64)
    end = tl.load(indptr + row + 1).to(tl.int64)
    length = end - tl.load(indptr + row)
    count = tl.minimum(length, WINDOW + (1 if DECODE else 0))
    tl.store(resolved_indptr + row, base)
    if row == B - 1:
        tl.store(resolved_indptr + B, base + count)
    for off in range(0, count, BLOCK):
        x = off + tl.arange(0, BLOCK)
        ids = tl.load(indices + end - count + x, x < count, other=0)
        physical = ring_location(ids, length - count + x, request, BASE, STRIDE, prefill_mapping)
        tl.store(resolved + base + x, physical, x < count)
    if DECODE:
        start, end = row, row + 1
    else:
        start = tl.load(qo_indptr + row)
        end = tl.load(qo_indptr + row + 1)
    for off in range(start, end, BLOCK):
        x = off + tl.arange(0, BLOCK)
        ids = tl.load(locations + x, x < end, other=0)
        physical = ring_location(ids, length + x - start - (1 if DECODE else 0), request, BASE, STRIDE, prefill_mapping)
        tl.store(resolved_locations + x, physical, x < end)


def resolve_window(batch, ring, indices):
    indptr = torch.empty_like(batch.kv_indptr)
    locations = torch.empty_like(batch.out_cache_loc)
    rows = batch.kv_indptr.numel() - 1
    _window_indices[(rows,)](
        batch.kv_indices, batch.kv_indptr, batch.out_cache_loc, batch.qo_indptr,
        ring.request_indices, indices, indptr, locations, B=rows, ROWS=triton.next_power_of_2(rows),
        WINDOW=ring.history, DECODE=batch.is_decode(), BLOCK=256, BASE=ring.base, STRIDE=ring.stride,
        prefill_mapping=ring.prefill_mapping if batch.forward_mode.name == "EXTEND" else None,
    )
    batch.kv_indices, batch.kv_indptr, batch.out_cache_loc = indices, indptr, locations


@triton.jit
def ring_location(ids, positions, request, BASE: tl.constexpr, STRIDE: tl.constexpr,
                  prefill_mapping=None):
    physical = BASE + request * STRIDE + positions % STRIDE
    # Sink and fixed proposal slots precede every request's working ring.
    physical = tl.where(ids < BASE, tl.maximum(ids, 0), physical)
    if prefill_mapping is not None:
        temporary = tl.load(prefill_mapping + tl.maximum(ids, 0))
        physical = tl.where(temporary > 0, temporary, physical)
    return physical


@triton.jit
def _ring_write_locations(locations, positions, indptr, requests, output,
                          T: tl.constexpr, B: tl.constexpr, BASE: tl.constexpr,
                          STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    start = tl.load(indptr + row).to(tl.int64)
    end = tl.load(indptr + row + 1).to(tl.int64)
    # Padding already has fixed scratch IDs; the last CTA copies it as well.
    if row == B - 1:
        end = tl.full((), T, tl.int64)
    request = tl.load(requests + row)
    for off in range(start, end, BLOCK):
        x = off + tl.arange(0, BLOCK)
        ids = tl.load(locations + x, x < end, other=0)
        pos = tl.load(positions + x, x < end, other=0)
        physical = ring_location(ids, pos, request, BASE, STRIDE)
        tl.store(output + x, physical, x < end)


def ring_write_locations(locations, positions, indptr, requests, base, stride):
    output = torch.empty_like(locations)
    _ring_write_locations[(indptr.numel() - 1,)](
        locations, positions, indptr, requests, output,
        T=locations.numel(), B=indptr.numel() - 1, BASE=base, STRIDE=stride,
        BLOCK=32, num_warps=1,
    )
    return output


@triton.jit
def _copy_cache_slots(source_key, source_value, key, value, other_key, other_value,
                      sources, destinations, split,
                      SKL: tl.constexpr, SVL: tl.constexpr,
                      KL: tl.constexpr, VL: tl.constexpr,
                      OKL: tl.constexpr, OVL: tl.constexpr,
                      KE: tl.constexpr, VE: tl.constexpr,
                      SECONDARY: tl.constexpr, BLOCK: tl.constexpr):
    row, layer, block = tl.program_id(0), tl.program_id(1).to(tl.int64), tl.program_id(2)
    source = tl.load(sources + row).to(tl.int64)
    destination = tl.load(destinations + row).to(tl.int64)
    x = block * BLOCK + tl.arange(0, BLOCK)
    k = tl.load(source_key + layer * SKL + source * KE + x, x < KE, other=0)
    v = tl.load(source_value + layer * SVL + source * VE + x, x < VE, other=0)
    kp, vp = key + layer * KL, value + layer * VL
    if SECONDARY:
        kp = tl.where(row < split, kp, other_key + layer * OKL)
        vp = tl.where(row < split, vp, other_value + layer * OVL)
    tl.store(kp + destination * KE + x, k, x < KE)
    tl.store(vp + destination * VE + x, v, x < VE)


def copy_cache_slots(source, destination, sources, destinations, secondary=None):
    if not sources.numel():
        return
    sk, sv = source
    key, value = destination
    ok, ov, split = (key, value, sources.numel()) if secondary is None else secondary
    ke, ve = sk.stride(1), sv.stride(1)
    _copy_cache_slots[(sources.numel(), sk.shape[0], triton.cdiv(max(ke, ve), 512))](
        sk, sv, key, value, ok, ov, sources, destinations, split,
        SKL=sk.stride(0), SVL=sv.stride(0), KL=key.stride(0), VL=value.stride(0),
        OKL=ok.stride(0), OVL=ov.stride(0), KE=ke, VE=ve,
        SECONDARY=secondary is not None, BLOCK=512, num_warps=4,
    )
