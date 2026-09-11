"""Argmax over vocabulary logits, optionally fused with an addition."""

import torch
import triton
import triton.language as tl


@triton.jit
def _maximum_propagate_nan(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _argmax_partial_kernel(
    logits_ptr, addend_ptr, partial_values, partial_indices,
    vocab_size: tl.constexpr, logits_row_stride: tl.constexpr,
    addend_row_stride: tl.constexpr, HAS_ADDEND: tl.constexpr,
    PARTS: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    part = tl.program_id(1)
    col = part * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(logits_ptr + row * logits_row_stride + col, col < vocab_size, other=-float('inf'))
    value = x.to(tl.float32)
    if HAS_ADDEND:
        addend = tl.load(addend_ptr + row * addend_row_stride + col, col < vocab_size, other=0).to(tl.float32)
        value = (value + addend).to(x.dtype).to(tl.float32)
    best = tl.reduce(value, 0, _maximum_propagate_nan)
    same = tl.where(best != best, value != value, value == best)
    index = tl.min(tl.where(same & (col < vocab_size), col, 2147483647), 0)
    tl.store(partial_values + row * PARTS + part, best)
    tl.store(partial_indices + row * PARTS + part, index)


@triton.jit
def _argmax_merge_kernel(
    partial_values, partial_indices, output_ptr, output_stride: tl.constexpr,
    PARTS: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    part = tl.arange(0, BLOCK)
    value = tl.load(partial_values + row * PARTS + part, part < PARTS, other=-float('inf'))
    index = tl.load(partial_indices + row * PARTS + part, part < PARTS, other=2147483647)
    best = tl.reduce(value, 0, _maximum_propagate_nan)
    same = tl.where(best != best, value != value, value == best)
    chosen = tl.min(tl.where(same, index, 2147483647), 0)
    tl.store(output_ptr + row * output_stride, chosen)


def logits_argmax(x: torch.Tensor, addend: torch.Tensor = None,
                  out: torch.Tensor = None) -> torch.Tensor:
    """Compute ``x.argmax(-1)`` or ``(x + addend).argmax(-1)``, including ties and NaNs.

    The optional addition rounds to the logits dtype before reduction. Large
    vocabularies are split across CTAs; only partial maxima reach global memory.
    ``out`` may be strided and is also returned.
    """
    if (not x.is_cuda or x.ndim != 2 or x.stride(-1) != 1
            or x.shape[-1] < 4096 or x.shape[0] == 0
            or x.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            or (addend is not None and (addend.shape != x.shape or addend.dtype != x.dtype
                                        or addend.stride(-1) != 1))):
        result = (x if addend is None else x + addend).argmax(-1)
        return result if out is None else out.copy_(result)
    batch, vocab = x.shape
    block = 4096 if addend is None else 2048
    parts = triton.cdiv(vocab, block)
    partial_values = torch.empty((batch, parts), device=x.device, dtype=torch.float32)
    partial_indices = torch.empty((batch, parts), device=x.device, dtype=torch.int32)
    if out is None:
        out = torch.empty(batch, device=x.device, dtype=torch.int64)
    _argmax_partial_kernel[(batch, parts)](
        x, addend, partial_values, partial_indices, vocab, x.stride(0),
        0 if addend is None else addend.stride(0), addend is not None,
        parts, block, num_warps=4,
    )
    _argmax_merge_kernel[(batch,)](
        partial_values, partial_indices, out, out.stride(0), parts,
        triton.next_power_of_2(parts), num_warps=4,
    )
    return out
