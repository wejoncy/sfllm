# SPDX-License-Identifier: MIT
# MoE inference code derived from Triton v3.7.1 (f797708c0626e5f9840ca5b0a98790e2c7cb09ad).
# See LICENSE and SOURCE.json for attribution and extraction details.
"""Single-device top-k routing, expert scheduling and FP32 weighted reduction."""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from triton.language.target_info import is_hip


@dataclass
class RoutingMask:
    data: torch.Tensor
    shape: tuple
    shape_max: tuple

    @property
    def device(self):
        return self.data.device

    def stride(self, dim):
        return self.data.stride(dim)


@dataclass
class SparseMatrix:
    indx: torch.Tensor
    vals: torch.Tensor
    mask: RoutingMask

    def __post_init__(self):
        self.mask_metadata = make_bitmatrix_metadata(self.indx, self.mask)

@triton.jit
def get_topmask_and_fullmask(x):
    tl.static_assert(x.dtype.is_int_unsigned(), "floating-point value must be passed as bits")
    tm: tl.constexpr = 1 << (-1 + x.dtype.primitive_bitwidth)
    fm: tl.constexpr = (1 << x.dtype.primitive_bitwidth) - 1
    tm_arr = tl.full(x.shape, tm, dtype=x.dtype)
    fm_arr = tl.full(x.shape, fm, dtype=x.dtype)
    return tm_arr, fm_arr


@triton.jit
def fpval_to_key(x):
    tm, fm = get_topmask_and_fullmask(x)
    return x ^ tl.where((x & tm) != 0, fm, tm)


@triton.jit
def key_to_fpval(x):
    tm, fm = get_topmask_and_fullmask(x)
    return x ^ tl.where((x & tm) == 0, fm, tm)


@triton.jit
def indx_to_key(indx, N_EXPTS_PAD: tl.constexpr):
    return N_EXPTS_PAD - indx


@triton.jit
def key_to_indx(indx, N_EXPTS_PAD: tl.constexpr):
    return N_EXPTS_PAD - indx


@triton.jit
def streaming_topk(X, stride_xm, n_expts_tot, offs_m, mask_m, N_EXPTS_PAD: tl.constexpr, N_EXPTS_ACT: tl.constexpr,
                   BLOCK_N: tl.constexpr):
    x_nbits: tl.constexpr = X.dtype.element_ty.primitive_bitwidth
    x_utype: tl.constexpr = tl.dtype(f"uint{x_nbits}")
    if x_nbits < 16:
        # this ensures that we leave at least 16 bits for expert index
        # even if the input dtype is smaller than 16 bits:
        y_nbits: tl.constexpr = 32
    else:
        y_nbits: tl.constexpr = x_nbits * 2
    x_ultype: tl.constexpr = tl.dtype(f"uint{y_nbits}")
    x_dtype: tl.constexpr = X.dtype.element_ty

    # subtract 1 from loop iterations because we peel the first (masked) iteration:
    loop_iterations: tl.constexpr = N_EXPTS_PAD // BLOCK_N - 1
    offs_x_n = loop_iterations * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_x_n[None, :] < n_expts_tot

    # first iteration:
    X_ptrs = X + offs_m[:, None] * stride_xm + offs_x_n[None, :]
    x = tl.load(X_ptrs, mask=(mask_m & mask_n), other=float("-inf"))
    x = fpval_to_key(x.to(x_utype, bitcast=True))
    x = (x.to(x_ultype) << 16) | indx_to_key(offs_x_n, N_EXPTS_PAD)[None, :]
    acc = tl.topk(x, N_EXPTS_ACT, dim=1)

    # subsequent iterations:
    for _i in (tl.static_range if loop_iterations <= 4 else range)(loop_iterations):
        acc = tl.bitonic_merge(acc)  # ensure sorted ascending for the merge
        X_ptrs -= BLOCK_N
        offs_x_n -= BLOCK_N
        x = tl.load(X_ptrs, mask=mask_m, other=float("-inf"))
        x = fpval_to_key(x.to(x_utype, bitcast=True))
        x = (x.to(x_ultype) << 16) | indx_to_key(offs_x_n, N_EXPTS_PAD)[None, :]
        acc = tl.maximum(acc, tl.topk(x, N_EXPTS_ACT, dim=1))

    # sort packed (value_key, index_key) descending:
    # this keeps outputs ordered by gate value and uses smaller expert index for ties
    acc = tl.sort(acc, dim=1, descending=True)
    # 0000vvvvvvvviiii --> 0000iiii:
    y_indices_raw = (acc & 0xFFFF).to(tl.uint32)
    y_indices = key_to_indx(y_indices_raw, N_EXPTS_PAD)
    # 0000vvvvvvvviiii --> vvvvvvvv:
    y_values_raw = (acc >> 16).to(x_utype)
    y_values = key_to_fpval(y_values_raw).to(x_dtype, bitcast=True)

    return y_values, y_indices


@triton.jit
def _topk_forward(X, stride_xm,  # inputs
                  PeerYvs, PeerYis, stride_ym,  # topk values/indices
                  USE_PROVIDED_INDX: tl.constexpr, PeerBits, stride_rm: tl.constexpr,
                  stride_rn: tl.constexpr,  # bitmatrix
                  n_rows, n_expts_tot,  # shape
                  dst_offs_m, APPLY_SOFTMAX: tl.constexpr,  # constant
                  BLOCK_M: tl.constexpr, N_EXPTS_PAD: tl.constexpr, N_EXPTS_ACT: tl.constexpr, BLOCK_N: tl.constexpr):

    N_PEERS: tl.constexpr = len(PeerYvs)

    pid = tl.program_id(0)
    if isinstance(n_rows, tl.tensor) and n_rows.dtype.is_ptr():
        n_rows = tl.load(n_rows)

    if pid * BLOCK_M >= n_rows:
        # early exit:
        return

    tl.static_assert(BLOCK_N % 32 == 0)
    tl.static_assert(N_EXPTS_PAD % BLOCK_N == 0)
    x_dtype: tl.constexpr = X.dtype.element_ty

    # load logits
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_y_n = tl.arange(0, N_EXPTS_ACT)
    mask_m = offs_m[:, None] < n_rows
    if USE_PROVIDED_INDX:
        tl.static_assert(len(PeerYis) == 1)
        Yi_ptrs = PeerYis[0] + (dst_offs_m + offs_m[:, None]) * stride_ym + offs_y_n[None, :]
        y_indices = tl.load(Yi_ptrs, mask=mask_m)
        Xv_ptrs = X + offs_m[:, None] * stride_xm + y_indices
        y_values = tl.load(Xv_ptrs, mask=mask_m)
    else:
        y_values, y_indices = streaming_topk(X, stride_xm, n_expts_tot, offs_m, mask_m,  #
                                             N_EXPTS_PAD, N_EXPTS_ACT, BLOCK_N)

    # normalize selected values
    if APPLY_SOFTMAX:
        y_values = tl.softmax(y_values.to(tl.float32), dim=1, keep_dims=True).to(x_dtype)

    # write back
    for rank in tl.static_range(N_PEERS):
        Yv_ptrs = PeerYvs[rank] + (dst_offs_m + offs_m[:, None]) * stride_ym + offs_y_n[None, :]
        tl.store(Yv_ptrs, y_values, mask=mask_m)
    if not USE_PROVIDED_INDX:
        for rank in tl.static_range(N_PEERS):
            Yi_ptrs = PeerYis[rank] + (dst_offs_m + offs_m[:, None]) * stride_ym + offs_y_n[None, :]
            tl.store(Yi_ptrs, y_indices, mask=mask_m)

    # pack into bitmatrix
    y_div = y_indices // 32
    y_rem = y_indices % 32
    loop_iterations = N_EXPTS_PAD // BLOCK_N
    for i in range(loop_iterations):
        offs_r_n = tl.arange(0, BLOCK_N // 32) + i * (BLOCK_N // 32)
        y2 = tl.where(y_div[:, :, None] == offs_r_n[None, None, :], (1 << y_rem)[:, :, None], 0)
        r = tl.reduce_or(y2, axis=1)
        for rank in tl.static_range(N_PEERS):
            BitsPtrs = PeerBits[rank] + (dst_offs_m + offs_m[:, None]) * stride_rm + offs_r_n[None, :] * stride_rn
            tl.store(BitsPtrs, r, mask=mask_m)


def topk_forward(x, k, apply_softmax=True, n_rows=None):
    capacity, n_cols = x.shape
    n_rows = capacity if n_rows is None else n_rows
    assert len(x.shape) == 2 and n_cols < 32768
    # Preserve the measured kernel's tile sizes, output layouts and launch.
    block_m = block_n = 32
    n_cols_pad = triton.cdiv(n_cols, block_n) * block_n
    values = torch.empty((capacity, k), dtype=x.dtype, device=x.device)
    indices = torch.empty((capacity, k), dtype=torch.int16, device=x.device)
    bits = torch.empty(
        (n_cols_pad // 32, triton.cdiv(capacity, 32) * 32),
        dtype=torch.uint32, device=x.device,
    )
    bitmatrix_data = bits.T[:capacity]
    _topk_forward[(triton.cdiv(capacity, block_m),)](
        x, x.stride(0),
        (values,), (indices,), values.stride(0), False,
        (bits,), bitmatrix_data.stride(0), bitmatrix_data.stride(1),
        n_rows, n_cols, 0, BLOCK_M=block_m, BLOCK_N=block_n,
        APPLY_SOFTMAX=apply_softmax, N_EXPTS_PAD=n_cols_pad, N_EXPTS_ACT=k,
    )
    bitmatrix = RoutingMask(bitmatrix_data, (n_rows, n_cols), (capacity, n_cols))
    return values, indices, bitmatrix


def topk(x, k, apply_softmax=True, n_rows=None):
    values, indices, bitmatrix = topk_forward(x, k, apply_softmax, n_rows)
    return SparseMatrix(vals=values, indx=indices, mask=bitmatrix)


@triton.jit
def vpopc(x):
    """
    Vertical popcount
    Input  x : uint32[..., N]
    Output y : uint32[..., 32]
    semantics : y[..., i] = sum_j((x[..., j] >> i) & 1)
    credits: @apgoucher
    """

    tl.static_assert(x.dtype == tl.uint32, "x should consist of 32-bit unsigned integers")

    BLOCK_N: tl.constexpr = x.shape[-1]  # summation axis
    BATCHES: tl.constexpr = x.numel // BLOCK_N  # number of batches
    if BLOCK_N >= 8:
        sa1: tl.constexpr = 8
    else:
        sa1: tl.constexpr = BLOCK_N
    # create 8-way sums in 4-bit fields:
    y = tl.reshape(x, [BATCHES, BLOCK_N // sa1, sa1, 1])
    y = (y >> tl.arange(0, 4)[None, None, None, :]) & 0x11111111
    y = tl.sum(y, 2)  # [BATCHES, BLOCK_N // sa1, 4]
    if BLOCK_N >= 128:
        sa2: tl.constexpr = 16
    else:
        sa2: tl.constexpr = BLOCK_N // sa1
    # create 128-way sums in 8-bit fields:
    y = tl.reshape(y, [BATCHES, BLOCK_N // (sa1 * sa2), sa2, 1, 4])
    y = (y >> (4 * tl.arange(0, 2))[None, None, None, :, None]) & 0x0f0f0f0f
    y = tl.sum(y, 2)  # [BATCHES, BLOCK_N // (sa1 * sa2), 2, 4]
    sa3: tl.constexpr = BLOCK_N // (sa1 * sa2)
    # create N-way sums in 32-bit fields:
    y = tl.reshape(y, [BATCHES, 1, sa3, 8])
    y = (y >> (8 * tl.arange(0, 4))[None, :, None, None]) & 0x000000ff
    y = tl.sum(y, 2)  # [BATCHES, 4, 8]
    y = tl.reshape(y, x.shape[:-1] + [32])
    return y


@triton.jit
def _sum_bitmatrix_rows(B, shape_bm, stride_bm: tl.constexpr, stride_bn: tl.constexpr,  # input bitmatrix
                        Out, OutPartials, stride_pm: tl.constexpr, stride_pn, shape_pn,  # outputs
                        BLOCK_MM: tl.constexpr, BLOCK_M: tl.constexpr):
    tl.static_assert(BLOCK_MM % BLOCK_M == 0)
    TILE_SIZE: tl.constexpr = BLOCK_MM // BLOCK_M
    if isinstance(shape_bm, tl.tensor) and shape_bm.dtype.is_ptr():
        shape_bm = tl.load(shape_bm)
    # load input bits
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_bm = pid_m * BLOCK_MM + tl.arange(0, BLOCK_MM)
    bits = tl.load(B + pid_n * stride_bn + offs_bm * stride_bm, mask=offs_bm < shape_bm, other=0)
    bits = tl.reshape(bits, [TILE_SIZE, BLOCK_M])
    # partial row sum
    partial_row_sum = vpopc(bits)  # [TILE_SIZE, 32]
    # write-back partial row sum
    offs_pm = pid_m * TILE_SIZE + tl.arange(0, TILE_SIZE)
    offs_n = pid_n * 32 + tl.arange(0, 32)
    tl.store(OutPartials + offs_pm[:, None] * stride_pm + offs_n[None, :] * stride_pn, partial_row_sum)
    # update final row sum
    tl.atomic_add(Out + offs_n, tl.sum(partial_row_sum, 0), sem="relaxed")


def cdiv(x, y):
    return (x + y - 1) // y


def sum_bitmatrix_rows(x, partials_block_size=None):
    assert partials_block_size is not None
    PARTIALS_BLOCK_M = partials_block_size
    n_rows, n_cols = x.shape
    n_rows_max = x.shape_max[0]

    TILE_SIZE = max(1, 128 // PARTIALS_BLOCK_M)
    BLOCK_MM = PARTIALS_BLOCK_M * TILE_SIZE

    grid_m = cdiv(n_rows_max, BLOCK_MM)
    grid_n = cdiv(n_cols, 32)
    out = torch.zeros((cdiv(n_cols, 128) * 128, ), device=x.device, dtype=torch.int32)[:n_cols]
    out_partials = torch.empty((grid_n * 32, grid_m * TILE_SIZE), device=x.device, dtype=torch.int32)
    out_partials = torch.transpose(out_partials, 0, 1)
    # output tensors
    _sum_bitmatrix_rows[(grid_m, grid_n)](
        x.data, n_rows, x.stride(0), x.stride(1),  # input
        out,  # output [final reduction]
        out_partials, out_partials.stride(0), out_partials.stride(1),
        out_partials.shape[1],  # output [partial reductions]
        BLOCK_M=PARTIALS_BLOCK_M, BLOCK_MM=BLOCK_MM,  # constants
        num_warps=8)
    out_partials = out_partials[:cdiv(n_rows_max, PARTIALS_BLOCK_M), :]
    return out, out_partials


@dataclass
class BitmatrixMetadata:
    """
    Example:
    `bitmatrix` = [0 0 1 0 1 1 0
                   0 1 0 0 0 1 0
                   1 1 1 0 0 0 1
                   0 0 1 0 1 0 0]
    `col_sum` = [1 2 3 0 2 2 1]
    `col_sorted_indx` = cat([5], [3 6], [0 7], [], [9 1 10], [2 4], [8])
    `row_sorted_indx` = cat([3 6 8], [1 9], [0 2 4 10], [5 7])
    """
    # the number of entries equal to 1 in each column
    col_sum: torch.Tensor
    # indices of nonzero values numbered row-major, grouped by cols, concatenated
    col_sorted_indx: torch.Tensor
    # indices of nonzero values numbered col-major, grouped by rows, concatenated
    row_sorted_indx: torch.Tensor


@triton.jit
def _keyed_add(x, y):
    # we keep the key in the upper 16 bits of a uint32:
    key_mask: tl.constexpr = 0xffff0000

    kx = x & key_mask
    ky = y & key_mask
    z = tl.where(kx == ky, x + y - kx, y)
    return z


@triton.jit
def _bitmatrix_metadata_compute_stage2(ColSortedIndx, RowSortedIndx, NonzeroIndx, n_tokens, ColPartialSum, stride_pm,
                                       stride_pn, ColOffs, TOKS_PER_ROW: tl.constexpr, BLOCK_PER_TOK: tl.constexpr):
    BLOCK_SIZE: tl.constexpr = BLOCK_PER_TOK * TOKS_PER_ROW
    tl.static_assert(BLOCK_SIZE <= 32768)
    if isinstance(n_tokens, tl.tensor) and n_tokens.dtype.is_ptr():
        n_tokens = tl.load(n_tokens)
    nonzero_indx_size = n_tokens * TOKS_PER_ROW
    pid_m = tl.program_id(0)
    # load column indices
    offs_local = tl.arange(0, BLOCK_SIZE)
    offs_global = pid_m * BLOCK_SIZE + offs_local
    mask = offs_global < nonzero_indx_size
    col_indx = tl.load(NonzeroIndx + offs_global, mask=mask, other=-1).to(tl.uint32)
    # stable-sort by columns index
    kv_pairs = ((col_indx << 16) | offs_local).to(tl.uint32)
    kv_pairs = tl.sort(kv_pairs, 0)
    col_indx = kv_pairs >> 16
    offs_global = pid_m * BLOCK_SIZE + (kv_pairs & 0xffff)
    mask = col_indx != 0xffff
    # compute run lengths in column-sorted order:
    x = (kv_pairs & 0xffff0000 | 0x00000001)
    cols_and_inclusive_run_lengths = tl.associative_scan(x, 0, _keyed_add)
    exclusive_run_lengths = (cols_and_inclusive_run_lengths - 1) & 0xffff
    # compute output
    row_sorted_indx = tl.load(ColPartialSum + pid_m * stride_pm + col_indx * stride_pn, mask=mask)
    row_sorted_indx += tl.load(ColOffs + col_indx, mask=mask)
    row_sorted_indx += exclusive_run_lengths
    # write back output
    tl.store(RowSortedIndx + offs_global, row_sorted_indx, mask=mask)
    tl.store(ColSortedIndx + row_sorted_indx, offs_global, mask=mask)


@triton.jit
def _bitmatrix_metadata_compute_stage1(CombinedIndx, n_combined_indx, sentinel, BLOCK: tl.constexpr, ColSum, ColOffs,
                                       n_cols, PartialColSum, shape_pm, stride_pm, stride_pn, BLOCK_M: tl.constexpr,
                                       BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    # compute col_partial_sums
    if pid < n_cols:
        PartialColSum += pid * stride_pn
        curr_sum = 0
        for start in range(0, shape_pm, BLOCK_M):
            offs = start + tl.arange(0, BLOCK_M) * stride_pm
            partial_col_sum = tl.load(PartialColSum + offs, mask=offs < shape_pm)
            out = tl.cumsum(partial_col_sum, 0) - partial_col_sum + curr_sum
            curr_sum += tl.sum(partial_col_sum, 0)
            tl.store(PartialColSum + offs, out, mask=offs < shape_pm)
    # compute col_offs
    elif pid == n_cols:
        curr_sum = 0
        for start in range(0, n_cols, BLOCK_N):
            offs = start + tl.arange(0, BLOCK_N)
            col_sum = tl.load(ColSum + offs, mask=offs < n_cols)
            col_offs = tl.cumsum(col_sum, 0) - col_sum + curr_sum
            curr_sum += tl.sum(col_sum, 0)
            tl.store(ColOffs + offs, col_offs, mask=offs < n_cols)
    # memset `combined_indx` to `sentinel`
    else:
        offs = (pid - n_cols - 1) * BLOCK + tl.arange(0, BLOCK)
        tl.store(CombinedIndx + offs, sentinel, mask=offs < n_combined_indx)


def make_bitmatrix_metadata(nonzero_indx, bitmatrix):
    assert nonzero_indx.ndim == 2
    PARTIAL_BLOCK_M = 32
    col_sum, col_partial_sum = sum_bitmatrix_rows(bitmatrix, partials_block_size=PARTIAL_BLOCK_M)
    # allocate memory
    device = bitmatrix.device
    n_indx = nonzero_indx.numel()
    n_cols = bitmatrix.shape[1]
    col_offs = torch.empty(n_cols, dtype=torch.int32, device=device)
    combined_indx = torch.empty(n_indx * 2, dtype=torch.int32, device=device)
    col_sorted_indx = combined_indx[:n_indx]
    row_sorted_indx = combined_indx[n_indx:]
    # this kernel:
    # - initializes `{row,col}_sorted_indx` to `sentinel`
    # - computes col_offs; necessary for computing `{row,col}_sorted_indx`
    # - computes col_partial_sums; necessary for computing `{row,col}_sorted_indx`
    MEMSET_BLOCK = 1024
    memset_grid = (cdiv(n_indx * 2, MEMSET_BLOCK) + n_cols + 1, )
    _bitmatrix_metadata_compute_stage1[memset_grid](
        combined_indx, n_indx * 2, -1, MEMSET_BLOCK, col_sum,  #
        col_offs, col_sum.shape[0], col_partial_sum,  # inputs
        col_partial_sum.shape[0], col_partial_sum.stride(0), col_partial_sum.stride(1),  # outputs
        BLOCK_M=512, BLOCK_N=512,  # tunable parameters
    )
    # this kernel computes valid entries of `{row,col}_sorted_indx`
    # using `col_offs` and `col_partial_sums`
    n_indx = nonzero_indx.numel()
    toks_per_row = nonzero_indx.shape[-1]
    compute_grid = (cdiv(bitmatrix.shape_max[0], PARTIAL_BLOCK_M), )
    _bitmatrix_metadata_compute_stage2[compute_grid](
        col_sorted_indx, row_sorted_indx,  # outputs
        nonzero_indx, bitmatrix.shape[0], col_partial_sum, col_partial_sum.stride(0),
        col_partial_sum.stride(1),  # inputs
        col_offs,  #
        TOKS_PER_ROW=toks_per_row, BLOCK_PER_TOK=PARTIAL_BLOCK_M,  #
    )
    return BitmatrixMetadata(
        col_sum=col_sum,
        col_sorted_indx=col_sorted_indx,
        row_sorted_indx=row_sorted_indx,
    )


@dataclass
class RaggedTensorMetadata:
    """
    Example:
    `slice_sizes`= [15 17 0 127]
    `slice_offs`= [0 15 32 32 332]
    `block_offs_data` = {
        16: [0 1 3 3 11]
        32: [0 1 2 2 6]
        64: [0 1 2 2 4]
        128: [0 1 2 2 3]
    }
    `block_schedule_data` = {
        16:  [(0, 0) (0, 1) (0, 3) (1, 3) (2, 3) ... (7, 3) -1 ... -1]
        32:  [(0, 0) (0, 1) (0, 3) (1, 3) (2, 3) (3, 3) -1 ...     -1]
        64:  [(0, 0) (0, 1) (0, 3) (1, 3) (2, 3) -1 ...            -1]
        128: [(0, 0) (0, 1) (0, 3) (1, 3) -1 ...                   -1]
    }
    """
    # slice_sizes[i] is the number of elements in slice i along the ragged dimension
    slice_sizes: torch.Tensor
    # slice_offs = [0] + cumsum(slice_sizes)
    # i.e., slice_offs[i] is the offset of the first element in slice `i`
    slice_offs: torch.Tensor
    # block_offs_data[k] = [0] + cumsum(ceil_div(slice_sizes, 16 * k))
    # i.e., `block_offs_data[k][i]` is the offset of the first block of
    # `16*k`` token for batch `i` in a `bath_sizes`-shaped ragged tensor
    block_offs_data: torch.Tensor
    # let `num_blocks[k] = block_offs_data[k, 1:] - block_offs_data[k, :-1]
    # block_schedule_data[k] = cat(*[[(batch, blk) for blk in range(blks)] for batch, blks in enumerate(num_blocks)])
    # i.e., if the schedule of batch `i` is [(i, 0), (i, 1), ..., (i, num_blocks[k][i] - 1)]
    # then `block_schedule_data[k]` is the concatenation of the schedules for all batches
    # NOTE 1: `block_schedule_data[k][j]` is a packed 32-bit integer
    # NOTE 2: because the size of `block_schedule_data[k]` is data-dependent, we pad it with -1s
    # up to an user-provided upper bound
    block_schedule_data: torch.Tensor
    # expected slice size (for heuristics)
    expected_slice_size: int | None = None
    # divisibility hint for values in `slice_sizes`
    slice_sizes_divisibility: int = None

    def __post_init__(self):
        assert self.block_offs_data.shape[0] == len(RaggedTensorMetadata.block_sizes())
        assert self.block_schedule_data.shape[0] == len(RaggedTensorMetadata.block_sizes())
        assert self.block_offs_data.dtype == torch.int32
        assert self.block_schedule_data.dtype == torch.int32
        if self.slice_sizes is not None:
            assert self.slice_sizes.dtype == torch.int32
        if self.slice_offs is not None:
            assert self.slice_offs.dtype == torch.int32

    @property
    def n_slices(self):
        return self.slice_sizes.shape[0]

    def block_offs(self, block_size):
        return self.block_offs_data[RaggedTensorMetadata.block_sizes().index(block_size)]

    def block_schedule(self, block_size):
        return self.block_schedule_data[RaggedTensorMetadata.block_sizes().index(block_size)]

    @staticmethod
    def n_blocks(n_slices, n_total_rows, block_size):
        if n_total_rows <= n_slices:
            return n_total_rows
        return n_slices - 1 - ((n_slices - n_total_rows - 1) // block_size)

    @staticmethod
    def max_n_blocks(n_slices, n_total_rows):
        return RaggedTensorMetadata.n_blocks(n_slices, n_total_rows, min(RaggedTensorMetadata.block_sizes()))

    @staticmethod
    def block_sizes_log2():
        return range(4, 9) if is_hip() else range(4, 8)

    @staticmethod
    def block_sizes():
        return [2**x for x in RaggedTensorMetadata.block_sizes_log2()]


def ragged_metadata_fields(metadata, block_size):
    return (metadata.slice_sizes, metadata.slice_offs, metadata.block_offs(block_size),
            metadata.block_schedule(block_size), metadata.expected_slice_size, metadata.slice_sizes_divisibility or 1)


def exact_div(x, y):
    assert x % y == 0
    return x // y


def empty_aligned(shape, dtype, device, pad_size):
    cdiv = lambda x, y: (x + y - 1) // y
    pad = lambda x: cdiv(x, pad_size) * pad_size
    ret = torch.empty((*shape[:-1], pad(shape[-1])), dtype=dtype, device=device)
    ret_slices = (*[slice(None)] * (len(shape) - 1), slice(0, shape[-1]))
    return ret[ret_slices], ret.numel()


@triton.jit
def _cdiv_pow2(n, log2_k):
    # ceil_div(n, 2**log2_k)
    return (n + ((1 << log2_k) - 1)) >> log2_k


@triton.jit
def _ragged_tensor_metadata_memset(SliceSizes, n_slices, BlockOffs, slice_offs_stride_m, BlockSchedule,
                                   first_block_size_log2, SIZES: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    if pid <= SIZES:
        BlockOffs += pid * slice_offs_stride_m
        BlockOffsPtrs = BlockOffs + tl.arange(0, BLOCK)
        block_size_log2 = tl.where(pid == 0, 0, pid + first_block_size_log2 - 1)
        # total number of blocks in slice processed as the loop iterates
        n_blocks_tot = tl.zeros([BLOCK], dtype=BlockOffs.dtype.element_ty)
        for i in range(0, n_slices + 1, BLOCK):
            # load slice sizes
            offs = tl.arange(0, BLOCK) + i
            mask = offs < n_slices
            slice_sizes = tl.load(SliceSizes + offs, mask=mask, other=0)
            # number of blocks in the slices loaded
            n_blocks = _cdiv_pow2(slice_sizes, block_size_log2)
            # start index of the blocks for the slices loaded
            block_starts = tl.cumsum(n_blocks, 0) + n_blocks_tot
            n_blocks_tot += tl.sum(n_blocks, 0)
            tl.store(BlockOffsPtrs, block_starts - n_blocks)
            BlockOffsPtrs += BLOCK
    else:
        # initialize block schedule to -1
        pid -= (SIZES + 1)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        tl.store(BlockSchedule + offs, 0xffffffff)


@triton.jit
def _ragged_tensor_metadata_compute(SliceSizes,  #
                                    BlockOffs, block_offs_stride_m,  #
                                    BlockSchedule, block_schedule_stride_m,  #
                                    first_block_size_log2,  #
                                    SIZES: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    slice_id = pid // SIZES
    block_size_id = pid % SIZES
    # offset pointers
    BlockOffs += block_size_id * block_offs_stride_m
    BlockSchedule += block_size_id * block_schedule_stride_m
    # load slice sizes
    slice_sizes = tl.load(SliceSizes + slice_id)
    # number of blocks in the slices loaded
    block_size_log2 = first_block_size_log2 + block_size_id
    n_blocks = _cdiv_pow2(slice_sizes, block_size_log2)
    # compute block schedule
    block_off = tl.load(BlockOffs + slice_id)
    BlockSchedule += block_off
    for block_off in range(0, n_blocks, BLOCK):
        block_offs = block_off + tl.arange(0, BLOCK)
        data = (block_offs << 16) + slice_id
        tl.store(BlockSchedule + block_offs, data, mask=block_offs < n_blocks)


def make_ragged_tensor_metadata(slice_sizes, n_total_rows):
    assert slice_sizes.ndim == 1
    n_slices = slice_sizes.shape[0]
    block_sizes_log2 = RaggedTensorMetadata.block_sizes_log2()
    block_size_num = len(block_sizes_log2)
    MEMSET_BLOCK = 512
    dtype = torch.int32
    device = slice_sizes.device
    max_n_blocks = RaggedTensorMetadata.max_n_blocks(n_slices, n_total_rows)
    slice_offs_combined, _ = empty_aligned((block_size_num + 1, n_slices + 1), dtype, device, MEMSET_BLOCK)
    block_schedule_data, n_memset_elts = empty_aligned((block_size_num, max_n_blocks), dtype, device, MEMSET_BLOCK)
    slice_offs, block_offs_data = slice_offs_combined[0], slice_offs_combined[1:]
    n_memset_blocks = exact_div(n_memset_elts, MEMSET_BLOCK)

    _ragged_tensor_metadata_memset[(slice_offs_combined.shape[0] + n_memset_blocks, )](
        slice_sizes, n_slices,  #
        slice_offs_combined, slice_offs_combined.stride(0),  #
        block_schedule_data,  #
        block_sizes_log2[0], SIZES=len(block_sizes_log2), BLOCK=MEMSET_BLOCK,  # optimization parameters
        num_warps=4)

    _ragged_tensor_metadata_compute[(block_size_num * n_slices, )](
        slice_sizes, block_offs_data, block_offs_data.stride(0), block_schedule_data,
        block_schedule_data.stride(0),  # outputs
        block_sizes_log2[0], SIZES=len(block_sizes_log2), BLOCK=512,  # optimization parameters
        num_warps=4)

    return RaggedTensorMetadata(slice_sizes, slice_offs, block_offs_data, block_schedule_data)


@triton.jit
def _reduce_forward(
    X, stride_xr: tl.int64, stride_x0: tl.int64, stride_x1, Y, stride_y0: tl.int64, stride_y1,
    Scale, stride_sr, stride_s0, stride_s1, K: tl.constexpr, S0, X_S1, Y_S1,
    IS_SCALE_NONE: tl.constexpr, SCALE_BROADCAST_R: tl.constexpr, SCALE_BROADCAST_S0: tl.constexpr,
    SCALE_BROADCAST_S1: tl.constexpr, BLOCK_S0: tl.constexpr, BLOCK_X_S1: tl.constexpr,
    BLOCK_Y_S1: tl.constexpr,
):
    pid_s0 = tl.program_id(0)
    pid_s1 = tl.program_id(1)
    tl.static_assert(BLOCK_X_S1 % 32 == 0)
    offs_s0 = pid_s0 * BLOCK_S0 + tl.arange(0, BLOCK_S0)
    offs_x_s1 = pid_s1 * BLOCK_X_S1 + tl.arange(0, BLOCK_X_S1)
    valid_s0 = offs_s0 < S0
    valid_x_s1 = offs_x_s1 < X_S1
    y = tl.zeros((BLOCK_S0, BLOCK_X_S1), dtype=tl.float32)
    x_flex_scale = 1.0
    for k in (tl.static_range if K <= 8 else tl.range)(0, K):
        x_ptrs = X + k * stride_xr + offs_s0[:, None] * stride_x0 + offs_x_s1[None, :] * stride_x1
        mask = valid_s0[:, None] & valid_x_s1[None, :]
        x = tl.load(x_ptrs, mask=mask, other=0.0)
        x = x.to(tl.float32)
        x = x * x_flex_scale
        if not IS_SCALE_NONE:
            k_term_s = 0 if SCALE_BROADCAST_R else k * stride_sr
            s0_term_s = 0 if SCALE_BROADCAST_S0 else offs_s0[:, None] * stride_s0
            s1_term_s = 0 if SCALE_BROADCAST_S1 else offs_x_s1[None, :] * stride_s1
            s_ptrs = Scale + k_term_s + s0_term_s + s1_term_s
            s = tl.load(s_ptrs, mask=mask, other=1)
            x = tl.fma(x, s, 0.0)
        y += x
    offs_y_s1 = pid_s1 * BLOCK_Y_S1 + tl.arange(0, BLOCK_Y_S1)
    valid_y_s1 = offs_y_s1 < Y_S1
    y = y * 1.0
    y_ptrs = Y + offs_s0[:, None] * stride_y0 + offs_y_s1[None, :] * stride_y1
    tl.store(y_ptrs, y, mask=valid_s0[:, None] & valid_y_s1[None, :])


def reduce_forward(x, dim, scale=None, y=None, y_dtype=None):
    """Preserve the original sequential FP32 accumulation and output rounding."""
    if x.ndim != 3 or dim not in (0, 1):
        raise ValueError("Expected expert or split-K reduction of a 3D tensor")
    nonred = [i for i in range(3) if i != dim]
    s0, s1 = (x.shape[i] for i in nonred)
    if y is None:
        y = torch.empty((s0, s1), device=x.device, dtype=y_dtype or x.dtype)
    sr, s0_stride, s1_stride = (0, 0, 0) if scale is None else tuple(scale.stride(i) for i in (dim, *nonred))
    _reduce_forward[(triton.cdiv(s0, 32), triton.cdiv(s1, 128))](
        X=x, stride_xr=x.stride(dim), stride_x0=x.stride(nonred[0]), stride_x1=x.stride(nonred[1]),
        Y=y, stride_y0=y.stride(0), stride_y1=y.stride(1),
        Scale=scale, stride_sr=sr, stride_s0=s0_stride, stride_s1=s1_stride,
        K=x.shape[dim], S0=s0, X_S1=s1, Y_S1=s1,
        IS_SCALE_NONE=scale is None, SCALE_BROADCAST_R=sr == 0,
        SCALE_BROADCAST_S0=s0_stride == 0, SCALE_BROADCAST_S1=s1_stride == 0,
        BLOCK_S0=32, BLOCK_X_S1=128, BLOCK_Y_S1=128, num_warps=4,
    )
    return y, None
