# SPDX-License-Identifier: MIT
# MoE inference code derived from Triton v3.7.1 (f797708c0626e5f9840ca5b0a98790e2c7cb09ad).
# See LICENSE and SOURCE.json for attribution and extraction details.
"""BF16/FP16 expert grouped GEMM, retaining Triton's original launch heuristics."""

import collections
from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from triton.language.target_info import cuda_capability_geq, is_hip
from triton.tools.ragged_tma import create_ragged_descriptor, load_ragged, store_ragged
from triton.tools.tensor_descriptor import TensorDescriptor

from . import reduce_forward

@triton.jit
def xcd_swizzle(pid, domain_size, XCD_SWIZZLE: tl.constexpr):
    """
    Swizzle the program id based on integer XCD_SWIZZLE.
    This is useful for reording how blocks are ordered. A scheduler may, for example,
    assign sequential blocks 0, 1, 2, 3, ..., 8, 9, 10.. to its 8 hardware units 0, 1, 2, 3, ..., 0, 1, 2.
    This pattern may not be ideal for memory access, and it may be better to swizzle so the assignment
    becomes 0, 0, 0, 0, ..., 1, 1, 1, ... In the swizzled arrangement, sequential blocks are assigned to
    the same hardware unit.
    """
    # Number of pids per group in the new arrangement
    pids_per_group = domain_size // XCD_SWIZZLE
    extra_pid_groups = domain_size % XCD_SWIZZLE

    # Compute current current and local pid within the group
    group = pid % XCD_SWIZZLE
    local_pid = pid // XCD_SWIZZLE

    # Calculate new pid based on the new grouping
    new_pid = group * pids_per_group + min(group, extra_pid_groups) + local_pid
    return new_pid


@triton.jit
def swizzle2d(pid, grid_m, grid_n, GROUP_M: tl.constexpr):
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    tl.assume(group_size >= 0)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    return pid_m, pid_n


@triton.jit
def compute_pids(block_id, grid_m, grid_n, num_blocks, XCD_SWIZZLE: tl.constexpr, GROUP_M: tl.constexpr,
                 SPLIT_K: tl.constexpr):
    pid_zmnk = block_id
    if XCD_SWIZZLE != 1:
        pid_zmnk = xcd_swizzle(pid_zmnk, num_blocks, XCD_SWIZZLE)
    pid_z = pid_zmnk // (grid_m * grid_n * SPLIT_K)
    pid_mnk = pid_zmnk % (grid_m * grid_n * SPLIT_K)
    if SPLIT_K > 1:
        pid_k = pid_mnk % SPLIT_K
        pid_mn = pid_mnk // SPLIT_K
    else:
        pid_k: tl.constexpr = 0
        pid_mn = pid_mnk
    pid_m, pid_n = swizzle2d(pid_mn, grid_m, grid_n, GROUP_M)
    return pid_z, pid_m, pid_n, pid_k


@triton.jit
def compute_offsets(
    pid_m, pid_k, XBlockSchedule, XSliceOffs, XBlockOffs, BLOCK_M: tl.constexpr,
    BLOCK_K_X: tl.constexpr, PACKED_BLOCK_K_W: tl.constexpr,
):
    off_x_k = pid_k * BLOCK_K_X
    off_w_k = pid_k * PACKED_BLOCK_K_W
    block_schedule = tl.load(XBlockSchedule + pid_m)
    off_w_z = block_schedule & 65535
    block_id = block_schedule >> 16
    off_x_slice = tl.load(XSliceOffs + off_w_z)
    off_x_slice_tile = tl.load(XBlockOffs + off_w_z)
    off_x_z, off_y_z = (0, 0)
    off_x_m = BLOCK_M * block_id
    return (off_w_z, off_x_z, off_y_z, off_x_slice, off_x_slice_tile, off_x_m, off_x_k, off_w_k)


@triton.jit
def _load_writeback_idx_and_mask(WriteBackIndx, writeback_size, offs, mask):
    mask = mask & (offs < writeback_size)
    offs = tl.load(WriteBackIndx + offs, mask=mask, other=-1)
    mask = offs != -1
    return (offs, mask)


@triton.jit
def _matmul(
    Y, stride_y_k, stride_y_z, stride_y_m, stride_y_n, X, stride_x_z, stride_x_m, stride_x_k, W,
    stride_w_e, stride_w_k, stride_w_n, N, K, K_W, GatherIndx, WriteBackIndx, writeback_size,
    XSliceSizes, XSliceOffs, XBlockOffs, XBlockSchedule, batch_size, grid_m, grid_n,
    N_EXPTS_TOT: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, XCD_SWIZZLE: tl.constexpr, EVEN_K: tl.constexpr, SPLIT_K: tl.constexpr,
    W_CACHE_MODIFIER: tl.constexpr, UPCAST_INDICES: tl.constexpr,
):
    tl.assume(stride_y_k >= 0)
    tl.assume(stride_y_z >= 0)
    tl.assume(stride_y_m >= 0)
    tl.assume(stride_y_n >= 0)
    tl.assume(stride_x_z >= 0)
    tl.assume(stride_x_m >= 0)
    tl.assume(stride_x_k >= 0)
    tl.assume(stride_w_e >= 0)
    tl.assume(stride_w_k >= 0)
    tl.assume(stride_w_n >= 0)
    tl.assume(batch_size >= 0)
    tl.assume(grid_m >= 0)
    tl.assume(grid_n >= 0)
    PACKED_BLOCK_K_W: tl.constexpr = BLOCK_K
    PACKED_BLOCK_N_W: tl.constexpr = BLOCK_N
    pid = tl.program_id(0)
    padding_m = grid_m - tl.load(XBlockOffs + N_EXPTS_TOT)
    index_type: tl.constexpr = tl.int64 if UPCAST_INDICES else tl.int32
    unpadded_m = grid_m - padding_m
    tl.assume(unpadded_m >= 0)
    total_actual_tiles = batch_size * unpadded_m * grid_n * SPLIT_K
    if padding_m > 0 and pid >= total_actual_tiles:
        return
    pid_s, pid_m, pid_n, pid_k = compute_pids(pid, unpadded_m, grid_n, total_actual_tiles, XCD_SWIZZLE, GROUP_M, SPLIT_K)
    expt_id, start_z, start_z_out, start_m, _, off_m, off_k_x, off_k_w = compute_offsets(pid_m, pid_k, XBlockSchedule, XSliceOffs, XBlockOffs, BLOCK_M, BLOCK_K, PACKED_BLOCK_K_W)
    off_k_x = off_k_x // 1 * 1
    off_k_w = off_k_w // 1 * 1
    eM = tl.multiple_of(tl.load(XSliceSizes + expt_id), 1)
    K_W = K * (PACKED_BLOCK_K_W // BLOCK_K) if PACKED_BLOCK_K_W >= BLOCK_K else K // (BLOCK_K // PACKED_BLOCK_K_W)
    K_X = K
    loop_k = K - off_k_x
    k_tiles = tl.cdiv(loop_k, BLOCK_K * SPLIT_K)
    if SPLIT_K > 1:
        Y += pid_k.to(index_type) * stride_y_k
    expt_id, off_m = (expt_id.to(index_type), off_m.to(index_type))
    start_m, start_z = (start_m.to(index_type), start_z.to(index_type))
    pid_n, pid_k = (pid_n.to(index_type), pid_k.to(index_type))
    offs_x_m = off_m + tl.arange(0, BLOCK_M)
    offs_x_m = tl.max_contiguous(tl.multiple_of(offs_x_m % eM, BLOCK_M), BLOCK_M)
    X += start_z * stride_x_z
    if GatherIndx is None:
        X += start_m * stride_x_m
    else:
        GatherIndx += start_m
        offs_x_m = tl.load(GatherIndx + offs_x_m)
    offs_k = off_k_x + tl.arange(0, BLOCK_K)
    XPtrs = X + offs_x_m.to(index_type)[:, None] * stride_x_m + offs_k.to(index_type)[None, :] * stride_x_k
    offs_w_n = pid_n * PACKED_BLOCK_N_W + tl.arange(0, PACKED_BLOCK_N_W)
    N_W = N
    offs_w_n = tl.max_contiguous(tl.multiple_of(offs_w_n % (N_W // 1), PACKED_BLOCK_N_W), PACKED_BLOCK_N_W)
    offs_w_k = off_k_w + tl.arange(0, PACKED_BLOCK_K_W)
    W += expt_id * stride_w_e
    WPtrs = W + (offs_w_k.to(index_type)[:, None] * stride_w_k + offs_w_n.to(index_type)[None, :] * stride_w_n)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    x_k_limit = K_X + BLOCK_K * SPLIT_K
    w_k_limit = K_W + PACKED_BLOCK_K_W * SPLIT_K
    for ki in range(k_tiles):
        x_k_limit -= BLOCK_K * SPLIT_K
        w_k_limit -= PACKED_BLOCK_K_W * SPLIT_K
        if EVEN_K:
            mask_k_x = tl.full([BLOCK_K], True, dtype=tl.int1)
            mask_k_w = tl.full([PACKED_BLOCK_K_W], True, dtype=tl.int1)
        else:
            mask_k_x = offs_k < x_k_limit
            mask_k_w = offs_w_k < w_k_limit
        x = tl.load(XPtrs, mask=mask_k_x[None, :], other=0.0)
        w = tl.load(WPtrs, mask=mask_k_w[:, None], other=0.0, cache_modifier=W_CACHE_MODIFIER)
        acc = tl.dot(x, w, acc, max_num_imprecise_acc=None, allow_tf32=True)
        XPtrs += BLOCK_K * SPLIT_K * stride_x_k
        WPtrs += PACKED_BLOCK_K_W * SPLIT_K * stride_w_k
    offs_m = off_m + tl.arange(0, BLOCK_M)
    offs_y_n = BLOCK_N * pid_n + tl.arange(0, BLOCK_N)
    mask_m = offs_m < eM
    mask_n = offs_y_n < N
    bias = tl.full([BLOCK_N], 0, dtype=tl.float32)
    betas = tl.full([BLOCK_M], 1, dtype=tl.float32)
    gammas = tl.full([BLOCK_M], 1, dtype=tl.float32)
    x_scale = 1.0
    w_scale = 1.0
    acc *= x_scale * w_scale
    acc = acc + bias[None, :] * betas[:, None]
    out = acc
    out *= gammas[:, None]
    Y += start_z_out.to(index_type) * stride_y_z
    if WriteBackIndx is not None:
        WriteBackIndx += start_m
        dst_idx = tl.load(WriteBackIndx + offs_m, mask=start_m + offs_m < writeback_size, other=-1)
        mask_m = mask_m & (dst_idx != -1)
        offs_y_m = dst_idx
    else:
        Y += start_m * stride_y_m
        offs_y_m = offs_m
    YPtrs = Y + offs_y_m.to(index_type)[:, None] * stride_y_m + offs_y_n.to(index_type)[None, :] * stride_y_n
    mask = mask_m[:, None] & mask_n[None, :]
    out = out * 1.0
    tl.store(YPtrs, out, mask=mask)


@triton.jit
def _p_matmul(
    Y, YPtr, stride_y_k, stride_y_z, stride_y_m, stride_y_n, X, stride_x_z, stride_x_m, stride_x_k,
    W, W_TRANSPOSE: tl.constexpr, N, K, GatherIndx, WriteBackIndx, writeback_size, XSliceSizes,
    XSliceOffs, XBlockOffs, XBlockSchedule, batch_size, grid_n, N_SLICES: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
    XCD_SWIZZLE: tl.constexpr, EPILOGUE_SUBTILE: tl.constexpr, EVEN_K: tl.constexpr,
    SPLIT_K: tl.constexpr, NUM_SMS: tl.constexpr, X_TMA_MODE: tl.constexpr,
    Y_TMA_MODE: tl.constexpr,
):
    if Y_TMA_MODE is not None:
        Y = tl.make_tensor_descriptor(YPtr, Y.shape, Y.strides[:-1] + (1,), Y.block_shape)
    PACKED_BLOCK_K_W: tl.constexpr = BLOCK_K
    PACKED_BLOCK_N_W: tl.constexpr = BLOCK_N
    useful_grid_m = tl.load(XBlockOffs + N_SLICES)
    index_type: tl.constexpr = tl.int64
    HAS_SCATTER: tl.constexpr = WriteBackIndx is not None
    HAS_GATHER: tl.constexpr = GatherIndx is not None
    USE_GATHER_TMA: tl.constexpr = HAS_GATHER and X_TMA_MODE == 'dense'
    USE_SCATTER_TMA: tl.constexpr = HAS_SCATTER and Y_TMA_MODE == 'dense'
    if EPILOGUE_SUBTILE is None:
        SUBTILE_FACTOR: tl.constexpr = 1
    else:
        SUBTILE_FACTOR: tl.constexpr = EPILOGUE_SUBTILE
    EPILOGUE_BLOCK_N: tl.constexpr = BLOCK_N // SUBTILE_FACTOR
    OUT_BLOCK_N: tl.constexpr = EPILOGUE_BLOCK_N // 1
    yN = N // 1
    num_blocks = batch_size * useful_grid_m * grid_n * SPLIT_K
    INDEPENDENT_EPILOGUE: tl.constexpr = cuda_capability_geq(10, 0)
    if INDEPENDENT_EPILOGUE:
        tile_id1 = tl.program_id(0) - NUM_SMS
    for block_id in tl.range(tl.program_id(0), num_blocks, NUM_SMS, flatten=True, disallow_acc_multi_buffer=False, warp_specialize=True):
        pid_z, pid_m, pid_n, pid_k = compute_pids(block_id, useful_grid_m, grid_n, num_blocks, XCD_SWIZZLE, GROUP_M, SPLIT_K)
        off_w_z, off_x_z, off_y_z, slice_off_m, slice_block_off_m, off_m, off_k_x0, off_k_w0 = compute_offsets(pid_m, pid_k, XBlockSchedule, XSliceOffs, XBlockOffs, BLOCK_M, BLOCK_K, PACKED_BLOCK_K_W)
        shape_m = tl.load(XSliceSizes + off_w_z)
        off_n = BLOCK_N * pid_n
        off_w_n = PACKED_BLOCK_N_W * pid_n
        if USE_GATHER_TMA:
            offs_m = off_m + tl.arange(0, BLOCK_M)
            mask_m = offs_m < shape_m
            if XBlockSchedule is None:
                offs_x_m = tl.load(GatherIndx + slice_off_m.to(index_type) + offs_m, mask=mask_m)
                offs_x_m += off_x_z * (stride_x_z // stride_x_m)
                offs_x_m = tl.where(mask_m, offs_x_m, -1)
            else:
                offs_x_m = tl.load(GatherIndx + slice_off_m.to(index_type) + offs_m, mask=mask_m, other=-1)
        if X_TMA_MODE is None:
            XBase = X + off_x_z.to(index_type) * stride_x_z
            offs_m = off_m + tl.arange(0, BLOCK_M)
            offs_m = tl.max_contiguous(tl.multiple_of(offs_m % shape_m, BLOCK_M), BLOCK_M)
            if GatherIndx is not None:
                tl.static_assert(HAS_GATHER)
                offs_m = tl.load(GatherIndx + slice_off_m.to(index_type) + offs_m)
            offs_x_m = offs_m.to(index_type)[:, None] * stride_x_m
            offs_x_k = (off_k_x0.to(index_type) + tl.arange(0, BLOCK_K))[None, :] * stride_x_k
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        loop_k = K - off_k_x0
        k_tiles = tl.cdiv(loop_k, BLOCK_K * SPLIT_K)
        loop_bound = tl.maximum(k_tiles, 1)
        tl.assume(loop_bound > 0)
        for ki in tl.range(loop_bound, disallow_acc_multi_buffer=False):
            off_k_x = off_k_x0 + ki * BLOCK_K * SPLIT_K
            off_k_w = off_k_w0 + ki * PACKED_BLOCK_K_W * SPLIT_K
            if USE_GATHER_TMA:
                x = X.gather(offs_x_m, off_k_x)
            elif X_TMA_MODE == 'dense':
                x = X.load([off_x_z, slice_off_m + off_m, off_k_x])
                x = x.reshape(BLOCK_M, BLOCK_K)
            elif X_TMA_MODE == 'ragged':
                x = load_ragged(X, slice_off_m, shape_m, [off_x_z, off_m, off_k_x], ragged_dim=1)
                x = x.reshape(BLOCK_M, BLOCK_K)
            else:
                tl.static_assert(X_TMA_MODE is None)
                XPtrs = XBase + offs_x_m + offs_x_k
                XBase += BLOCK_K * SPLIT_K * stride_x_k
                mask_k = tl.arange(0, BLOCK_K) < K - off_k_x
                if EVEN_K:
                    if SPLIT_K > 1:
                        x = tl.load(XPtrs, mask=mask_k[None, :], other=0.0)
                    else:
                        x = tl.load(XPtrs)
                else:
                    x = tl.load(XPtrs, mask=mask_k[None, :], other=0.0)
            if W_TRANSPOSE:
                w = tl.reshape(W.load([off_w_z, off_w_n, off_k_w]), W.block_shape[1:]).T
            else:
                w = tl.reshape(W.load([off_w_z, off_k_w, off_w_n]), W.block_shape[1:])
            acc = tl.dot(x, w, acc, max_num_imprecise_acc=None, allow_tf32=True)
        if INDEPENDENT_EPILOGUE:
            tile_id1 += NUM_SMS
            pid_s1, pid_m1, pid_n1, pid_k1 = compute_pids(tile_id1, useful_grid_m, grid_n, num_blocks, XCD_SWIZZLE, GROUP_M, SPLIT_K)
            expt_id1, _, start_z1, start_m1, _, off_m1, _, _ = compute_offsets(pid_m, pid_k, XBlockSchedule, XSliceOffs, XBlockOffs, BLOCK_M, BLOCK_K, PACKED_BLOCK_K_W)
            off_n1 = pid_n1 * BLOCK_N
            eM1 = tl.load(XSliceSizes + expt_id1)
        else:
            tile_id1, expt_id1, start_z1, start_m1, eM1 = (block_id, off_w_z, off_y_z, slice_off_m, shape_m)
            off_m1, off_n1, pid_k1 = (off_m, off_n, pid_k)
        offs_m = off_m1 + tl.arange(0, BLOCK_M)
        mask_m = offs_m < eM1
        if USE_SCATTER_TMA:
            offs_y_m, mask_m = _load_writeback_idx_and_mask(WriteBackIndx, writeback_size, start_m1 + offs_m, mask_m)
            if SPLIT_K > 1:
                tl.device_assert(stride_y_k // stride_y_m == tl.cdiv(stride_y_k, stride_y_m))
                split_k_row_offs = pid_k1 * (stride_y_k // stride_y_m)
                offs_y_m = tl.where(mask_m, offs_y_m + split_k_row_offs, offs_y_m)
        elif Y_TMA_MODE is None and HAS_SCATTER:
            offs_y_m, mask_m = _load_writeback_idx_and_mask(WriteBackIndx, writeback_size, start_m1 + offs_m, mask_m)
        else:
            offs_y_m = start_m1 + offs_m
        offs_y_n = off_n1 + tl.arange(0, BLOCK_N)
        mask_n = offs_y_n < N
        bias = tl.full([BLOCK_N], 0, dtype=tl.float32)
        betas = tl.full([BLOCK_M], 1, dtype=tl.float32)
        gammas = tl.full([BLOCK_M], 1, dtype=tl.float32)
        x_scale = 1.0
        w_scale = 1.0
        accs = (acc,)
        biases = (bias,)
        if SUBTILE_FACTOR >= 2:
            acc = acc.reshape(BLOCK_M, 2, BLOCK_N // 2).permute(0, 2, 1)
            acc0, acc1 = acc.split()
            accs = (acc0, acc1)
            bias0, bias1 = bias.reshape(2, BLOCK_N // 2).permute(1, 0).split()
            biases = (bias0, bias1)
        if SUBTILE_FACTOR >= 4:
            acc0 = acc0.reshape(BLOCK_M, 2, BLOCK_N // 4).permute(0, 2, 1)
            acc1 = acc1.reshape(BLOCK_M, 2, BLOCK_N // 4).permute(0, 2, 1)
            acc00, acc01 = acc0.split()
            acc10, acc11 = acc1.split()
            accs = (acc00, acc01, acc10, acc11)
            bias00, bias01 = bias0.reshape(2, BLOCK_N // 4).permute(1, 0).split()
            bias10, bias11 = bias1.reshape(2, BLOCK_N // 4).permute(1, 0).split()
            biases = (bias00, bias01, bias10, bias11)
        tl.static_assert(EPILOGUE_BLOCK_N == BLOCK_N // SUBTILE_FACTOR)
        tl.static_assert(len(accs) == SUBTILE_FACTOR)
        for a_i in tl.static_range(len(accs)):
            acc_tile = accs[a_i]
            acc_tile *= x_scale * w_scale
            acc_tile = acc_tile + biases[a_i][None, :] * betas[:, None]
            out = acc_tile
            out *= gammas[:, None]
            out_off_n = off_n1 // 1 + a_i * OUT_BLOCK_N
            out = out * 1.0
            out = out.to(YPtr.dtype.element_ty)
            if USE_SCATTER_TMA:
                offs_y_m = (offs_y_m.to(tl.uint32, bitcast=True) & 2147483647).to(tl.int32, bitcast=True)
                Y.scatter(out, offs_y_m, out_off_n)
            elif Y_TMA_MODE == 'dense':
                out = tl.reshape(out, [1] + out.shape)
                off_kz = pid_k * batch_size + start_z1
                Y.store([off_kz, off_m1, out_off_n], out)
            elif Y_TMA_MODE == 'ragged':
                out = tl.reshape(out, [1] + out.shape)
                store_ragged(Y, start_m1, eM1, [pid_k, off_m1, out_off_n], out, ragged_dim=1)
            else:
                tl.static_assert(Y_TMA_MODE is None)
                offs_y_n = out_off_n + tl.arange(0, OUT_BLOCK_N)
                mask_n = offs_y_n < yN
                mask = mask_m[:, None] & mask_n[None, :]
                offs_kzmn = pid_k1.to(index_type) * stride_y_k + start_z1.to(index_type) * stride_y_z + offs_y_m.to(index_type)[:, None] * stride_y_m + offs_y_n[None, :] * stride_y_n
                tl.store(YPtr + offs_kzmn, out, mask=mask)


_per_device_alloc_fns = {}


def get_per_device_per_stream_alloc_fn(device):
    if device not in _per_device_alloc_fns:
        _per_stream_tensors = collections.defaultdict(list)

        def alloc_fn(size: int, alignment: int, stream: int):
            assert alignment == 128
            tensors = _per_stream_tensors[stream]
            if not tensors or tensors[-1].numel() < size:
                tensors.append(torch.empty(size, device=device, dtype=torch.int8))
                tensors[-1].__hibernate__ = {"type": "ignore"}
            return tensors[-1]

        _per_device_alloc_fns[device] = alloc_fn
    return _per_device_alloc_fns[device]


@dataclass(frozen=True)
class LaunchConfig:
    block_m: int
    block_n: int
    block_k: int
    split_k: int
    num_warps: int
    num_stages: int
    persistent: bool
    epilogue_subtile: int = 1
    maxnreg: int | None = None


def _launch_config(m, n, k, metadata, can_use_tma, can_split_k):
    """The upstream defaults specialized to unquantized, row-ragged MoE."""
    props = torch.cuda.get_device_properties(0)
    sms = props.multi_processor_count
    slice_size = max(1, m // metadata.n_slices)
    if is_hip():
        arch = triton.runtime.driver.active.get_current_target().arch
        cdna4 = arch == 'gfx950'
        rdna = arch.startswith(('gfx11', 'gfx12')) and not arch.startswith('gfx125')
        if slice_size >= 512 and n >= 2048:
            bm = 256 if cdna4 else 128
        elif cdna4 and m >= 512:
            bm = 128
        elif rdna and m >= 512:
            bm = 64
        else:
            bm = max(32, min(triton.next_power_of_2(slice_size), 64))
        gm = metadata.n_blocks(metadata.n_slices, m, bm)
        bn = n if n <= 128 and n & (n - 1) == 0 else max(32, min(64 if cdna4 else 256, triton.next_power_of_2(gm * n * 8 // sms)))
        if rdna and bm == 64:
            bn = 256
        split = max(1, sms // (gm * triton.cdiv(n, bn))) if can_split_k else 1
        return LaunchConfig(bm, bn, 64, split, 2 if m <= 16 else 8, 2, False)
    if slice_size <= 64:
        bm = max(16, min(triton.next_power_of_2(2 * slice_size), 64))
    else:
        bm = max(16, min(triton.next_power_of_2(slice_size), 128))
    bn = 256 if n > 128 else max(8, min(128, triton.next_power_of_2(n)))
    bn_tma = max(16, bn)
    gm = metadata.n_blocks(metadata.n_slices, m, bm)
    persistent = can_use_tma and gm * triton.cdiv(n, bn_tma) / sms >= 2.0 and m * n * k >= 131072
    if persistent:
        bn = bn_tma
    bk = max(32 if persistent else 16, min(triton.next_power_of_2(k), 64))
    split = 1
    if can_split_k:
        grid = triton.cdiv(m, bm) * triton.cdiv(n, bn)
        split = max(1, min(sms // grid, triton.cdiv(k, bk) // 4))
    nw = max(bm * bn // 4096, 4 if persistent else 1)
    is_bw = cuda_capability_geq(10, 0)
    maxnreg = min(256, 65536 // (nw * 32)) if persistent and not is_bw else None
    best_stages, best_ep = -1, 1
    for ep in (1, 2, 4):
        stage_size = (bm + bn) * bk * 2
        available = props.shared_memory_per_block_optin
        if persistent:
            stage_size += 8
            available -= (bm + 4) * (bn // ep if is_bw else bn) * (4 if split > 1 else 2)
        stages = max(1, min(available // stage_size, 4))
        if stages > best_stages:
            best_stages, best_ep = stages, ep
    return LaunchConfig(bm, bn, bk, split, nw, best_stages, persistent, best_ep, maxnreg)


def _tma_compliant(tensor):
    if not cuda_capability_geq(9, 0):
        return False
    strides = tensor.stride()
    major = strides.index(1) if 1 in strides else -1
    return all(s * tensor.element_size() % 16 == 0 for i, s in enumerate(strides) if i != major)


def _canonicalize(tensor, ndim):
    padding = ndim - tensor.ndim
    return tensor.as_strided([1] * padding + list(tensor.shape), [0] * padding + list(tensor.stride()))


def _descriptor(tensor, blocks, ragged=False):
    if ragged:
        return create_ragged_descriptor(tensor, blocks, ragged_dim=tensor.ndim - 2)
    shape, strides = list(tensor.shape), list(tensor.stride())
    if strides[-1] != 1:
        shape[-2:] = reversed(shape[-2:])
        strides[-2:] = reversed(strides[-2:])
        blocks = blocks[:-2] + [blocks[-1], blocks[-2]]
    return TensorDescriptor(tensor, shape, strides, blocks)


def matmul(a, b, bias, *, a_ragged_metadata, gather_indx=None, scatter_indx=None):
    """Gather-FC1 or scatter-FC2, with FP32 accumulation and split-K scratch."""
    if bias is not None or a.dtype not in (torch.float16, torch.bfloat16) or b.dtype != a.dtype:
        raise ValueError("Expert GEMM requires unquantized BF16/FP16 tensors without bias")
    if a.ndim != 2 or b.ndim != 3 or a.stride(-1) != 1 or a.shape[-1] != b.shape[-2]:
        raise ValueError("Expected contiguous token rows and [expert, input, output] weights")
    metadata = a_ragged_metadata
    m = a.shape[0] if gather_indx is None else gather_indx.shape[0]
    n, k = b.shape[-1], a.shape[-1]
    cfg = _launch_config(m, n, k, metadata, _tma_compliant(a) and _tma_compliant(b), scatter_indx is None)
    bm, bn, bk, split = cfg.block_m, cfg.block_n, cfg.block_k, cfg.split_k
    rows = m if scatter_indx is None else scatter_indx.shape[0]
    output = torch.empty((1, rows, n), device=a.device, dtype=a.dtype)
    scratch = torch.empty((split, 1, m, n), device=a.device, dtype=torch.float32) if split > 1 else output[None]
    if not m * n:
        return output.squeeze(0)
    has_gather_tma = gather_indx is not None and cuda_capability_geq(10, 0)
    has_scatter_tma = scatter_indx is not None and cuda_capability_geq(10, 0)
    a_storage = _canonicalize(a, 2 if has_gather_tma else 3)
    y_storage = scratch.view(-1, n) if scatter_indx is not None else scratch.view(-1, rows, n)
    y_storage = _canonicalize(y_storage, 2 if has_scatter_tma else 3)
    a_has_tma = cfg.persistent and (has_gather_tma or gather_indx is None)
    c_has_tma = cfg.persistent and (scatter_indx is None or has_scatter_tma)
    x_mode = ('dense' if has_gather_tma else 'ragged') if a_has_tma else None
    y_mode = ('dense' if has_scatter_tma else 'ragged') if c_has_tma else None
    a_desc = _descriptor(a_storage, [1, bk] if has_gather_tma else [1, bm, bk], x_mode == 'ragged') if a_has_tma else a_storage
    out_bn = bn // cfg.epilogue_subtile
    y_desc = _descriptor(y_storage, [1, out_bn] if has_scatter_tma else [1, bm, out_bn], y_mode == 'ragged') if c_has_tma else y_storage
    b_desc = _descriptor(b, [1, bk, bn]) if cfg.persistent else b
    gm = metadata.n_blocks(metadata.n_slices, m, bm)
    gn = triton.cdiv(n, bn)
    grid = gm * gn * split
    if cfg.persistent:
        grid = min(torch.cuda.get_device_properties(0).multi_processor_count, grid)
        triton.set_allocator(get_per_device_per_stream_alloc_fn(a.device))
    upcast = any(sum((dim - 1) * stride for dim, stride in zip(t.shape, t.stride())) > (1 << 31) - 1 for t in (a, b, scratch))
    a_strides = [0] * (3 - a_storage.ndim) + list(a_storage.stride())
    args = dict(
        Y=y_desc, YPtr=y_storage, stride_y_k=scratch.stride(0), stride_y_z=scratch.stride(1), stride_y_m=scratch.stride(2), stride_y_n=scratch.stride(3),
        X=a_desc, XPtr=a_storage, stride_x_z=a_strides[0], stride_x_m=a_strides[1], stride_x_k=a_strides[2], X_TRANSPOSE=False,
        W=b_desc, WPtr=b, stride_w_e=b.stride(0), stride_w_k=b.stride(1), stride_w_n=b.stride(2), W_TRANSPOSE=b.stride(-2) == 1,
        M=None, N=n, K=k, K_W=k, GatherIndx=gather_indx, WriteBackIndx=scatter_indx,
        writeback_size=None if scatter_indx is None else scatter_indx.shape[0],
        XSliceSizes=metadata.slice_sizes, XSliceOffs=metadata.slice_offs,
        XBlockOffs=metadata.block_offs(bm), XBlockSchedule=metadata.block_schedule(bm),
        batch_size=1, grid_m=gm, grid_n=gn, N_SLICES=b.shape[0], N_EXPTS_TOT=b.shape[0],
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=4 if is_hip() else 8,
        XCD_SWIZZLE=8 if is_hip() else 1, EPILOGUE_SUBTILE=cfg.epilogue_subtile,
        EVEN_K=k % bk == 0, SPLIT_K=split, W_CACHE_MODIFIER='.cg' if is_hip() and bm <= 32 else None,
        NUM_SMS=grid if cfg.persistent else 0, X_TMA_MODE=x_mode, Y_TMA_MODE=y_mode, UPCAST_INDICES=upcast,
    )
    kernel = _p_matmul if cfg.persistent else _matmul
    options = dict(num_warps=cfg.num_warps, num_stages=cfg.num_stages)
    if is_hip():
        options.update(waves_per_eu=0, matrix_instr_nonkdim=16, kpack=1)
    else:
        options.update(maxnreg=cfg.maxnreg)
    kernel[(grid,)](**{name: args[name] for name in kernel.arg_names}, **options)
    if split > 1:
        reduce_forward(scratch.view(split, -1, n), dim=0, y=output.view(-1, n))
    return output.squeeze(0)
