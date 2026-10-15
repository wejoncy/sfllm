/* Copyright 2026 SFLLM contributors. Licensed under the Apache License,
 * Version 2.0. */

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <torch/all.h>

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

namespace {
using namespace cute;

void check_cutlass(cutlass::Status status) {
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              cutlassGetStatusString(status));
}

template <int BlockM, int BlockN, int BlockK, int Stages, bool Swapped = true>
struct ScaledGemmTraits {
  using Tile = Shape<Int<BlockM>, Int<BlockN>, Int<BlockK>>;
  using FP8 = cutlass::float_e4m3_t;
  using BF16 = cutlass::bfloat16_t;
  // Swap the problem's M/N dimensions so a small decode M occupies the
  // contiguous epilogue dimension. Apply row scales first, channel scales
  // second, and preserve the original final BF16 rounding.
  using Accum = cutlass::epilogue::fusion::Sm90AccFetch;
  using ColScale =
      cutlass::epilogue::fusion::Sm90ColBroadcast<0, Tile, float, float,
                                                  Stride<_1, _0, _0>, 4, false>;
  using RowScale =
      cutlass::epilogue::fusion::Sm90RowBroadcast<0, Tile, float, float,
                                                  Stride<_0, _1, _0>, 4, false>;
  using ChannelScale = conditional_t<Swapped, ColScale, RowScale>;
  using TokenScale = conditional_t<Swapped, RowScale, ColScale>;
  using OutputLayout = conditional_t<Swapped, cutlass::layout::ColumnMajor,
                                     cutlass::layout::RowMajor>;
  using ScaleFloat = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies, float, float,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using ScaleBF16 = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies, BF16, float,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using ScaledAccum =
      cutlass::epilogue::fusion::Sm90EVT<ScaleFloat, TokenScale, Accum>;
  using EpilogueTree =
      cutlass::epilogue::fusion::Sm90EVT<ScaleBF16, ChannelScale, ScaledAccum>;
  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, Tile,
      Shape<_1, _1, _1>, cutlass::epilogue::collective::EpilogueTileAuto, float,
      float, BF16, OutputLayout, 8, BF16, OutputLayout, 8,
      cutlass::epilogue::TmaWarpSpecialized, EpilogueTree>::CollectiveOp;
  using StageCount = conditional_t<
      Stages == 0,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename Epilogue::SharedStorage))>,
      cutlass::gemm::collective::StageCount<Stages>>;
  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, FP8,
      cutlass::layout::RowMajor, 16, FP8, cutlass::layout::ColumnMajor, 16,
      float, Tile, Shape<_1, _1, _1>, StageCount,
      cutlass::gemm::KernelTmaWarpSpecializedFP8FastAccum>::CollectiveOp;
  using Kernel =
      cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Mainloop,
                                           Epilogue,
                                           cutlass::gemm::PersistentScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

template <int BlockM, int BlockN, int BlockK, int Stages, bool Swapped = true>
at::Tensor launch(const at::Tensor &input, const at::Tensor &weight,
                  const at::Tensor &input_scale, const at::Tensor &weight_scale,
                  bool enable_pdl) {
  using Traits = ScaledGemmTraits<BlockM, BlockN, BlockK, Stages, Swapped>;
  using Kernel = typename Traits::Kernel;
  using Gemm = typename Traits::Gemm;
  using FP8 = typename Traits::FP8;
  using BF16 = typename Traits::BF16;
  using ScaledAccum = typename Traits::ScaledAccum;
  using EpilogueTree = typename Traits::EpilogueTree;

  const int rows = input.size(0), columns = weight.size(1),
            inner = input.size(1);
  auto output =
      at::empty({rows, columns}, input.options().dtype(at::kBFloat16));
  const int problem_m = Swapped ? columns : rows,
            problem_n = Swapped ? rows : columns;
  auto a_stride = cutlass::make_cute_packed_stride(
      typename Kernel::StrideA{}, make_shape(problem_m, inner, 1));
  auto b_stride = cutlass::make_cute_packed_stride(
      typename Kernel::StrideB{}, make_shape(problem_n, inner, 1));
  auto d_stride = cutlass::make_cute_packed_stride(
      typename Kernel::StrideD{}, make_shape(problem_m, problem_n, 1));
  typename ScaledAccum::Arguments scaled_accum{
      {input_scale.data_ptr<float>()}, {}, {}};
  typename EpilogueTree::Arguments scales{
      {weight_scale.data_ptr<float>()}, scaled_accum, {}};
  cutlass::KernelHardwareInfo hardware;
  hardware.device_id = input.get_device();
  hardware.sm_count =
      at::cuda::getDeviceProperties(input.get_device())->multiProcessorCount;
  typename Kernel::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      make_shape(problem_m, problem_n, inner, 1),
      {(const FP8 *)(Swapped ? weight.data_ptr() : input.data_ptr()), a_stride,
       (const FP8 *)(Swapped ? input.data_ptr() : weight.data_ptr()), b_stride},
      {scales, (const BF16 *)output.data_ptr(), d_stride,
       (BF16 *)output.data_ptr(), d_stride},
      hardware,
      {}};
  Gemm operation;
  check_cutlass(operation.can_implement(args));
  auto workspace =
      at::empty({static_cast<int64_t>(operation.get_workspace_size(args))},
                input.options().dtype(at::kByte));
  auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
  check_cutlass(
      operation.run(args, workspace.data_ptr(), stream, nullptr, enable_pdl));
  return output;
}
using G = ScaledGemmTraits<64, 64, 128, 4>;

struct GateEpilogue {
  using Base = G::Epilogue;
  using ThreadEpilogueOp = typename Base::ThreadEpilogueOp;
  using ElementC = cutlass::bfloat16_t;
  using ElementD = ElementC;
  using StrideC = typename Base::StrideC;
  using StrideD = typename Base::StrideD;
  using LoadPipeline = typename Base::LoadPipeline;
  using StorePipeline = typename Base::StorePipeline;
  using LoadPipelineState = typename Base::LoadPipelineState;
  using StorePipelineState = typename Base::StorePipelineState;
  using PipelineStorage = typename Base::PipelineStorage;
  static constexpr bool RequiresTransactionBytes = false;
  struct TensorStorage {
    alignas(128) __nv_bfloat16 activated[64 * 32];
  };
  struct Arguments {
    float const *rows;
    float const *channels;
    __nv_bfloat16 *out;
    float *maxima;
  };
  using Params = Arguments;
  Params params;
  CUTLASS_DEVICE GateEpilogue(Params const &p, TensorStorage &) : params(p) {}
  template <class P>
  static Params to_underlying_arguments(P const &, Arguments const &a, void *) {
    return a;
  }
  template <class P> static bool can_implement(P const &p, Arguments const &) {
    return get<0>(p) > 0 && get<0>(p) % 64 == 0 && get<1>(p) <= 192 &&
           get<1>(p) > 0 && get<2>(p) % 16 == 0;
  }
  template <class P>
  static size_t get_workspace_size(P const &, Arguments const &) {
    return 0;
  }
  template <class P>
  static cutlass::Status
  initialize_workspace(P const &, Arguments const &, void *, cudaStream_t,
                       cutlass::CudaHostAdapter * = nullptr) {
    return cutlass::Status::kSuccess;
  }
  CUTLASS_DEVICE static void prefetch_tma_descriptors(Params const &) {}
  CUTLASS_DEVICE bool is_producer_load_needed() const { return false; }
  template <class... X>
  CUTLASS_DEVICE auto load(LoadPipeline, LoadPipelineState s, X...) {
    return s;
  }
  CUTLASS_DEVICE void load_tail(LoadPipeline, LoadPipelineState) {}
  CUTLASS_DEVICE void store_tail(LoadPipeline, LoadPipelineState, StorePipeline,
                                 StorePipelineState) {}
  template <class P, class TS, class TC, class AE, class AL, class TM>
  CUTLASS_DEVICE auto store(LoadPipeline, LoadPipelineState ls, StorePipeline,
                            StorePipelineState ss, P problem, TS tile, TC block,
                            Tensor<AE, AL> acc, TM mma, int tid,
                            TensorStorage &smem) {
    static_assert(size<1>(AL{}) == 1 && size<2>(AL{}) == 1);
    auto coord = mma.get_thread_slice(tid).partition_C(
        make_identity_tensor(make_shape(_64{}, _64{})));
    int tile_id = get<0>(block), row_tile = get<1>(block);
    int rows = get<1>(problem);
    int row_base = row_tile * 64;
    int width = get<0>(problem) / 2, tiles = get<0>(problem) / 64;
    // Each pair of accumulator fragments holds interleaved gate/up channels
    // separated by eight columns within the 16-column startup packing group.
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 16; i++) {
      constexpr int PAIR_STRIDE = 2;
      int gi = (i / 2) * 4 + i % 2, ui = gi + PAIR_STRIDE;
      auto c = coord(gi, 0, 0);
      int chan = get<0>(c), lr = get<1>(c), row = lr + row_base;
      float rs = row < rows ? params.rows[row] : 0.f;
      float g = __bfloat162float(__float2bfloat16_rn(
          (acc(gi, 0, 0) * rs) * params.channels[tile_id * 64 + chan]));
      float u = __bfloat162float(__float2bfloat16_rn(
          (acc(ui, 0, 0) * rs) * params.channels[tile_id * 64 + chan + 8]));
      __nv_bfloat16 v =
          __float2bfloat16_rn(__fdividef(g, 1.f + __expf(-g)) * u);
      int outchan = (chan / 16) * 8 + chan % 8;
      smem.activated[lr * 32 + (outchan ^ (((lr >> 1) & 3) * 8))] = v;
      // Bound epilogue live ranges without emitting a GPU instruction.
      asm volatile("" :: "h"(__bfloat16_as_ushort(v)) : "memory");
    }
    cutlass::arch::NamedBarrier::sync(
        128, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < 2; j++) {
      int lr = tid / 4 + j * 32, row = row_base + lr, col = (tid % 4) * 8;
      auto *src = smem.activated + lr * 32 + (col ^ (((lr >> 1) & 3) * 8));
      uint4 vec = *reinterpret_cast<uint4 const *>(src);
      if (row < rows)
        *reinterpret_cast<uint4 *>(params.out + row * width + tile_id * 32 +
                                   col) = vec;
      float mx = 0.f;
      CUTLASS_PRAGMA_UNROLL
      for (int z = 0; z < 8; z++)
        mx =
            fmaxf(mx, fabsf(__bfloat162float(__ushort_as_bfloat16(
                          reinterpret_cast<unsigned short const *>(&vec)[z]))));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 1));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 2));
      if (tid % 4 == 0 && row < rows)
        params.maxima[row * tiles + tile_id] = mx;
    }
    return make_tuple(ls, ss);
  }
};
struct GateKernel
    : cutlass::gemm::kernel::GemmUniversal<
          Shape<int, int, int, int>, G::Mainloop, GateEpilogue,
          cutlass::gemm::PersistentScheduler> {
  static constexpr int MinBlocksPerMultiprocessor = 3;
};

std::vector<at::Tensor> launch_gate(const at::Tensor &a, const at::Tensor &w,
                                    const at::Tensor &sa, const at::Tensor &sw,
                                    bool enable_pdl) {
  using K = GateKernel;
  using AB = cutlass::float_e4m3_t;
  int rows = a.size(0), columns = w.size(1), inner = a.size(1);
  auto out = at::empty({rows, columns / 2}, a.options().dtype(at::kBFloat16));
  auto maxima = at::empty({rows, columns / 64}, a.options().dtype(at::kFloat));
  auto problem = make_shape(columns, rows, inner, 1);
  auto as = cutlass::make_cute_packed_stride(typename K::StrideA{},
                                             make_shape(columns, inner, 1));
  auto bs = cutlass::make_cute_packed_stride(typename K::StrideB{},
                                             make_shape(rows, inner, 1));
  typename K::MainloopArguments main{(const AB *)w.data_ptr(), as,
                                     (const AB *)a.data_ptr(), bs};
  typename K::EpilogueArguments epilogue{
      sa.data_ptr<float>(), sw.data_ptr<float>(),
      (__nv_bfloat16 *)out.data_ptr(), maxima.data_ptr<float>()};
  cutlass::KernelHardwareInfo hardware;
  hardware.device_id = a.get_device();
  hardware.sm_count =
      at::cuda::getDeviceProperties(a.get_device())->multiProcessorCount;
  typename K::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,
                             problem,
                             main,
                             epilogue,
                             hardware,
                             {}};
  cutlass::gemm::device::GemmUniversalAdapter<K> operation;
  check_cutlass(operation.can_implement(args));
  auto workspace =
      at::empty({static_cast<int64_t>(operation.get_workspace_size(args))},
                a.options().dtype(at::kByte));
  check_cutlass(operation.run(args, workspace.data_ptr(),
                              at::cuda::getCurrentCUDAStream(a.get_device()),
                              nullptr, enable_pdl));
  return {out, maxima};
}

} // namespace

at::Tensor fp8_scaled_mm_sm90(const at::Tensor &input, const at::Tensor &weight,
                              const at::Tensor &input_scale,
                              const at::Tensor &weight_scale, int64_t config,
                              bool enable_pdl) {
  TORCH_CHECK(input.is_cuda() && weight.device() == input.device() &&
                  input_scale.device() == input.device() &&
                  weight_scale.device() == input.device(),
              "SM90 FP8 GEMM operands must be on the same CUDA device");
  const c10::cuda::CUDAGuard guard(input.device());
  TORCH_CHECK(at::cuda::getDeviceProperties(input.get_device())->major == 9,
              "SM90 FP8 GEMM requires Hopper");
  TORCH_CHECK(input.scalar_type() == at::kFloat8_e4m3fn &&
                  weight.scalar_type() == at::kFloat8_e4m3fn,
              "SM90 FP8 GEMM requires e4m3fn operands");
  TORCH_CHECK(input.dim() == 2 && weight.dim() == 2 && input.is_contiguous() &&
                  weight.stride(0) == 1 && weight.stride(1) == weight.size(0),
              "SM90 FP8 GEMM requires row-major input and column-major weight");
  const int64_t rows = input.size(0), inner = input.size(1),
                columns = weight.size(1);
  TORCH_CHECK(
      rows > 0 && rows <= 192 && inner == weight.size(0) && inner % 16 == 0 &&
          columns > 0 && columns % 8 == 0,
      "SM90 FP8 GEMM requires 1..192 rows and aligned inner/output dimensions");
  TORCH_CHECK(input_scale.scalar_type() == at::kFloat &&
                  weight_scale.scalar_type() == at::kFloat &&
                  input_scale.is_contiguous() && weight_scale.is_contiguous() &&
                  input_scale.numel() == rows &&
                  weight_scale.numel() == columns,
              "SM90 FP8 GEMM requires FP32 row and channel scales");
  TORCH_CHECK(config == 0 || config == 1 || config == 2 || config == 4 ||
                  config == 6,
              "Invalid SM90 FP8 GEMM tile configuration");
  if (config == 0)
    return launch<64, 64, 128, 4>(input, weight, input_scale, weight_scale,
                                  enable_pdl);
  if (config == 1)
    return launch<64, 64, 256, 3>(input, weight, input_scale, weight_scale,
                                  enable_pdl);
  if (config == 2)
    return launch<64, 32, 256, 0>(input, weight, input_scale, weight_scale,
                                  enable_pdl);
  if (config == 4)
    return launch<64, 32, 256, 0, false>(input, weight, input_scale,
                                         weight_scale, enable_pdl);
  return launch<64, 64, 256, 0, false>(input, weight, input_scale, weight_scale,
                                       enable_pdl);
}

std::vector<at::Tensor> fp8_gate_up_sm90(const at::Tensor &input,
                                         const at::Tensor &weight,
                                         const at::Tensor &input_scale,
                                         const at::Tensor &weight_scale,
                                         bool enable_pdl) {
  TORCH_CHECK(input.is_cuda() && weight.device() == input.device() &&
                  input_scale.device() == input.device() &&
                  weight_scale.device() == input.device(),
              "SM90 gate/up operands must be on the same CUDA device");
  const c10::cuda::CUDAGuard guard(input.device());
  TORCH_CHECK(at::cuda::getDeviceProperties(input.get_device())->major == 9,
              "SM90 gate/up requires Hopper");
  TORCH_CHECK(input.scalar_type() == at::kFloat8_e4m3fn &&
                  weight.scalar_type() == at::kFloat8_e4m3fn,
              "SM90 gate/up requires e4m3fn operands");
  TORCH_CHECK(
      input.dim() == 2 && weight.dim() == 2 && input.is_contiguous() &&
          weight.stride(0) == 1 && weight.stride(1) == weight.size(0),
      "SM90 gate/up requires row-major input and packed column-major weight");
  TORCH_CHECK(
      input.size(0) > 0 && input.size(0) <= 192 &&
          input.size(1) == weight.size(0) && input.size(1) % 16 == 0 &&
          weight.size(1) > 0 && weight.size(1) % 64 == 0,
      "SM90 gate/up requires 1..192 rows and aligned inner/output dimensions");
  TORCH_CHECK(input_scale.scalar_type() == at::kFloat &&
                  weight_scale.scalar_type() == at::kFloat &&
                  input_scale.is_contiguous() && weight_scale.is_contiguous() &&
                  input_scale.numel() == input.size(0) &&
                  weight_scale.numel() == weight.size(1),
              "SM90 gate/up requires FP32 row and packed channel scales");
  return launch_gate(input, weight, input_scale, weight_scale, enable_pdl);
}
