/*
 * Copyright (c) 2024 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#ifndef USE_ROCM

#include <flashinfer/activation.cuh>

#include "utils.h"

#else
#include "hip/hip_act_and_mul.cuh"
#endif

#include "cast.cuh"

// Adapted from flashinfer activation
// https://github.com/flashinfer-ai/flashinfer/blob/4e8eb1879f9c3ba6d75511e5893183bf8f289a62/csrc/activation.cu#L44

template <typename T>
__device__ __forceinline__ T silu(const T& x) {
  float f32_val = to_float(x);
  return from_float<T>(f32_val / (1.0f + expf(-f32_val)));
}

#ifndef USE_ROCM
// Flatten vectors across tokens so small decode batches launch enough blocks.
template <typename T>
__global__ void silu_and_mul_flat_kernel(
    T* __restrict__ out,
    const T* __restrict__ input,
    const uint32_t d,
    const uint32_t num_tokens) {
  constexpr uint32_t vec_size = 16 / sizeof(T);
  const uint32_t vecs_per_token = d / vec_size;
  const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= num_tokens * vecs_per_token) {
    return;
  }

  const uint32_t token = index / vecs_per_token;
  const uint32_t column = (index % vecs_per_token) * vec_size;
  const uint32_t input_offset = token * 2 * d + column;
  flashinfer::vec_t<float, vec_size> gate, up, output;
  gate.cast_load(input + input_offset);
  up.cast_load(input + input_offset + d);
#pragma unroll
  for (uint32_t i = 0; i < vec_size; ++i) {
    output[i] = silu(gate[i]) * up[i];
  }
  output.cast_store(out + token * d + column);
}

template <typename T, uint32_t vec_size>
__global__ void fused_sigmoid_mul_kernel(
    T* __restrict__ output,
    const T* __restrict__ gate,
    const uint32_t num_heads,
    const uint32_t head_dim,
    const int64_t gate_token_stride,
    const int64_t gate_head_stride) {
  const uint32_t token = blockIdx.x;
  const uint32_t head = blockIdx.y * blockDim.y + threadIdx.y;
  if (head >= num_heads) {
    return;
  }
  const uint64_t row = static_cast<uint64_t>(token) * num_heads + head;
  const int64_t gate_offset = token * gate_token_stride +
      head * gate_head_stride;
  for (uint32_t column = threadIdx.x * vec_size; column < head_dim;
       column += blockDim.x * vec_size) {
    const uint64_t index = row * head_dim + column;
    flashinfer::vec_t<float, vec_size> values, gates;
    // vec_t converts FP16/BF16 pairs with __half22float2/__bfloat1622float2.
    values.cast_load(output + index);
    gates.cast_load(gate + gate_offset + column);
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      values[i] *= __fdividef(1.0f, 1.0f + __expf(-gates[i]));
    }
    values.cast_store(output + index);
  }
}
#endif

template <typename T>
__device__ __forceinline__ T gelu(const T& x) {
  constexpr float kAlpha = M_SQRT1_2;
  float f32_val = to_float(x);
  return from_float<T>(f32_val * (0.5f * (1.0f + erf(f32_val * kAlpha))));
}

// gelu_quick(x) = x * torch.sigmoid(1.702 * x)
template <typename T>
__device__ __forceinline__ T gelu_quick_act(const T& x) {
  float f32_val = to_float(x);
  return from_float<T>(f32_val / (1.0f + expf(-f32_val * 1.702f)));
}

template <typename T>
__device__ __forceinline__ T gelu_tanh(const T& x) {
  constexpr float kAlpha = 0.044715f;
  constexpr float kBeta = 0.7978845608028654f;
  float f32_val = to_float(x);
  const float cdf = 0.5f * (1.0f + tanhf((kBeta * (f32_val + kAlpha * f32_val * f32_val * f32_val))));
  return from_float<T>(f32_val * cdf);
}

// Helpers to avoid #if inside macro arguments
template<typename c_type>
void launch_silu_kernel(at::Tensor& out, at::Tensor& input, int d, dim3 grid, dim3 block, cudaStream_t stream) {
#if USE_ROCM
    sgl_hip::activation::act_and_mul_kernel<c_type, silu>
        <<<grid, block, 0, stream>>>(static_cast<c_type*>(out.data_ptr()), static_cast<c_type*>(input.data_ptr()), d);
#else
    flashinfer::activation::act_and_mul_kernel<c_type, silu>
        <<<grid, block, 0, stream>>>(static_cast<c_type*>(out.data_ptr()), static_cast<c_type*>(input.data_ptr()), d);
#endif
}

template<typename c_type>
void launch_gelu_kernel(at::Tensor& out, at::Tensor& input, int d, dim3 grid, dim3 block, cudaStream_t stream) {
#if USE_ROCM
    sgl_hip::activation::act_and_mul_kernel<c_type, gelu>
        <<<grid, block, 0, stream>>>(static_cast<c_type*>(out.data_ptr()), static_cast<c_type*>(input.data_ptr()), d);
#else
    flashinfer::activation::act_and_mul_kernel<c_type, gelu>
        <<<grid, block, 0, stream>>>(static_cast<c_type*>(out.data_ptr()), static_cast<c_type*>(input.data_ptr()), d);
#endif
}

template<typename c_type>
void launch_gelu_tanh_kernel(at::Tensor& out, at::Tensor& input, int d, dim3 grid, dim3 block, cudaStream_t stream) {
#if USE_ROCM
    sgl_hip::activation::act_and_mul_kernel<c_type, gelu_tanh>
        <<<grid, block, 0, stream>>>(static_cast<c_type*>(out.data_ptr()), static_cast<c_type*>(input.data_ptr()), d);
#else
    flashinfer::activation::act_and_mul_kernel<c_type, gelu_tanh>
        <<<grid, block, 0, stream>>>(static_cast<c_type*>(out.data_ptr()), static_cast<c_type*>(input.data_ptr()), d);
#endif
}

void silu_and_mul(at::Tensor& out, at::Tensor& input) {
  int d = input.size(-1) / 2;
  int64_t num_tokens = input.numel() / input.size(-1);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(input.scalar_type(), c_type, [&] {
    uint32_t vec_size = 16 / sizeof(c_type);
#ifndef USE_ROCM
    if (d % vec_size == 0) {
      constexpr uint32_t block_size = 256;
      uint32_t num_vectors = num_tokens * d / vec_size;
      dim3 grid((num_vectors + block_size - 1) / block_size);
      silu_and_mul_flat_kernel<c_type>
          <<<grid, block_size, 0, stream>>>(
              static_cast<c_type*>(out.data_ptr()),
              static_cast<c_type*>(input.data_ptr()),
              d,
              num_tokens);
      return true;
    }
#endif
    dim3 grid(num_tokens);
    dim3 block(std::min(d / vec_size, 1024U));
    launch_silu_kernel<c_type>(out, input, d, grid, block, stream);
    return true;
  });
}

void fused_sigmoid_mul(at::Tensor& output, const at::Tensor& gate) {
  CHECK_INPUT(output);
  TORCH_CHECK(output.dim() == 2 && (gate.dim() == 2 || gate.dim() == 3),
              "output must be 2D and gate must be 2D or 3D");
  TORCH_CHECK(gate.is_cuda() && gate.stride(-1) == 1,
              "gate must be a CUDA tensor contiguous in its last dimension");
  TORCH_CHECK(output.device() == gate.device(), "output and gate must be on the same device");
  TORCH_CHECK(output.scalar_type() == gate.scalar_type(), "output and gate must have the same dtype");

  const uint32_t num_heads = gate.dim() == 3 ? gate.size(1) : 1;
  const uint32_t head_dim = gate.size(-1);
  TORCH_CHECK(output.size(0) == gate.size(0) &&
              output.size(1) == static_cast<int64_t>(num_heads) * head_dim,
              "output and gate shapes do not match");

  const uint64_t num_elements = output.numel();
  if (num_elements == 0) {
    return;
  }

  const int64_t gate_token_stride = gate.stride(0);
  const int64_t gate_head_stride = gate.dim() == 3 ? gate.stride(1) : 0;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(output));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(output.scalar_type(), c_type, [&] {
#ifndef USE_ROCM
    auto launch = [&](auto vector_size) {
      constexpr uint32_t vec_size = decltype(vector_size)::value;
      const uint32_t vectors_per_head = head_dim / vec_size;
      const uint32_t threads_x = std::min(vectors_per_head, 1024U);
      const dim3 block(threads_x, std::min(num_heads, 1024U / threads_x));
      const dim3 grid(output.size(0), (num_heads + block.y - 1) / block.y);
      fused_sigmoid_mul_kernel<c_type, vec_size>
          <<<grid, block, 0, stream>>>(
              static_cast<c_type*>(output.data_ptr()),
              static_cast<const c_type*>(gate.data_ptr()),
              num_heads,
              head_dim,
              gate_token_stride,
              gate_head_stride);
    };
    // Use the widest safe vector for both pointers and every row/head start.
    const uintptr_t alignment = reinterpret_cast<uintptr_t>(output.data_ptr()) |
        reinterpret_cast<uintptr_t>(gate.data_ptr()) |
        (head_dim * sizeof(c_type)) |
        (gate_token_stride * sizeof(c_type)) |
        (gate_head_stride * sizeof(c_type));
    if (alignment % 16 == 0) {
      launch(std::integral_constant<uint32_t, 16 / sizeof(c_type)>{});
    } else if (alignment % 8 == 0) {
      launch(std::integral_constant<uint32_t, 8 / sizeof(c_type)>{});
    } else if (alignment % 4 == 0) {
      launch(std::integral_constant<uint32_t, 4 / sizeof(c_type)>{});
    } else {
      launch(std::integral_constant<uint32_t, 1>{});
    }
    return true;
#else
    TORCH_CHECK(false, "fused_sigmoid_mul is only available on CUDA");
#endif
  });
}

void gelu_tanh_and_mul(at::Tensor& out, at::Tensor& input) {
  int d = input.size(-1) / 2;
  int64_t num_tokens = input.numel() / input.size(-1);
  dim3 grid(num_tokens);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(input.scalar_type(), c_type, [&] {
    uint32_t vec_size = 16 / sizeof(c_type);
    dim3 block(std::min(d / vec_size, 1024U));
    launch_gelu_tanh_kernel<c_type>(out, input, d, grid, block, stream);
    return true;
  });
}

void gelu_and_mul(at::Tensor& out, at::Tensor& input) {
  int d = input.size(-1) / 2;
  int64_t num_tokens = input.numel() / input.size(-1);
  dim3 grid(num_tokens);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(input.scalar_type(), c_type, [&] {
    uint32_t vec_size = 16 / sizeof(c_type);
    dim3 block(std::min(d / vec_size, 1024U));
    launch_gelu_kernel<c_type>(out, input, d, grid, block, stream);
    return true;
  });
}

#if USE_ROCM
void gelu_quick(at::Tensor& out, const at::Tensor& input) {
  int d = input.size(-1);
  int64_t num_tokens = input.numel() / input.size(-1);
  dim3 grid(num_tokens);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(input.scalar_type(), c_type, [&] {
    uint32_t vec_size = 16 / sizeof(c_type);
    dim3 block(std::min(d / vec_size, 1024U));
    sgl_hip::activation::act_only_kernel<c_type, gelu_quick_act>
        <<<grid, block, 0, stream>>>(static_cast<c_type*>(out.data_ptr()), static_cast<c_type*>(input.data_ptr()), d);

    return true;
  });
}
#endif
