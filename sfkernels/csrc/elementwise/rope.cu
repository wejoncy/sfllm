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

#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/all.h>

#include "pos_enc.cuh"
#include "utils.h"

using namespace flashinfer;

namespace {

void apply_rope_pos_ids_cos_sin_cache_impl(
    at::Tensor q,
    at::Tensor k,
    at::Tensor q_rope,
    at::Tensor k_rope,
    at::Tensor cos_sin_cache,
    at::Tensor pos_ids,
    bool interleave,
    bool enable_pdl,
    const std::optional<at::Tensor>& v,
    const std::optional<at::Tensor>& k_buffer,
    const std::optional<at::Tensor>& v_buffer,
    const std::optional<at::Tensor>& kv_cache_loc,
    const at::Tensor* q_norm_weight,
    const at::Tensor* k_norm_weight,
    double qk_norm_epsilon,
    double qk_norm_weight_bias) {
  CHECK_LAST_DIM_CONTIGUOUS(q);
  CHECK_LAST_DIM_CONTIGUOUS(k);
  CHECK_LAST_DIM_CONTIGUOUS(q_rope);
  CHECK_LAST_DIM_CONTIGUOUS(k_rope);

  const bool save_kv_cache = v.has_value();
  if (save_kv_cache) {
    TORCH_CHECK(v.has_value());
    TORCH_CHECK(k_buffer.has_value());
    TORCH_CHECK(v_buffer.has_value());
    TORCH_CHECK(kv_cache_loc.has_value());
    CHECK_LAST_DIM_CONTIGUOUS(v.value());
    CHECK_LAST_DIM_CONTIGUOUS(k_buffer.value());
    CHECK_LAST_DIM_CONTIGUOUS(v_buffer.value());
    CHECK_DIM(3, k_buffer.value());      // k_buffer: (nnz, H_K, D)
    CHECK_DIM(3, v_buffer.value());      // v_buffer: (nnz, H_V, D)
    CHECK_DIM(3, v.value());             // v: (nnz, H_V, D)
    CHECK_DIM(1, kv_cache_loc.value());  // v: (n)
    CHECK_INPUT(kv_cache_loc.value());
  }
  size_t k_buffer_stride_n = save_kv_cache ? k_buffer->stride(0) : 0;
  size_t k_buffer_stride_h = save_kv_cache ? k_buffer->stride(1) : 0;
  size_t v_buffer_stride_n = save_kv_cache ? v_buffer->stride(0) : 0;
  size_t v_buffer_stride_h = save_kv_cache ? v_buffer->stride(1) : 0;
  size_t v_stride_n = save_kv_cache ? v->stride(0) : 0;
  size_t v_stride_h = save_kv_cache ? v->stride(1) : 0;
  auto kv_cache_loc_ptr = save_kv_cache ? static_cast<int64_t*>(kv_cache_loc->data_ptr()) : nullptr;

  CHECK_INPUT(cos_sin_cache);
  CHECK_INPUT(pos_ids);
  auto device = q.device();
  CHECK_EQ(k.device(), device);
  CHECK_EQ(q_rope.device(), device);
  CHECK_EQ(k_rope.device(), device);
  CHECK_EQ(cos_sin_cache.device(), device);
  CHECK_EQ(pos_ids.device(), device);
  CHECK_DIM(3, q);  // q: (nnz, H_Q, D)
  CHECK_DIM(3, k);  // k: (nnz, H_K, D)

  // cos_sin_cache: (max_seq_len, R)
  // First half of R is cos, second half is sin
  CHECK_DIM(2, cos_sin_cache);
  CHECK_EQ(q.size(0), k.size(0));
  CHECK_EQ(q.size(2), k.size(2));
  CHECK_EQ(q.sizes(), q_rope.sizes());
  CHECK_EQ(k.sizes(), k_rope.sizes());
  CHECK_EQ(q.scalar_type(), k.scalar_type());
  CHECK_EQ(q.scalar_type(), q_rope.scalar_type());
  CHECK_EQ(q.scalar_type(), k_rope.scalar_type());
  unsigned int rotary_dim = cos_sin_cache.size(1);
  unsigned int num_qo_heads = q.size(1);
  unsigned int num_kv_heads = k.size(1);
  unsigned int head_dim = q.size(2);
  unsigned int nnz = q.size(0);
  const bool apply_qk_norm = q_norm_weight != nullptr;
  if (apply_qk_norm) {
    const at::Tensor& q_weight = *q_norm_weight;
    const at::Tensor& k_weight = *k_norm_weight;
    CHECK_INPUT(q_weight);
    CHECK_INPUT(k_weight);
    CHECK_EQ(q_weight.scalar_type(), q.scalar_type());
    CHECK_EQ(k_weight.scalar_type(), q.scalar_type());
    CHECK_EQ(q_weight.numel(), head_dim);
    CHECK_EQ(k_weight.numel(), head_dim);
    CHECK_GE(head_dim, rotary_dim);
    const unsigned int vec_size = std::max<unsigned int>(
        16 / q.element_size(), head_dim / 32);
    CHECK_EQ(rotary_dim % vec_size, 0);
    if (!interleave) {
      const unsigned int half_rotary_threads = rotary_dim / vec_size / 2;
      TORCH_CHECK(
          half_rotary_threads > 0 &&
              (half_rotary_threads & (half_rotary_threads - 1)) == 0,
          "NeoX partial rotary dimension must map to a power-of-two half warp");
    }
  }
  size_t q_stride_n = q.stride(0);
  size_t q_stride_h = q.stride(1);
  size_t k_stride_n = k.stride(0);
  size_t k_stride_h = k.stride(1);

  size_t q_rope_stride_n = q_rope.stride(0);
  size_t q_rope_stride_h = q_rope.stride(1);
  size_t k_rope_stride_n = k_rope.stride(0);
  size_t k_rope_stride_h = k_rope.stride(1);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FLOAT_FP16(q.scalar_type(), c_type, [&] {
    if (save_kv_cache || apply_qk_norm) {
      cudaError_t status = BatchQKApplyRotaryPosIdsCosSinCacheEnhanced(
          static_cast<c_type*>(q.data_ptr()),
          static_cast<c_type*>(k.data_ptr()),
          save_kv_cache ? static_cast<c_type*>(v->data_ptr()) : nullptr,
          apply_qk_norm ? static_cast<c_type*>(q_norm_weight->data_ptr()) : nullptr,
          apply_qk_norm ? static_cast<c_type*>(k_norm_weight->data_ptr()) : nullptr,
          static_cast<float>(qk_norm_epsilon),
          static_cast<float>(qk_norm_weight_bias),
          static_cast<c_type*>(q_rope.data_ptr()),
          static_cast<c_type*>(k_rope.data_ptr()),
          save_kv_cache ? static_cast<c_type*>(k_buffer->data_ptr()) : nullptr,
          save_kv_cache ? static_cast<c_type*>(v_buffer->data_ptr()) : nullptr,
          static_cast<float*>(cos_sin_cache.data_ptr()),
          static_cast<int64_t*>(pos_ids.data_ptr()),
          nnz,
          num_qo_heads,
          num_kv_heads,
          rotary_dim,
          head_dim,
          q_stride_n,
          q_stride_h,
          k_stride_n,
          k_stride_h,
          v_stride_n,
          v_stride_h,
          q_rope_stride_n,
          q_rope_stride_h,
          k_rope_stride_n,
          k_rope_stride_h,
          k_buffer_stride_n,
          k_buffer_stride_h,
          v_buffer_stride_n,
          v_buffer_stride_h,
          kv_cache_loc_ptr,
          interleave,
          save_kv_cache,
          apply_qk_norm,
          enable_pdl,
          stream);
      TORCH_CHECK(
          status == cudaSuccess,
          "BatchQKApplyRotaryPosIdsCosSinCacheEnhanced failed with error code " +
              std::string(cudaGetErrorString(status)));
    } else {
      TORCH_CHECK(!enable_pdl);
      cudaError_t status = BatchQKApplyRotaryPosIdsCosSinCache(
          static_cast<c_type*>(q.data_ptr()),
          static_cast<c_type*>(k.data_ptr()),
          static_cast<c_type*>(q_rope.data_ptr()),
          static_cast<c_type*>(k_rope.data_ptr()),
          static_cast<float*>(cos_sin_cache.data_ptr()),
          static_cast<int64_t*>(pos_ids.data_ptr()),
          nnz,
          num_qo_heads,
          num_kv_heads,
          rotary_dim,
          head_dim,
          q_stride_n,
          q_stride_h,
          k_stride_n,
          k_stride_h,
          q_rope_stride_n,
          q_rope_stride_h,
          k_rope_stride_n,
          k_rope_stride_h,
          interleave,
          stream);
      TORCH_CHECK(
          status == cudaSuccess,
          "BatchQKApplyRotaryPosIdsCosSinCache failed with error code " + std::string(cudaGetErrorString(status)));
    }
    return true;
  });
}

}  // namespace

void apply_rope_pos_ids_cos_sin_cache(
    at::Tensor q,
    at::Tensor k,
    at::Tensor q_rope,
    at::Tensor k_rope,
    at::Tensor cos_sin_cache,
    at::Tensor pos_ids,
    bool interleave,
    bool enable_pdl,
    const std::optional<at::Tensor>& v,
    const std::optional<at::Tensor>& k_buffer,
    const std::optional<at::Tensor>& v_buffer,
    const std::optional<at::Tensor>& kv_cache_loc) {
  apply_rope_pos_ids_cos_sin_cache_impl(
      q, k, q_rope, k_rope, cos_sin_cache, pos_ids, interleave, enable_pdl,
      v, k_buffer, v_buffer, kv_cache_loc, nullptr, nullptr, 0.0, 0.0);
}

void qk_norm_rope_and_cache(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor q_norm_weight,
    at::Tensor k_norm_weight,
    at::Tensor cos_sin_cache,
    at::Tensor pos_ids,
    bool interleave,
    at::Tensor k_buffer,
    at::Tensor v_buffer,
    at::Tensor kv_cache_loc,
    double epsilon) {
  apply_rope_pos_ids_cos_sin_cache_impl(
      q, k, q, k, cos_sin_cache, pos_ids, interleave, false,
      v, k_buffer, v_buffer, kv_cache_loc,
      &q_norm_weight, &k_norm_weight, epsilon, 0.0);
}

void gemma_qk_norm_rope(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor q_rope,
    at::Tensor k_rope,
    at::Tensor q_norm_weight,
    at::Tensor k_norm_weight,
    at::Tensor cos_sin_cache,
    at::Tensor pos_ids,
    const std::optional<at::Tensor>& k_buffer,
    const std::optional<at::Tensor>& v_buffer,
    const std::optional<at::Tensor>& kv_cache_loc,
    double epsilon) {
  const bool save_kv_cache = k_buffer.has_value();
  TORCH_CHECK(
      save_kv_cache == v_buffer.has_value() &&
          save_kv_cache == kv_cache_loc.has_value(),
      "K cache, V cache, and cache locations must be provided together");
  const std::optional<at::Tensor> value =
      save_kv_cache ? std::make_optional(v) : std::nullopt;
  apply_rope_pos_ids_cos_sin_cache_impl(
      q, k, q_rope, k_rope, cos_sin_cache, pos_ids, false, false,
      value, k_buffer, v_buffer, kv_cache_loc,
      &q_norm_weight, &k_norm_weight, epsilon, 1.0);
}
