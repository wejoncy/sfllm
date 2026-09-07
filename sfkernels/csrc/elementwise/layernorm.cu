#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cub/cub.cuh>
#include "utils.h"
#include "cast.cuh"

#define ALIGN_BYTES 16

struct SumOp {
    __device__ __forceinline__ float operator()(const float &a, const float &b) const {
        return a + b;
    }
};

constexpr int kGatedRmsNormThreads = 128;

template <typename scalar_t>
__global__ void gated_rms_norm_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ gate,
    const float* __restrict__ weight,
    const float epsilon,
    const int num_heads,
    const int hidden_size,
    const int64_t input_token_stride,
    const int64_t input_head_stride,
    const int64_t gate_token_stride,
    const int64_t gate_head_stride) {
    using BlockReduce = cub::BlockReduce<float, kGatedRmsNormThreads>;
    __shared__ typename BlockReduce::TempStorage reduce_storage;
    __shared__ float inverse_rms;

    const int64_t row = blockIdx.x;
    const int64_t token = row / num_heads;
    const int head = row % num_heads;
    const int64_t input_offset =
        token * input_token_stride + head * input_head_stride;
    float sum = 0.0f;
    for (int column = threadIdx.x; column < hidden_size; column += blockDim.x) {
        const float value = to_float(input[input_offset + column]);
        sum += value * value;
    }
    const float square_sum = BlockReduce(reduce_storage).Reduce(
        sum, SumOp{});
    if (threadIdx.x == 0) {
        inverse_rms = rsqrtf(square_sum / hidden_size + epsilon);
    }
    __syncthreads();

    for (int column = threadIdx.x; column < hidden_size; column += blockDim.x) {
        const float value = to_float(input[input_offset + column]);
        const int64_t gate_offset =
            token * gate_token_stride + head * gate_head_stride + column;
        const float gate_value = to_float(gate[gate_offset]);
        const float gated = gate_value / (1.0f + expf(-gate_value));
        output[row * hidden_size + column] = from_float<scalar_t>(
            value * inverse_rms * weight[column] * gated);
    }
}

void gated_rmsnorm(
    at::Tensor& output,
    const at::Tensor& input,
    const at::Tensor& gate,
    const at::Tensor& weight,
    double eps) {
    CHECK_LAST_DIM_CONTIGUOUS_INPUT(input);
    CHECK_LAST_DIM_CONTIGUOUS_INPUT(gate);
    CHECK_INPUT(weight);
    CHECK_INPUT(output);
    TORCH_CHECK(input.dim() == 3, "input must have shape [tokens, heads, dim]");
    TORCH_CHECK(gate.sizes() == input.sizes(), "gate shape must match input");
    TORCH_CHECK(output.sizes() == input.sizes(), "output shape must match input");
    TORCH_CHECK(gate.scalar_type() == input.scalar_type(),
                "gate and input must have the same dtype");
    TORCH_CHECK(output.scalar_type() == input.scalar_type(),
                "output and input must have the same dtype");
    TORCH_CHECK(input.device() == gate.device() && input.device() == output.device() &&
                input.device() == weight.device(), "all tensors must be on the same device");
    TORCH_CHECK(weight.scalar_type() == at::ScalarType::Float,
                "weight must have float32 dtype");
    const int num_heads = input.size(1);
    const int hidden_size = input.size(2);
    TORCH_CHECK(weight.dim() == 1 && weight.numel() == hidden_size,
                "weight must have shape [", hidden_size, "]");
    TORCH_CHECK(hidden_size > 0, "hidden size must be positive");

    const int64_t num_rows = input.size(0) * num_heads;
    if (num_rows == 0) {
        return;
    }
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input.scalar_type(), scalar_t, [&] {
        gated_rms_norm_kernel<scalar_t>
            <<<static_cast<uint32_t>(num_rows), kGatedRmsNormThreads, 0, stream>>>(
                reinterpret_cast<scalar_t*>(output.data_ptr()),
                reinterpret_cast<const scalar_t*>(input.data_ptr()),
                reinterpret_cast<const scalar_t*>(gate.data_ptr()),
                weight.data_ptr<float>(),
                static_cast<float>(eps),
                num_heads,
                hidden_size,
                input.stride(0),
                input.stride(1),
                gate.stride(0),
                gate.stride(1));
        return true;
    });
}

template<typename T>
__device__ __forceinline__ T mul(T a, T b) {
    return a * b;
}

template<>
__device__ __forceinline__ __half mul(__half a, __half b) {
    return __hmul(a, b);
}

template<>
__device__ __forceinline__ __nv_bfloat16 mul(__nv_bfloat16 a, __nv_bfloat16 b) {
    return __hmul(a, b);
}

template<typename T, int N>
struct alignas(sizeof(T) * N) aligned_vector {
    T val[N];
    
    __device__ __host__ T& operator[](int i) {
        return val[i];
    }
    
    __device__ __host__ const T& operator[](int i) const {
        return val[i];
    }
};

template <int ILP, int BLOCK_SIZE, typename scalar_t>
__global__ void rms_norm_kernel_opt_v2(
    scalar_t* __restrict__ out,           // [..., hidden_size]
    const scalar_t* __restrict__ input,   // [..., hidden_size]
    scalar_t* __restrict__ input_res,   // [..., hidden_size]
    const scalar_t* __restrict__ weight,  // [hidden_size]
    const float epsilon, const int num_tokens, const int hidden_size,
    const int rows_per_outer, const int64_t outer_stride,
    const int64_t row_stride, const bool gemma_style) {

    using LoadT = aligned_vector<scalar_t, ILP>;
    scalar_t v[ILP];
    scalar_t v_res[ILP];
    LoadT* value = reinterpret_cast<LoadT*>(&v);
    LoadT* value_res = reinterpret_cast<LoadT*>(&v_res);
    __shared__ float s_variance;
    extern __shared__ char shared_mem[];
    using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage reduceStore;
    int shift = ((uint64_t)shared_mem) % ALIGN_BYTES;
    int shift_reverse = shift == 0 ? 0 : ALIGN_BYTES - shift;
    scalar_t* weight_cache = reinterpret_cast<scalar_t* >(shared_mem + shift_reverse);

    const LoadT* weight_vec = reinterpret_cast<const LoadT*>(weight);
    LoadT* weight_cache_vec = reinterpret_cast<LoadT*>(weight_cache);
    
    for (int idx = threadIdx.x; idx < hidden_size / ILP; idx += blockDim.x) {
        *value = weight_vec[idx];
        weight_cache_vec[idx] = *value;
    }
    __syncthreads();

    int block_work_niter = (num_tokens + gridDim.x - 1) / gridDim.x;

    for(int work_iter=0; work_iter<block_work_niter; work_iter++) {
        int batch_idx = work_iter * gridDim.x + blockIdx.x;

        if(batch_idx < num_tokens) {
            float variance = 0.0f;

            const scalar_t* input_for_this = input
                + static_cast<int64_t>(batch_idx / rows_per_outer) * outer_stride
                + static_cast<int64_t>(batch_idx % rows_per_outer) * row_stride;
            scalar_t* input_res_for_this = input_res == nullptr
                ? nullptr
                : input_res + static_cast<int64_t>(batch_idx) * hidden_size;
            scalar_t* out_for_this = out + static_cast<int64_t>(batch_idx) * hidden_size;
            const LoadT* input_for_this_vec = reinterpret_cast<const LoadT*>(input_for_this);
            LoadT* input_res_for_this_vec = input_res_for_this==nullptr? nullptr: reinterpret_cast<LoadT*>(input_res_for_this);
            LoadT* out_for_this_vec = reinterpret_cast<LoadT*>(out_for_this);
            
            for (int idx = threadIdx.x; idx < hidden_size / ILP; idx += blockDim.x) {
                *value = input_for_this_vec[idx];
                if (input_res_for_this_vec != nullptr) {
                    *value_res = input_res_for_this_vec[idx];
                }
                for(int j = 0; j < ILP; j++) {
                    float x = to_float(v[j]);
                    if (input_res_for_this_vec != nullptr) {
                        float r = to_float(v_res[j]);
                        x += r;
                    }
                    variance += x * x;
                }    
            }


            variance = BlockReduce(reduceStore).Reduce(variance, SumOp{}, blockDim.x);

            if (threadIdx.x == 0) {
                s_variance = rsqrtf(variance / hidden_size + epsilon);
            }
            __syncthreads();

            for (int idx = threadIdx.x; idx < hidden_size / ILP; idx += blockDim.x) {
                *value = input_for_this_vec[idx];
                if (input_res_for_this_vec != nullptr) {
                    *value_res = input_res_for_this_vec[idx];
                }
                const LoadT weights = weight_cache_vec[idx];
                for(int j = 0; j < ILP; j++) {
                    float x = to_float(v[j]);
                    if (input_res_for_this_vec != nullptr) {
                        x += to_float(v_res[j]);
                    }
                    float w = to_float(weights[j])
                        + static_cast<float>(gemma_style);
                    v[j] = from_float<scalar_t>(x * s_variance * w);
                    if (input_res_for_this_vec != nullptr) {
                        v_res[j] = from_float<scalar_t>(x);
                    }
                }
                out_for_this_vec[idx] = *value;
                if (input_res_for_this_vec != nullptr) {
                    input_res_for_this_vec[idx] = *value_res;
                }
            }
        }
        __syncthreads();
    }
}

void rmsnorm(at::Tensor& output, at::Tensor& input, at::Tensor& weight, 
            double eps,at::optional<at::Tensor> input_2=at::nullopt,
            bool gemma_style=false) {
    TORCH_CHECK(input.dim() > 0, "input must have at least one dimension");
    int hidden_size = input.size(-1);
    int num_tokens = input.numel() / hidden_size;
    
    CHECK_LAST_DIM_CONTIGUOUS_INPUT(input);
    if (input_2.has_value()) {
        CHECK_INPUT(input_2.value());
    }
    CHECK_INPUT(weight);
    CHECK_INPUT(output);
    TORCH_CHECK(output.sizes() == input.sizes(), "output shape must match input");
    TORCH_CHECK(weight.dim() == 1 && weight.numel() == hidden_size,
                "weight must have shape [", hidden_size, "]");
    TORCH_CHECK(output.scalar_type() == input.scalar_type(),
                "output and input must have the same dtype");
    TORCH_CHECK(weight.scalar_type() == input.scalar_type(),
                "weight and input must have the same dtype");
    TORCH_CHECK(output.device() == input.device() && weight.device() == input.device(),
                "all tensors must be on the same device");
    if (input_2.has_value()) {
        TORCH_CHECK(input_2->sizes() == input.sizes(),
                    "residual shape must match input");
        TORCH_CHECK(input_2->scalar_type() == input.scalar_type(),
                    "residual and input must have the same dtype");
        TORCH_CHECK(input_2->device() == input.device(),
                    "residual and input must be on the same device");
    }

    TORCH_CHECK(input.is_contiguous() || input.dim() == 2 || input.dim() == 3,
                "strided rmsnorm input must be 2D or 3D");
    const int rows_per_outer = input.dim() == 3 ? input.size(1) : num_tokens;
    const int64_t outer_stride = input.dim() == 3 ? input.stride(0) : 0;
    const int64_t row_stride = input.dim() == 3 ? input.stride(1) :
        (input.dim() == 1 ? hidden_size : input.stride(-2));
    
    if (num_tokens == 0) {
        return;
    }
    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input.scalar_type(), scalar_t, [&] {
        int shared_mem_size = hidden_size * sizeof(scalar_t) + ALIGN_BYTES;
        constexpr int ILP = 16 / sizeof(scalar_t); 
        TORCH_CHECK(hidden_size % ILP == 0);
        TORCH_CHECK(reinterpret_cast<uintptr_t>(input.data_ptr()) % ALIGN_BYTES == 0 &&
                    outer_stride % ILP == 0 && row_stride % ILP == 0,
                    "rmsnorm rows must be 16-byte aligned");

        auto* output_ptr = reinterpret_cast<scalar_t*>(output.data_ptr());
        auto* input_ptr = reinterpret_cast<scalar_t*>(input.data_ptr());
        auto* residual_ptr = input_2.has_value()
            ? reinterpret_cast<scalar_t*>(input_2->data_ptr())
            : nullptr;
        auto* weight_ptr = reinterpret_cast<scalar_t*>(weight.data_ptr());
        const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        dim3 grid(std::min(num_tokens, 1024));
        rms_norm_kernel_opt_v2<ILP, 256, scalar_t>
            <<<grid, 256, shared_mem_size, stream>>>(
                output_ptr, input_ptr, residual_ptr, weight_ptr,
                static_cast<float>(eps), num_tokens, hidden_size,
                rows_per_outer, outer_stride, row_stride, gemma_style);
        return true;
    });
}
