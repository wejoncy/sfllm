#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cub/cub.cuh>
#include "utils.h"
#include "cast.cuh"

#define ALIGN_BYTES 16

struct SumOp {
    __device__ __forceinline__ float operator()(const float &a, const float &b) const {
        return a + b;
    }
};

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

template <int ILP, typename scalar_t>
__global__ void rms_norm_kernel_opt_v2(
    scalar_t* __restrict__ out,           // [..., hidden_size]
    const scalar_t* __restrict__ input,   // [..., hidden_size]
    scalar_t* __restrict__ input_res,   // [..., hidden_size]
    const scalar_t* __restrict__ weight,  // [hidden_size]
    const float epsilon, const int num_tokens, const int hidden_size) {

    using LoadT = aligned_vector<scalar_t, ILP>;
    scalar_t v[ILP];
    scalar_t v_res[ILP];
    LoadT* value = reinterpret_cast<LoadT*>(&v);
    LoadT* value_res = reinterpret_cast<LoadT*>(&v_res);
    __shared__ float s_variance;
    extern __shared__ char shared_mem[];
    using BlockReduce = cub::BlockReduce<float, 1024>;
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

    int block_work_niter = (num_tokens +  gridDim.x - 1) / gridDim.x;

    for(int work_iter=0; work_iter<block_work_niter; work_iter++) {
        int batch_idx = work_iter * gridDim.x + blockIdx.x;

        if(batch_idx < num_tokens) {

            float variance = 0.0f;

            const scalar_t* input_for_this = input + batch_idx * hidden_size;
            scalar_t* input_res_for_this = input_res==nullptr? nullptr:input_res + batch_idx * hidden_size;
            scalar_t* out_for_this = out + batch_idx * hidden_size;
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
                for(int j = 0; j < ILP; j++) {
                    float x = to_float(v[j]);
                    if (input_res_for_this_vec != nullptr) {
                        x += to_float(v_res[j]);
                    }
                    float w = to_float(weight_cache[idx * ILP + j]);
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
            double eps,at::optional<at::Tensor> input_2=at::nullopt) {
    int hidden_size = input.size(-1);
    int num_tokens = input.numel() / hidden_size;
    
    CHECK_INPUT(input);
    if (input_2.has_value()) {
        CHECK_INPUT(input_2.value());
    }
    CHECK_INPUT(weight);
    CHECK_INPUT(output);
    
    dim3 grid(std::min(num_tokens, 1024));
    dim3 block(1024);
    
    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input.scalar_type(), scalar_t, [&] {
        int shared_mem_size = hidden_size * sizeof(scalar_t) + ALIGN_BYTES;
        constexpr int ILP = 16 / sizeof(scalar_t); 
        TORCH_CHECK(hidden_size % ILP == 0);

        rms_norm_kernel_opt_v2<ILP, scalar_t><<<grid, block, shared_mem_size, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<scalar_t*>(output.data_ptr()),
            reinterpret_cast<scalar_t*>(input.data_ptr()),
            input_2.has_value() ? reinterpret_cast<scalar_t*>(input_2->data_ptr()) : nullptr,
            reinterpret_cast<scalar_t*>(weight.data_ptr()),
            (float)eps,
            num_tokens,
            hidden_size
        );
        return true;
    });
}
