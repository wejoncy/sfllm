#include "utils.h"

#ifndef USE_ROCM
#include <cuda_fp8.h>
#endif


template<typename T>
__device__ __forceinline__ float to_float(T val);

template<>
__device__ __forceinline__ float to_float(__half val) {
    return __half2float(val);
}

template<>
__device__ __forceinline__ float to_float(__nv_bfloat16 val) {
    return __bfloat162float(val);
}

template<>
__device__ __forceinline__ float to_float(float val) {
    return val;
}

#if !defined(USE_ROCM)
template<>
__device__ __forceinline__ float to_float(__nv_fp8_e4m3 val) {
    return float(val);
}
#endif

template<typename T>
__device__ __forceinline__ T from_float(float val);

template<>
__device__ __forceinline__ __half from_float(float val) {
    return __float2half(val);
}

template<>
__device__ __forceinline__ __nv_bfloat16 from_float(float val) {
    return __float2bfloat16(val);
}

template<>
__device__ __forceinline__ float from_float(float val) {
    return val;
}

#if !defined(USE_ROCM)
template<>
__device__ __forceinline__ __nv_fp8_e4m3 from_float(float val) {
    return __nv_fp8_e4m3(val);
}
#endif