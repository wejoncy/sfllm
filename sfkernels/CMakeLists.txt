cmake_minimum_required(VERSION 3.20)

project(sf_kernel_common_ops LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Prefer a fast incremental generator (setup.py will pass -G Ninja when available).
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Configurable knobs (passed from setup.py)
set(GPU_TARGET "86" CACHE STRING "CUDA SM architecture, e.g. 70/80/86/90")
set(OPERATOR_NAMESPACE "sfkernels" CACHE STRING "Namespace used by TORCH_LIBRARY_FRAGMENT")

find_package(Python3 REQUIRED COMPONENTS Development.Module)
find_package(Torch REQUIRED)

set(SFKERNELS_INCLUDE_DIRS
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc"
  "${CMAKE_CURRENT_SOURCE_DIR}/third_patry/flashinfer/include"
)

set(SFKERNELS_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/common_extension.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/spec/eagle_utils.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/elementwise/activation.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/elementwise/layernorm.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/elementwise/rope.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/quant/per_token_quant_fp8.cu"
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/quant/per_tensor_quant_fp8.cu"
)

add_library(common_ops MODULE ${SFKERNELS_SOURCES})

target_include_directories(common_ops PRIVATE ${SFKERNELS_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS})

# Match torch.utils.cpp_extension defaults (important: your code relies on these)
target_compile_definitions(common_ops PRIVATE
  TORCH_API_INCLUDE_EXTENSION_H
  TORCH_EXTENSION_NAME=common_ops
  NDEBUG
  OPERATOR_NAMESPACE=${OPERATOR_NAMESPACE}
  ENABLE_BF16
  FLASHINFER_ENABLE_F16
  FLASHINFER_ENABLE_BF16
  __CUDA_NO_HALF_OPERATORS__
  __CUDA_NO_HALF_CONVERSIONS__
  __CUDA_NO_BFLOAT16_CONVERSIONS__
  __CUDA_NO_HALF2_OPERATORS__
)

# Reasonable defaults; users can override via CMAKE_CUDA_FLAGS/ENV
target_compile_options(common_ops PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

target_link_libraries(common_ops PRIVATE ${TORCH_LIBRARIES} Python3::Module)

# Build a Python extension module: no lib prefix (common_ops.pyd / common_ops.so)
set_target_properties(common_ops PROPERTIES PREFIX "" OUTPUT_NAME "common_ops")

# On Windows, ensure we get a .pyd suffix.
if (WIN32)
  set_target_properties(common_ops PROPERTIES SUFFIX ".pyd")
endif()
