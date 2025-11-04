#pragma once

#include <torch/torch.h>
#include <stdexcept>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA/ROCm tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define CHECK_DIM(expected_dim, x) \
    TORCH_CHECK(x.dim() == expected_dim, #x " must have " #expected_dim " dimensions, got ", x.dim())

#define CHECK_EQ(a, b) \
    TORCH_CHECK(a == b, "Expected " #a " == " #b ", got ", a, " != ", b)