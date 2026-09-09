# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Fused operators for normalization layers."""

import logging
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import sf_kernel

try:
    import flashinfer.norm as _flashinfer_norm
except ImportError:
    _flashinfer_norm = None

from sfllm.layers.op_base import CustomOp
from sfllm.layers.quantization.fp8_kernel import gemma_rmsnorm_quant_fp8
logger = logging.getLogger(__name__)

class RMSNorm(CustomOp):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        var_hidden_size: Optional[int] = None,
        cast_x_before_out_mul: bool = False,
        fp32_residual: bool = False,
        weight_dtype: Optional = None,
        override_orig_dtype: Optional = None,
    ) -> None:
        super().__init__()
        self.cast_x_before_out_mul = cast_x_before_out_mul
        self.fp32_residual = fp32_residual
        self.override_orig_dtype = override_orig_dtype
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=weight_dtype))
        self.variance_epsilon = eps
        self.hidden_size = hidden_size
        self.variance_size_override = (
            None if var_hidden_size == hidden_size else var_hidden_size
        )

    def forward_native(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if not x.is_contiguous():
            x = x.contiguous()
        orig_dtype = self.override_orig_dtype or x.dtype
        x = x.to(torch.float32)
        if residual is not None:
            x = x + residual.to(torch.float32)
            if self.fp32_residual:
                residual = x.clone()
            else:
                residual = x.to(orig_dtype)

        hidden_size = x.shape[-1]
        if hidden_size != self.hidden_size:
            raise ValueError(
                "Expected hidden_size to be "
                f"{self.hidden_size}, but found: {hidden_size}"
            )

        if self.variance_size_override is None:
            x_var = x
        else:
            if hidden_size < self.variance_size_override:
                raise ValueError(
                    "Expected hidden_size to be at least "
                    f"{self.variance_size_override}, but found: {hidden_size}"
                )

            x_var = x[..., : self.variance_size_override]

        variance = x_var.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)

        if self.cast_x_before_out_mul:
            x = self.weight * x.to(orig_dtype)
        else:
            x = (x * self.weight).to(orig_dtype)

        if residual is None:
            return x
        else:
            return x, residual
    
    def forward_cuda(self, x: torch.Tensor, residual: Optional[torch.Tensor] = None):
        if (
            self.variance_size_override is not None
            or self.cast_x_before_out_mul
            or self.weight.dtype != x.dtype
            or self.override_orig_dtype not in (None, x.dtype)
            or (
                residual is not None
                and (residual.dtype != x.dtype or self.fp32_residual)
            )
        ):
            return self.forward_native(x, residual)

        if _flashinfer_norm is not None:
            if residual is None:
                return _flashinfer_norm.rmsnorm(
                    x, self.weight, self.variance_epsilon
                )
            _flashinfer_norm.fused_add_rmsnorm(
                x, residual, self.weight, self.variance_epsilon
            )
            return x, residual

        out = torch.empty_like(x)
        sf_kernel.rmsnorm(
            out,
            x,
            self.weight,
            self.variance_epsilon,
            residual,
        )
        return out if residual is None else (out, residual)


class GemmaRMSNorm(CustomOp):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        quant_methods: tuple = (),
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size), requires_grad=False)
        self.variance_epsilon = eps
        self.quant_methods = quant_methods
        self.output_dtypes = tuple(method.input_dtype for method in quant_methods)
        self.fp8_output = any(dtype is not None for dtype in self.output_dtypes)

    def forward_native(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        orig_dtype = x.dtype
        if residual is not None:
            x = x + residual
            residual = x

        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = x * (1.0 + self.weight.float())
        x = x.to(orig_dtype)
        if len(self.quant_methods) > 1:
            x = (x,) * len(self.quant_methods)
        return x if residual is None else (x, residual)

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Both CUDA paths keep normalization and (1 + weight) in FP32.
        # Residual addition is fused into the same kernel.
        if self.fp8_output:
            output = gemma_rmsnorm_quant_fp8(
                x, weight=self.weight, residual=residual,
                eps=self.variance_epsilon,
                output_dtypes=self.output_dtypes,
                input_scales=tuple(method.input_scale for method in self.quant_methods),
            )
            return output if residual is None else (output, residual)
        if (
            _flashinfer_norm is None or self.weight.dtype != x.dtype
            or x.stride(-1) != 1 or self.weight.stride(0) != 1
            or (residual is not None and (
                residual.dtype != x.dtype or residual.stride(-1) != 1
            ))
        ):
            out = gemma_rmsnorm_quant_fp8(
                x, self.weight, residual, self.variance_epsilon,
                output_dtypes=(None,),
            )
        elif residual is None:
            out = _flashinfer_norm.gemma_rmsnorm(
                x, self.weight, self.variance_epsilon
            )
        else:
            _flashinfer_norm.gemma_fused_add_rmsnorm(
                x, residual, self.weight, self.variance_epsilon
            )
            out = x
        if len(self.quant_methods) > 1:
            out = (out,) * len(self.quant_methods)
        return out if residual is None else (out, residual)

class Gemma3RMSNorm(CustomOp):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))
        # Re-dispatch

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward_native(self, x):
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst Gemma3 is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"
