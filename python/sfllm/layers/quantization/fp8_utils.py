from typing import Callable, List, Optional, Tuple
import torch
from sfllm.layers.quantization.fp8_kernel import (
    fp8_dtype,
    fp8_max,
    is_fp8_fnuz,
    per_token_group_quant_fp8,
    sglang_per_token_quant_fp8,
    static_quant_fp8,
    w8a8_block_fp8_matmul_triton,
)
from sfllm.utils.platform import current_platform

try:
    from sgl_kernel import fp8_scaled_mm
except ImportError:
    fp8_scaled_mm = None

_is_fp8_fnuz = is_fp8_fnuz()
_is_cuda = current_platform.is_cuda()
def triton_w8a8_block_fp8_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    block_size: List[int],
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert input_scale is None
    input_2d = input.view(-1, input.shape[-1])
    output_shape = [*input.shape[:-1], weight.shape[0]]

    q_input, x_scale = per_token_group_quant_fp8(
        input_2d, block_size[1], column_major_scales=False
    )
    output = w8a8_block_fp8_matmul_triton(
        q_input, weight, x_scale, weight_scale, block_size, output_dtype=input_2d.dtype
    )
    if bias is not None:
        output += bias
    return output.to(dtype=input_2d.dtype).view(*output_shape)


def input_to_float8(
    x: torch.Tensor, dtype: torch.dtype = fp8_dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    """This function quantizes input values to float8 values with tensor-wise quantization."""
    min_val, max_val = x.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs()).float().clamp(min=1e-12)

    if _is_fp8_fnuz:
        dtype = fp8_dtype
        fp_max = fp8_max
    else:
        finfo = torch.finfo(dtype)
        fp_max = finfo.max

    scale = fp_max / amax
    x_scl_sat = (x.float() * scale).clamp(min=-fp_max, max=fp_max)
    return x_scl_sat.to(dtype).contiguous(), scale.float().reciprocal()

def normalize_e4m3fn_to_e4m3fnuz(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    assert weight.dtype == torch.float8_e4m3fn
    # The bits pattern 10000000(-128) represents zero in e4m3fn
    # but NaN in e4m3fnuz. So here we set it to 0.
    # https://onnx.ai/onnx/technical/float8.html
    weight_as_int8 = weight.view(torch.int8)
    ROCM_FP8_NAN_AS_INT = -128
    weight_as_int8[weight_as_int8 == ROCM_FP8_NAN_AS_INT] = 0
    weight = weight_as_int8.view(torch.float8_e4m3fnuz)

    # For the same bits representation, e4m3fnuz value is half of
    # the e4m3fn value, so we should double the scaling factor to
    # get the same dequantized value.
    # https://onnx.ai/onnx/technical/float8.html
    weight_scale = weight_scale * 2.0
    if input_scale is not None:
        input_scale = input_scale * 2.0
    return weight, weight_scale, input_scale

def torch_scaled_mm(input, weight, input_scale, weight_scale, out_dtype, bias=None):
    return torch._scaled_mm(
        input, weight, scale_a=input_scale, scale_b=weight_scale,
        out_dtype=out_dtype, bias=bias,
    )


def apply_fp8_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    *,
    scaled_mm: Callable,
) -> torch.Tensor:
    """Quantize a raw activation, then use the layer's selected scaled GEMM."""
    input_2d = input.view(-1, input.shape[-1])
    if input_scale is not None:
        qinput, x_scale = static_quant_fp8(input_2d, input_scale, repeat_scale=False)
    elif _is_cuda and input_2d.shape[-1] % 4 == 0:
        qinput, x_scale = sglang_per_token_quant_fp8(input_2d)
    else:
        qinput, x_scale = per_token_group_quant_fp8(
            input_2d, group_size=input_2d.shape[-1]
        )
    output = scaled_mm(qinput, weight, x_scale, weight_scale, input.dtype, bias)
    return output.view(*input.shape[:-1], weight.shape[1])
