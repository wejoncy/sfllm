from typing import Callable, List, Optional, Tuple
import torch
from sfllm.layers.quantization.fp8_kernel import (
    fp8_dtype,
    fp8_max,
    is_fp8_fnuz,
    per_token_group_quant_fp8,
    scaled_fp8_quant,
    sglang_per_token_quant_fp8,
    static_quant_fp8,
    triton_scaled_mm,
    w8a8_block_fp8_matmul_deepgemm,
    w8a8_block_fp8_matmul_triton,
)

_is_fp8_fnuz = is_fp8_fnuz()

CUTLASS_BLOCK_FP8_SUPPORTED = False

def cutlass_fp8_supported() -> bool:
    """Check if CUTLASS FP8 is supported."""
    return CUTLASS_BLOCK_FP8_SUPPORTED

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


def dispatch_w8a8_block_fp8_linear() -> Callable:
    return triton_w8a8_block_fp8_linear

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

def requant_weight_ue8m0_inplace(weight, weight_scale_inv, weight_block_size):
    assert isinstance(weight, torch.nn.Parameter)
    assert isinstance(weight_scale_inv, torch.nn.Parameter)

    new_weight, new_weight_scale_inv = requant_weight_ue8m0(
        weight.to(weight_scale_inv.device), weight_scale_inv, weight_block_size
    )

    offloader.update_param(weight, new_weight)
    weight_scale_inv.data = new_weight_scale_inv


def _process_scaled_mm_output(output, input_2d_shape, output_shape):
    if type(output) is tuple and len(output) == 2:
        output = output[0]
    return torch.narrow(output, 0, 0, input_2d_shape[0]).view(*output_shape)

def _apply_fallback_scaled_mm(
    qinput,
    weight,
    x_scale,
    weight_scale,
    input_2d_shape,
    output_shape,
    bias,
    input_dtype,
):
    global TORCH_DEVICE_IDENTITY
    if TORCH_DEVICE_IDENTITY is None:
        TORCH_DEVICE_IDENTITY = torch.ones(1, dtype=torch.float32, device=weight.device)

    output = torch._scaled_mm(
        qinput,
        weight,
        scale_a=TORCH_DEVICE_IDENTITY,
        scale_b=TORCH_DEVICE_IDENTITY,
        out_dtype=torch.float32,
    )

    output = _process_scaled_mm_output(output, input_2d_shape, output_shape)
    x_scale = torch.narrow(x_scale, 0, 0, input_2d_shape[0])

    output = output * x_scale * weight_scale.t()
    if bias is not None:
        output = output + bias
    return output.to(dtype=input_dtype)

def apply_fp8_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    input_scale_ub: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    cutlass_fp8_supported: bool = cutlass_fp8_supported(),
    use_per_token_if_dynamic: bool = False,
    pad_output: Optional[bool] = None,
    compressed_tensor_quant: bool = False,
) -> torch.Tensor:
    # Note: we pad the input because torch._scaled_mm is more performant
    # for matrices with batch dimension > 16.
    # This could change in the future.
    # We also don't pad when using torch.compile,
    # as it breaks with dynamic shapes.
    if pad_output is None:
        pad_output = (
            not get_bool_env_var("SGLANG_ENABLE_TORCH_COMPILE")
            and not cutlass_fp8_supported
        )
    output_padding = 17 if pad_output else None

    # View input as 2D matrix for fp8 methods
    input_2d = input.view(-1, input.shape[-1])
    output_shape = [*input.shape[:-1], weight.shape[1]]

    if compressed_tensor_quant:
        # Maybe apply padding to output, see comment in __init__
        num_token_padding = output_padding
        if cutlass_fp8_supported and weight_scale.numel() == weight.shape[1]:
            num_token_padding = None
        qinput, x_scale = scaled_fp8_quant(
            input_2d,
            input_scale,
            num_token_padding=num_token_padding,
            use_per_token_if_dynamic=use_per_token_if_dynamic,
        )
    else:
        # cutlass w8a8 fp8 sgl-kernel only supports per-token scale
        if input_scale is not None:
            assert input_scale.numel() == 1
            # broadcast per-tensor scale to per-token scale when supporting cutlass
            qinput, x_scale = static_quant_fp8(
                input_2d, input_scale, repeat_scale=cutlass_fp8_supported
            )
        else:
            # default use per-token quantization if dynamic
            if _is_cuda:
                qinput, x_scale = sglang_per_token_quant_fp8(input_2d)
            else:
                # TODO(kkhuang): temporarily enforce per-tensor activation scaling if weight is per-tensor scaling
                # final solution should be: 1. add support to per-tensor activation scaling.
                # 2. solve the torch.compile error from weight_scale.numel() == 1 and x_scale.numel() > 1 (below line#308)
                if _is_hip and weight_scale.numel() == 1:
                    qinput, x_scale = scaled_fp8_quant(
                        input_2d,
                        input_scale,
                        use_per_token_if_dynamic=use_per_token_if_dynamic,
                    )
                else:
                    qinput, x_scale = per_token_group_quant_fp8(
                        input_2d, group_size=input_2d.shape[1]
                    )
# torch.scaled_mm supports per tensor weights + activations only
    # so fallback to naive if per channel or per token
    per_tensor_weights = weight_scale.numel() == 1
    per_tensor_activations = x_scale.numel() == 1

    if (
        use_per_token_if_dynamic
        and not per_tensor_weights
        and not per_tensor_activations
        # and (USE_ROWWISE_TORCH_SCALED_MM)
    ):
        # For now validated on ROCm platform
        # fp8 rowwise scaling in torch._scaled_mm is introduced in
        # https://github.com/pytorch/pytorch/pull/144432 using hipBLASLt
        # and ROCm 6.3, which only exists in torch 2.7 and above.
        # For CUDA platform please validate if the
        # torch._scaled_mm support rowwise scaled GEMM
        # Fused GEMM_DQ Rowwise GEMM
        output = torch._scaled_mm(
            qinput,
            weight,
            out_dtype=input.dtype,
            scale_a=x_scale,
            scale_b=weight_scale.t(),
            bias=bias,
        )
        return _process_scaled_mm_output(output, input_2d.shape, output_shape)

    if per_tensor_weights and per_tensor_activations:
        # Fused GEMM_DQ
        output = torch._scaled_mm(
            qinput,
            weight,
            out_dtype=input.dtype,
            scale_a=x_scale,
            scale_b=weight_scale,
            bias=bias,
        )
        return _process_scaled_mm_output(output, input_2d.shape, output_shape)
    
    return _apply_fallback_scaled_mm(
        qinput,
        weight,
        x_scale,
        weight_scale,
        input_2d.shape,
        output_shape,
        bias,
        input.dtype,
    )