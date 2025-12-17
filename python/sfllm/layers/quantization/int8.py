import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter
from sfllm.layers.quantization.int8_kernel import (
    per_token_group_quant_int8, w8a8_block_int8_matmul, block_int8_weight_quant)

from sfllm.utils.platform import current_platform

# from vllm import _custom_ops as ops
from sfllm.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
    LinearMethodBase,
)
from sfllm.utils import get_tensor_model_parallel_world_size,get_tensor_model_parallel_rank

from sfllm.layers.quantization.unquant import UnquantizedLinearMethod
from sfllm.layers.quantization.utils import (
    convert_to_channelwise,
    is_layer_skipped,
    requantize_with_max_scale,
)

from sfllm.layers.parameter import (
    BlockQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)

ACTIVATION_SCHEMES = ["static", "dynamic"]

logger = logging.getLogger(__name__)


class Int8Config(QuantizationConfig):
    """Config class for INT8."""

    def __init__(
        self,
        is_checkpoint_int8_serialized: bool = False,
        activation_scheme: str = "dynamic",
        ignored_layers: Optional[List[str]] = None,
        weight_block_size: List[int] = None,
    ) -> None:
        self.is_checkpoint_int8_serialized = is_checkpoint_int8_serialized
        if is_checkpoint_int8_serialized:
            logger.info("Detected int8 checkpoint.")
        if activation_scheme not in ACTIVATION_SCHEMES:
            raise ValueError(f"Unsupported activation scheme {activation_scheme}")
        self.activation_scheme = activation_scheme
        self.ignored_layers = ignored_layers or []
        if activation_scheme == "dynamic" and not is_checkpoint_int8_serialized:
            weight_block_size = [128, 128]
        if weight_block_size is not None:
            if not is_checkpoint_int8_serialized:
                assert weight_block_size == [128, 128], \
                    "Only [128, 128] block size is supported for non-int8 serialized checkpoints."
            if len(weight_block_size) != 2:
                raise ValueError(
                    f"The quantization block size of weight must have 2 dimensions, but got {len(weight_block_size)} dimensions."
                )
            if activation_scheme != "dynamic":
                raise ValueError(
                    f"The block-wise quantization only supports dynamic activation scheme for now, but got {activation_scheme} activation scheme."
                )
        self.weight_block_size = weight_block_size

    @classmethod
    def get_name(cls) -> str:
        return "int8"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Int8Config":
        quant_method = cls.get_from_keys(config, ["quant_method"])
        is_checkpoint_int8_serialized = "int8" in quant_method
        activation_scheme = cls.get_from_keys(config, ["activation_scheme"])
        ignored_layers = cls.get_from_keys_or(
            config, ["ignored_layers", "modules_to_not_convert"], None
        )
        if ignored_layers:
            # hacking ministral
            ignored_layers = [layer.replace("model.", "") for layer in ignored_layers]
        weight_block_size = cls.get_from_keys_or(config, ["weight_block_size"], None)
        return cls(
            is_checkpoint_int8_serialized=is_checkpoint_int8_serialized,
            activation_scheme=activation_scheme,
            ignored_layers=ignored_layers,
            weight_block_size=weight_block_size,
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        from sfllm.layers.linear import LinearBase
        if isinstance(layer, LinearBase):
            if is_layer_skipped(prefix, self.ignored_layers):
                return UnquantizedLinearMethod()
            return Int8LinearMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []

def apply_w8a8_block_int8_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    block_size: List[int],
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert input_scale is None
    # View input as 2D matrix for fp8 methods
    input_2d = input.view(-1, input.shape[-1])
    output_shape = [*input.shape[:-1], weight.shape[0]]

    q_input, x_scale = per_token_group_quant_int8(input_2d, block_size[1])
    output = w8a8_block_int8_matmul(
        q_input, weight, x_scale, weight_scale, block_size, output_dtype=input.dtype
    )

    if bias is not None:
        output = output + bias
    return output.to(dtype=input.dtype).view(*output_shape)

def apply_int8_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    use_per_token_if_dynamic: bool = False):
    """Applies INT8 linear with daslab kernel."""

    input_dtype = input.dtype    
    result = input @ (weight.to(input_dtype) * weight_scale.to(input_dtype))
    if bias is not None:
        result += bias
    return result


def input_to_int8(weight: torch.Tensor, dtype=torch.int8,per_token: bool = True):
    """Quantize the input tensor to int8 and return the scale."""
    min_val, max_val = weight.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs()).float().clamp(min=1e-12)

    iinfo = torch.iinfo(dtype)
    int8_max = iinfo.max

    scale = int8_max / amax
    x_scl_sat = (weight.float() * scale).clamp(min=-int8_max, max=int8_max)
    return x_scl_sat.to(dtype).contiguous(), scale.float().reciprocal()

class Int8LinearMethod(LinearMethodBase):
    """Linear method for FP8.
    Supports loading FP8 checkpoints with static weight scale and
    dynamic/static activation scale.

    Also supports loading quantized FP16/BF16 model checkpoints with dynamic
    activation scaling. The weight scaling factor will be initialized after
    the model weights are loaded.

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: Union[Int8Config, ]):
        self.quant_config = quant_config

        # For GPUs that lack FP8 hardware support, we can leverage the Marlin
        # kernel for fast weight-only FP8 quantization
        self.use_marlin = False
        self.block_quant = self.quant_config.weight_block_size is not None

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        tp_size = get_tensor_model_parallel_world_size()
        if self.block_quant and self.quant_config.is_checkpoint_int8_serialized:
            block_n, block_k = (
                self.quant_config.weight_block_size[0],
                self.quant_config.weight_block_size[1],
            )
            # Required by row parallel
            if tp_size > 1 and input_size // input_size_per_partition == tp_size:
                if input_size_per_partition % block_k != 0:
                    raise ValueError(
                        f"Weight input_size_per_partition = "
                        f"{input_size_per_partition} is not divisible by "
                        f"weight quantization block_k = {block_k}."
                    )
            # Required by column parallel or enabling merged weights
            if (
                tp_size > 1 and output_size // output_size_per_partition == tp_size
            ) or len(output_partition_sizes) > 1:
                for output_partition_size in output_partition_sizes:
                    if output_partition_size % block_n != 0:
                        raise ValueError(
                            f"Weight output_partition_size = "
                            f"{output_partition_size} is not divisible by "
                            f"weight quantization block_n = {block_n}."
                        )

        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype

        # WEIGHT
        weight_dtype = (
            torch.int8
            if self.quant_config.is_checkpoint_int8_serialized
            else params_dtype
        )

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition, input_size_per_partition, dtype=weight_dtype
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        # If checkpoint is serialized int8, load them.
        # Otherwise, wait until process_weights_after_loading.
        if self.quant_config.is_checkpoint_int8_serialized:
            # WEIGHT SCALE
            if self.block_quant:
                if hasattr(self.quant_config, "activation_scheme"):
                    assert self.quant_config.activation_scheme == "dynamic"
                elif hasattr(self.quant_config, "linear_activation_scheme"):
                    assert self.quant_config.linear_activation_scheme == "dynamic"
                scale = BlockQuantScaleParameter(
                    data=torch.empty(
                        (output_size_per_partition + block_n - 1) // block_n,
                        (input_size_per_partition + block_k - 1) // block_k,
                        dtype=torch.float32,
                    ),
                    input_dim=1,
                    output_dim=0,
                    weight_loader=weight_loader,
                )
                scale[:] = torch.finfo(torch.float32).min
                layer.register_parameter("weight_scale_inv", scale)
            else:
                scale = PerTensorScaleParameter(
                    data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                    weight_loader=weight_loader,
                )
                scale[:] = torch.finfo(torch.float32).min
                layer.register_parameter("weight_scale", scale)

            # INPUT ACTIVATION SCALE
            if (
                hasattr(self.quant_config, "activation_scheme")
                and self.quant_config.activation_scheme == "static"
            ) or (
                hasattr(self.quant_config, "linear_activation_scheme")
                and self.quant_config.linear_activation_scheme == "static"
            ):
                scale = PerTensorScaleParameter(
                    data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                    weight_loader=weight_loader,
                )

                scale[:] = torch.finfo(torch.float32).min
                layer.register_parameter("input_scale", scale)
            else:
                layer.register_parameter("input_scale", None)

    def process_weights_after_loading(self, layer: Module) -> None:
        if self.block_quant:
            if self.quant_config.is_checkpoint_int8_serialized:
                weight, weight_scale = layer.weight.data, layer.weight_scale_inv.data
                layer.weight.data = weight.data
                layer.weight_scale_inv.data = weight_scale.data
            else:
                qweight, weight_scale = block_int8_weight_quant(
                    layer.weight.data,
                    self.quant_config.weight_block_size[0],
                )
                layer.weight = Parameter(qweight, requires_grad=False)
                layer.weight_scale_inv = Parameter(
                    weight_scale, requires_grad=False
                )
        else:
            layer.weight = Parameter(layer.weight.data, requires_grad=False)

            # If checkpoint not serialized int8, quantize the weights.
            if not self.quant_config.is_checkpoint_int8_serialized:
                # per-tensor quantization
                qweight, weight_scale = input_to_int8(layer.weight)
                # Update the layer with the new values.
                layer.weight = Parameter(qweight.t(), requires_grad=False)
                layer.weight_scale = Parameter(weight_scale, requires_grad=False)
                layer.input_scale = None


    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.block_quant:
            if isinstance(x, tuple):
                return apply_w8a8_block_int8_linear(
                    input=x[0],
                    weight=layer.weight,
                    block_size=self.quant_config.weight_block_size,
                    weight_scale=layer.weight_scale_inv,
                    input_scale=x[1],
                    bias=bias,
                )

            return apply_w8a8_block_int8_linear(
                input=x,
                weight=layer.weight,
                block_size=self.quant_config.weight_block_size,
                weight_scale=layer.weight_scale_inv,
                input_scale=None,
                bias=bias,
            )

        return apply_int8_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            input_scale=layer.input_scale,
            bias=bias,
            use_per_token_if_dynamic=False,
        )
