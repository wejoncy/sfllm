import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

from sfllm.utils.platform import current_platform

# from vllm import _custom_ops as ops
from sfllm.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
    LinearMethodBase,
)
from sfllm.utils import get_tensor_model_parallel_world_size,get_tensor_model_parallel_rank

from sfllm.layers.quantization.fp8_kernel import (
    fp8_dtype,
    triton_scaled_mm,
)
from sfllm.layers.quantization.fp8_utils import (
    apply_fp8_linear,
    fp8_scaled_mm,
    torch_scaled_mm,
    triton_w8a8_block_fp8_linear,
    input_to_float8,
    normalize_e4m3fn_to_e4m3fnuz,
    _is_fp8_fnuz,
)

from sfllm.layers.quantization.utils import (
    is_layer_skipped,
    requantize_with_max_scale,
)

from sfllm.layers.parameter import (
    BlockQuantScaleParameter,
    ChannelQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)


ACTIVATION_SCHEMES = ["static", "dynamic"]

logger = logging.getLogger(__name__)


class Fp8Config(QuantizationConfig):
    """Config class for FP8."""

    def __init__(
        self,
        is_checkpoint_fp8_serialized: bool = False,
        activation_scheme: str = "dynamic",
        ignored_layers: Optional[List[str]] = None,
        weight_block_size: List[int] = None,
        weight_strategy: str = "tensor",
    ) -> None:
        super().__init__()
        self.is_checkpoint_fp8_serialized = is_checkpoint_fp8_serialized
        if is_checkpoint_fp8_serialized:
            logger.info("Detected fp8 checkpoint.")
        if activation_scheme not in ACTIVATION_SCHEMES:
            raise ValueError(f"Unsupported activation scheme {activation_scheme}")
        self.activation_scheme = activation_scheme
        self.ignored_layers = ignored_layers or []
        if weight_block_size is not None:
            if not is_checkpoint_fp8_serialized:
                raise ValueError(
                    f"The block-wise quantization only supports fp8-serialized checkpoint for now."
                )
            if len(weight_block_size) != 2:
                raise ValueError(
                    f"The quantization block size of weight must have 2 dimensions, but got {len(weight_block_size)} dimensions."
                )
            if activation_scheme != "dynamic":
                raise ValueError(
                    f"The block-wise quantization only supports dynamic activation scheme for now, but got {activation_scheme} activation scheme."
                )
        self.weight_block_size = weight_block_size
        self.weight_strategy = weight_strategy

    @classmethod
    def get_name(cls) -> str:
        return "fp8"

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
    def from_config(cls, config: Dict[str, Any]) -> "Fp8Config":
        config = config.get("quantization", config)
        if config.get("quant_method") == "compressed-tensors":
            groups = list(config.get("config_groups", {}).values())
            if (
                config.get("format") != "float-quantized"
                or config.get("quantization_status") != "compressed"
                or len(groups) != 1
                or groups[0].get("targets") != ["Linear"]
                or groups[0].get("output_activations") is not None
                or config.get("kv_cache_scheme")
                or config.get("sparsity_config")
                or config.get("transform_config")
            ):
                raise ValueError("Only compressed-tensors FP8_DYNAMIC linear quantization is supported")
            for name, strategy, dynamic in (
                ("weights", "channel", False),
                ("input_activations", "token", True),
            ):
                args = groups[0].get(name) or {}
                if (
                    args.get("type") != "float"
                    or args.get("num_bits") != 8
                    or args.get("strategy") != strategy
                    or args.get("dynamic") is not dynamic
                    or args.get("symmetric") is not True
                    or args.get("group_size") is not None
                    or args.get("block_structure") is not None
                ):
                    raise ValueError(f"Unsupported compressed-tensors FP8_DYNAMIC {name}: {args}")
            return cls(
                is_checkpoint_fp8_serialized=True,
                activation_scheme="dynamic",
                ignored_layers=config.get("ignore"),
                weight_strategy="channel",
            )
        if "quant_algo" in config or config.get("quant_method") == "modelopt":
            if config.get("quant_algo") != "FP8":
                raise ValueError("Only ModelOpt per-tensor FP8 checkpoints are supported")
            if config.get("kv_cache_quant_algo") or config.get("kv_cache_scheme"):
                raise ValueError("ModelOpt FP8 KV-cache quantization is not supported")
            return cls(
                is_checkpoint_fp8_serialized=True,
                activation_scheme="static",
                ignored_layers=config.get("ignore", config.get("exclude_modules")),
            )
        quant_method = cls.get_from_keys(config, ["quant_method"])
        is_checkpoint_fp8_serialized = "fp8" in quant_method
        activation_scheme = cls.get_from_keys(config, ["activation_scheme"])
        ignored_layers = cls.get_from_keys_or(
            config, ["ignored_layers", "modules_to_not_convert"], None
        )
        if ignored_layers:
            # hacking ministral
            ignored_layers = [layer.replace("model.", "") for layer in ignored_layers]
        weight_block_size = cls.get_from_keys_or(config, ["weight_block_size"], None)
        return cls(
            is_checkpoint_fp8_serialized=is_checkpoint_fp8_serialized,
            activation_scheme=activation_scheme,
            ignored_layers=ignored_layers,
            weight_block_size=weight_block_size,
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        from sfllm.layers.linear import LinearBase, UnquantizedLinearMethod
        if isinstance(layer, LinearBase):
            if is_layer_skipped(prefix, self.ignored_layers, self.packed_modules_mapping):
                return UnquantizedLinearMethod()
            return Fp8LinearMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class Fp8LinearMethod(LinearMethodBase):
    """FP8 linear weights with tensor, channel, or block scales.

    Fused producers may pass (FP8 activations, scales) for non-block weights.
    """

    def __init__(self, quant_config: Union[Fp8Config, ]):
        self.quant_config = quant_config

        self.block_quant = self.quant_config.weight_block_size is not None
        self.input_dtype = None if self.block_quant else fp8_dtype
        self.scaled_mm = triton_scaled_mm
        if current_platform.is_cuda() and current_platform.has_device_capability((8, 9)):
            # Preserve the original accumulation for scalar activation/weight scales.
            if quant_config.activation_scheme == "static" and quant_config.weight_strategy == "tensor":
                self.scaled_mm = torch_scaled_mm
            elif fp8_scaled_mm is not None:
                self.scaled_mm = fp8_scaled_mm

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
        # Native GEMMs require aligned dimensions and BF16/FP16 output.
        output_alignment = 16 if self.scaled_mm is torch_scaled_mm else 8
        if (input_size_per_partition % 16 or output_size_per_partition % output_alignment
                or params_dtype not in (torch.float16, torch.bfloat16)):
            self.scaled_mm = triton_scaled_mm

        tp_size = get_tensor_model_parallel_world_size()
        if self.block_quant:
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
            torch.float8_e4m3fn
            if self.quant_config.is_checkpoint_fp8_serialized
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

        # If checkpoint is serialized fp8, load them.
        # Otherwise, wait until process_weights_after_loading.
        if self.quant_config.is_checkpoint_fp8_serialized:
            # WEIGHT SCALE
            if self.block_quant:
                assert self.quant_config.activation_scheme == "dynamic"
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
            elif self.quant_config.weight_strategy == "channel":
                scale = ChannelQuantScaleParameter(
                    data=torch.full(
                        (output_size_per_partition, 1),
                        torch.finfo(torch.float32).min,
                        dtype=torch.float32,
                    ),
                    output_dim=0,
                    weight_loader=weight_loader,
                )
                layer.register_parameter("weight_scale", scale)
            else:
                scale = PerTensorScaleParameter(
                    data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                    weight_loader=weight_loader,
                )
                scale[:] = torch.finfo(torch.float32).min
                layer.register_parameter("weight_scale", scale)

            # INPUT ACTIVATION SCALE
            if self.quant_config.activation_scheme == "static":
                scale = PerTensorScaleParameter(
                    data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                    weight_loader=weight_loader,
                )

                scale[:] = torch.finfo(torch.float32).min
                layer.register_parameter("input_scale", scale)
            else:
                layer.register_parameter("input_scale", None)

    def process_weights_after_loading(self, layer: Module) -> None:
        if self.quant_config.is_checkpoint_fp8_serialized and not self.block_quant:
            for name in ("weight_scale", "input_scale"):
                scale = getattr(layer, name, None)
                if scale is not None and not torch.all(torch.isfinite(scale) & (scale > 0)):
                    raise ValueError(f"Missing or invalid FP8 {name} in {layer.prefix}")
        if self.block_quant:
            # If ROCm, normalize the weights and scales to e4m3fnuz
            if _is_fp8_fnuz:
                # activation_scheme: dynamic
                weight, weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
                    weight=layer.weight,
                    weight_scale=layer.weight_scale_inv,
                    input_scale=None,
                )
                layer.input_scale = None
            else:
                weight, weight_scale = layer.weight.data, layer.weight_scale_inv.data

            layer.weight.data = weight.data
            layer.weight_scale_inv.data = weight_scale.data
        else:
            layer.weight = Parameter(layer.weight.data, requires_grad=False)

            # If checkpoint not serialized fp8, quantize the weights.
            if not self.quant_config.is_checkpoint_fp8_serialized:
                qweight, weight_scale = input_to_float8(layer.weight)

                # Update the layer with the new values.
                layer.weight = Parameter(qweight.t(), requires_grad=False)
                layer.weight_scale = Parameter(weight_scale, requires_grad=False)
                layer.input_scale = None

            # If checkpoint is fp8, handle that there are N scales for N
            # shards in a fused module
            else:
                layer.weight_scale = Parameter(
                    layer.weight_scale.data, requires_grad=False
                )
                weight = layer.weight
                weight_scale = layer.weight_scale
                # If ROCm, normalize the weights and scales to e4m3fnuz
                if _is_fp8_fnuz:
                    weight, weight_scale, input_scale = (
                        normalize_e4m3fn_to_e4m3fnuz(
                            weight=weight,
                            weight_scale=weight_scale,
                            input_scale=layer.input_scale,
                        )
                    )
                    if input_scale is not None:
                        layer.input_scale = Parameter(
                            input_scale, requires_grad=False
                        )

                if self.quant_config.weight_strategy != "channel":
                    # Merge per-tensor scales; preserve serialized channel scales.
                    weight_scale, weight = requantize_with_max_scale(
                        weight=weight,
                        weight_scale=weight_scale,
                        logical_widths=layer.logical_widths,
                    )

                # Update layer with new values.
                layer.weight = Parameter(weight.t(), requires_grad=False)
                layer.weight_scale = Parameter(weight_scale, requires_grad=False)
                if self.quant_config.activation_scheme == "static":
                    layer.input_scale = Parameter(
                        layer.input_scale.max(), requires_grad=False
                    )

        if not self.block_quant:
            self.input_scale = layer.input_scale
            if self.scaled_mm is fp8_scaled_mm and layer.weight_scale.numel() == 1:
                # SGL GEMM accepts scalar activation scales but needs one weight scale per column.
                layer.weight_scale = Parameter(
                    layer.weight_scale.expand(layer.weight.shape[1], 1).contiguous(),
                    requires_grad=False,
                )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.block_quant:
            return triton_w8a8_block_fp8_linear(
                input=x,
                weight=layer.weight,
                block_size=self.quant_config.weight_block_size,
                weight_scale=layer.weight_scale_inv,
                input_scale=None,
                bias=bias,
            )

        if isinstance(x, tuple):
            quantized, scale = x
            return self.scaled_mm(
                quantized, layer.weight, scale, layer.weight_scale,
                layer.params_dtype, bias,
            )

        return apply_fp8_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            input_scale=layer.input_scale,
            bias=bias,
            scaled_mm=self.scaled_mm,
        )
