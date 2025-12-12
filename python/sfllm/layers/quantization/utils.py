import torch
from typing import Callable, List, Mapping, Union, Tuple
from sfllm.layers.quantization.fp8_kernel import scaled_fp8_quant
from types import MappingProxyType


def per_tensor_dequantize(
    tensor: torch.Tensor, inv_scale: Union[float, torch.Tensor]
) -> torch.Tensor:
    fake_qweight = tensor.to(torch.float16)
    dq_weight = fake_qweight * inv_scale
    return dq_weight

def requantize_with_max_scale(
    weight: torch.Tensor, weight_scale: torch.Tensor, logical_widths: List[int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Max scale to be used for requanitzation.
    max_w_scale = weight_scale.max()

    # QKV / MLP is fused in the on disk checkpoint if any of the
    # weight scales are still set to the default since we initialize
    # N weight scales for N shards but we only load 1 weight scale
    # from disk in this case. Skip requantization in this case (since)
    # we already are quantized with the single scale.
    # * Sample Model: nm-testing/Phi-3-mini-128k-instruct-FP8
    unfused_module_in_checkpoint = (
        weight_scale[-1] > torch.finfo(torch.float8_e4m3fn).min
    )

    # If unfused checkpoint, need requanize with the single scale.
    if unfused_module_in_checkpoint:
        start = 0
        for idx, logical_width in enumerate(logical_widths):
            end = start + logical_width
            weight_dq = per_tensor_dequantize(weight[start:end, :], weight_scale[idx])
            weight[start:end, :], _ = scaled_fp8_quant(weight_dq, max_w_scale)
            start = end

    return max_w_scale, weight


def convert_to_channelwise(
    weight_scale: torch.Tensor, logical_widths: List[int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Create channelwise buffer
    weight_scale_channel = torch.empty(
        (sum(logical_widths), 1), dtype=torch.float32, device=weight_scale.device
    )

    # Handle scalar tensor case: broadcast same scale to all channels
    if weight_scale.dim() == 0:
        weight_scale_channel.fill_(weight_scale.item())
        return weight_scale_channel

    # Expand each scale to match the size of each logical matrix.
    start = 0
    for idx, logical_width in enumerate(logical_widths):
        end = start + logical_width
        weight_scale_channel[start:end, :] = weight_scale[idx]
        start = end

    return weight_scale_channel


def requantize_with_max_scale(
    weight: torch.Tensor, weight_scale: torch.Tensor, logical_widths: List[int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Max scale to be used for requanitzation.
    max_w_scale = weight_scale.max()

    # QKV / MLP is fused in the on disk checkpoint if any of the
    # weight scales are still set to the default since we initialize
    # N weight scales for N shards but we only load 1 weight scale
    # from disk in this case. Skip requantization in this case (since)
    # we already are quantized with the single scale.
    # * Sample Model: nm-testing/Phi-3-mini-128k-instruct-FP8
    unfused_module_in_checkpoint = (
        weight_scale[-1] > torch.finfo(torch.float8_e4m3fn).min
    )

    # If unfused checkpoint, need requanize with the single scale.
    if unfused_module_in_checkpoint:
        start = 0
        for idx, logical_width in enumerate(logical_widths):
            end = start + logical_width
            weight_dq = per_tensor_dequantize(weight[start:end, :], weight_scale[idx])
            weight[start:end, :], _ = scaled_fp8_quant(weight_dq, max_w_scale)
            start = end

    return max_w_scale, weight

def is_layer_skipped(
    prefix: str,
    ignored_layers: List[str],
    fused_mapping: Mapping[str, List[str]] = MappingProxyType({}),
) -> bool:
    # prefix: model.layers.0.self_attn.q_proj
    # proj_name: q_proj
    proj_name = prefix.split(".")[-1]

    # Fused layers like gate_up_proj or qkv_proj will not be fused
    # in the safetensors checkpoint. So, we convert the name
    # from the fused version to unfused + check to make sure that
    # each shard of the fused layer has the same scheme.
    if proj_name in fused_mapping:
        shard_prefixes = [
            prefix.replace(proj_name, shard_proj_name)
            for shard_proj_name in fused_mapping[proj_name]
        ]

        is_skipped = None
        for shard_prefix in shard_prefixes:
            is_shard_skipped = any(
                ignored in shard_prefix for ignored in ignored_layers
            )

            if is_skipped is None:
                is_skipped = is_shard_skipped
            elif is_shard_skipped != is_skipped:
                raise ValueError(
                    f"Detected some but not all shards of {prefix} "
                    "are quantized. All shards of fused layers "
                    "to have the same precision."
                )
    else:
        is_skipped = any(ignored in prefix for ignored in ignored_layers)
        if "gate_up_proj" in prefix:
            prefix_gate = prefix.replace("gate_up_proj", "gate_proj")
            prefix_up = prefix.replace("gate_up_proj", "up_proj")
            if prefix_gate in ignored_layers and prefix_up in ignored_layers:
                is_skipped = True
        elif "experts" in prefix:
            is_skipped = any(
                [
                    prefix in layer_name
                    for layer_name in ignored_layers
                    if "experts" in layer_name
                ]
            )

    assert is_skipped is not None
    return is_skipped