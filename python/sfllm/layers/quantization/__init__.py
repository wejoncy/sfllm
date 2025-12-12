from sfllm.layers.quantization.fp8 import Fp8Config
from .base_config import QuantizationConfig
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, Union

# Base quantization methods
QUANTIZATION_METHODS: Dict[str, Type[QuantizationConfig]] = {
    "fp8": Fp8Config,
}

def get_quantization_config(quantization: str) -> Type[QuantizationConfig]:
    if quantization not in QUANTIZATION_METHODS:
        raise ValueError(
            f"Invalid quantization method: {quantization}. "
            f"Available methods: {list(QUANTIZATION_METHODS.keys())}"
        )

    return QUANTIZATION_METHODS[quantization]