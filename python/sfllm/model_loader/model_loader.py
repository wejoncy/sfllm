from contextlib import ContextDecorator
import torch
import logging
import transformers
from typing import (
    Type,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)
from sfllm.model_loader.model_config import ModelConfig
from sfllm.layers.quantization import QuantizationConfig, get_quantization_config
logger = logging.getLogger(__name__)


class TorchDefaultReset(ContextDecorator):
    def __init__(self, dtype, device="cuda"):
        if not isinstance(dtype, torch.dtype):
            raise TypeError("dtype must be a torch.dtype")
        self.new_dtype = dtype
        self.new_device = device
        self._prev = None
        self.orig_default_device = torch.get_default_device()


    def __enter__(self):
        self._prev = torch.get_default_dtype()
        torch.set_default_dtype(self.new_dtype)
        torch.set_default_device(self.new_device)
        return self

    def __exit__(self, exc_type, exc, tb):
        torch.set_default_dtype(self._prev)
        torch.set_default_device(self.orig_default_device)
        return False

def get_model_architecture(hf_config) -> Tuple[Type[torch.nn.Module], str]:
    from sfllm.model_loader.registry import ModelRegistry

    architectures = getattr(hf_config, "architectures", [])
    supported_archs = ModelRegistry.get_supported_archs()
    is_native_supported = any(arch in supported_archs for arch in architectures)
    assert is_native_supported, f"{architectures} is not support yet"
    return ModelRegistry.resolve_model_cls(architectures)



def get_quant_config(
    model_config,
    packed_modules_mapping: Dict[str, List[str]],
    remap_prefix: Dict[str, str] | None = None,
) -> QuantizationConfig:
    quant_cls = get_quantization_config(model_config.quantization)
    possible_config_filenames = quant_cls.get_config_filenames()
    # If the quantization config is not found, use the default config.
    if not possible_config_filenames:
        return quant_cls()
    return quant_cls.from_config(
        {
            "model_config": model_config,
            "packed_modules_mapping": packed_modules_mapping,
            "remap_prefix": remap_prefix,
        }
    )


def _get_quantization_config(
    model_config,
    packed_modules_mapping: Dict[str, List[str]],
    remap_prefix: Dict[str, str] | None = None,
) -> Optional[QuantizationConfig]:
    """Get the quantization config."""
    if model_config.quantization is not None:
        quant_config = get_quant_config(
            model_config, packed_modules_mapping, remap_prefix
        )
        return quant_config
    return None

def initialize_model(model_name:str, dtype:str="auto", quantization:Optional[str]=None):
    """
    Initialize the ForwardModel with the model name or path.
    
    Args:
        model_name: The name or path of the model to load
    """
    model_config = ModelConfig(model_name, quantization=quantization)
    model = load_model(model_config)

    with TorchDefaultReset(model_config.dtype, device="cuda"):
        for _, module in model.named_modules():
            quant_method = getattr(module, "quant_method", None)
            if quant_method is not None:
                quant_method.process_weights_after_loading(module)
    return model


def load_model(model_config: ModelConfig):
    """
    Load the model and tokenizer
    
    Args:
        model_name: The name or path of the model to load
        
    Returns:
        A dictionary containing model, tokenizer, and processor
    """
    from .weight_utils import _load_check_point
    model_class, _ = get_model_architecture(model_config.hf_config)
    packed_modules_mapping = getattr(model_class, "packed_modules_mapping", {})
    remap_prefix = getattr(model_class, "remap_prefix", None)
    quant_config = _get_quantization_config(model_config, packed_modules_mapping, remap_prefix)
    before_avail_memory, _ = torch.cuda.mem_get_info(0)
    with TorchDefaultReset(model_config.dtype, device="cuda"):
        model = model_class(model_config.hf_config, quant_config=quant_config)
        weight_iterator = _load_check_point(model_config.model_path)
        model.load_weights(weight_iterator)
    model = model.eval()
    after_avail_memory,_ = torch.cuda.mem_get_info(0)
    weight_load_mem_usage = before_avail_memory - after_avail_memory
    logger.info(
        f"Load weight end. "
        f"type={type(model).__name__}, "
        f"dtype={model_config.dtype}, "
        f"avail mem={after_avail_memory / 1024 ** 3:.2f} GB, "
        f"weight load={weight_load_mem_usage / 1024 ** 3:.2f} GB."
    )
    model.dtype = model_config.dtype
    return model
