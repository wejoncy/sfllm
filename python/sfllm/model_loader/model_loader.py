from contextlib import ContextDecorator
import json
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
from sfllm.model_loader.weight_utils import _get_resolved_base_dir
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
) -> Optional[QuantizationConfig]:
    config = getattr(model_config.hf_config, "quantization_config", None)
    if config is None:
        config = getattr(model_config.hf_config.get_text_config(), "quantization_config", None)
    if config is None:
        config_path = _get_resolved_base_dir(model_config.model_path, "hf_quant_config.json")
        if config_path is not None:
            with open(config_path) as f:
                config = json.load(f)

    checkpoint_method = None
    if config is not None:
        checkpoint_method = config.get("quant_method")
        if checkpoint_method in ("modelopt", "compressed-tensors") or "quantization" in config or "quant_algo" in config:
            checkpoint_method = "fp8"
    method = model_config.quantization
    if method is not None and checkpoint_method is not None and method != checkpoint_method:
        raise ValueError(f"Requested {method} quantization, but checkpoint uses {checkpoint_method}")
    method = method or checkpoint_method
    if method is None:
        return None
    quant_cls = get_quantization_config(method)
    quant_config = quant_cls.from_config(config) if config is not None else quant_cls()
    quant_config.packed_modules_mapping = packed_modules_mapping
    for source, target in (remap_prefix or {}).items():
        quant_config.ignored_layers = [
            name.replace(source, target, 1) for name in quant_config.ignored_layers
        ]
    return quant_config


def initialize_model(model_name:str, dtype:str="auto", quantization:Optional[str]=None):
    """
    Initialize the ForwardModel with the model name or path.
    
    Args:
        model_name: The name or path of the model to load
    """
    model_config = ModelConfig(model_name, dtype=dtype, quantization=quantization)
    before_avail_memory, _ = torch.cuda.mem_get_info(0)
    model = load_model(model_config)
    with TorchDefaultReset(model_config.dtype, device="cuda"):
        for _, module in model.named_modules():
            quant_method = getattr(module, "quant_method", None)
            if quant_method is not None:
                quant_method.process_weights_after_loading(module)
    torch.cuda.empty_cache()
    after_avail_memory,_ = torch.cuda.mem_get_info(0)
    weight_load_mem_usage = before_avail_memory - after_avail_memory
    logger.info(
        f"Load weight end. "
        f"type={type(model).__name__}, "
        f"dtype={model_config.dtype}, "
        f"avail mem={after_avail_memory / 1024 ** 3:.2f} GB, "
        f"weight load={weight_load_mem_usage / 1024 ** 3:.2f} GB."
    )
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
    quant_config = get_quant_config(model_config, packed_modules_mapping, remap_prefix)
    with TorchDefaultReset(model_config.dtype, device="cuda"):
        model = model_class(model_config.hf_config, quant_config=quant_config)
        weight_iterator = _load_check_point(model_config.model_path)
        model.load_weights(weight_iterator)
    model = model.eval()
    model.dtype = model_config.dtype
    return model
