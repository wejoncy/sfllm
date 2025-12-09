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

from sfllm.models.qwen3 import Qwen3ForCausalLM
from sfllm.models.llama_eagle import LlamaForCausalLMEagle
from sfllm.models.llama_eagle3 import LlamaForCausalLMEagle3

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

ModelRegistry1 = {
    "Qwen3ForCausalLM": Qwen3ForCausalLM,
    "LlamaForCausalLMEagle": LlamaForCausalLMEagle,
    "LlamaForCausalLMEagle3": LlamaForCausalLMEagle3,
}

def get_model_architecture(hf_config) -> Tuple[Type[torch.nn.Module], str]:
    from sfllm.model_loader.registry import ModelRegistry

    architectures = getattr(hf_config, "architectures", [])
    supported_archs = ModelRegistry.get_supported_archs()
    is_native_supported = any(arch in supported_archs for arch in architectures)
    assert is_native_supported, f"{architectures} is not support yet"
    return ModelRegistry.resolve_model_cls(architectures)

def initialize_model(model_name:str, dtype:str="auto"):
    """
    Initialize the ForwardModel with the model name or path.
    
    Args:
        model_name: The name or path of the model to load
    """
    config = transformers.AutoConfig.from_pretrained(model_name)
    conf_dtype = getattr(config, "dtype", None)
    if conf_dtype is None:
        conf_dtype = getattr(config, "torch_dtype", None)
        assert conf_dtype is not None, "config dtype is None"
    conf_dtype = getattr(torch, conf_dtype) if isinstance(conf_dtype, str) else conf_dtype
    config.dtype = conf_dtype
    dtype = conf_dtype if dtype == "auto" else getattr(torch, dtype)
    return load_model(model_name, config, dtype)

def load_model(model_name:str, config, dtype:torch.dtype=torch.float16):
    """
    Load the model and tokenizer
    
    Args:
        model_name: The name or path of the model to load
        
    Returns:
        A dictionary containing model, tokenizer, and processor
    """
    from .weight_utils import _load_check_point
    model_class, _ = get_model_architecture(config)
    before_avail_memory, _ = torch.cuda.mem_get_info(0)
    with TorchDefaultReset(dtype, device="cuda"):
        model = model_class(config)
        weight_iterator = _load_check_point(model_name)
        if hasattr(model, 'load_weights'):
            model.load_weights(weight_iterator)
        else:
            ret = model.load_state_dict(next(weight_iterator), strict=False)
    model = model.eval()
    after_avail_memory,_ = torch.cuda.mem_get_info(0)
    weight_load_mem_usage = before_avail_memory - after_avail_memory
    logger.info(
        f"Load weight end. "
        f"type={type(model).__name__}, "
        f"dtype={dtype}, "
        f"avail mem={after_avail_memory / 1024 ** 3:.2f} GB, "
        f"weight load={weight_load_mem_usage / 1024 ** 3:.2f} GB."
    )
    model.dtype = dtype
    return model
