from contextlib import ContextDecorator
import glob
import json
import os
from pathlib import Path
import safetensors
import torch
import logging
from tqdm import tqdm
import transformers


from sfllm.models.qwen3 import Qwen3ForCausalLM
from sfllm.models.llama_eagle import LlamaForCausalLMEagle
from sfllm.models.llama_eagle3 import LlamaForCausalLMEagle3

logger = logging.getLogger(__name__)

def _hf_weight_generator(hf_weights_files, is_safetensors:bool):
    if is_safetensors:
        from safetensors.torch import safe_open
        for st_file in hf_weights_files:
            with safe_open(st_file, framework="pt", device="cuda") as f:
                for name in f.keys():  # noqa: SIM118
                    param = f.get_tensor(name)
                    yield name, param
    else:
        for bin_file in hf_weights_files:
            state = torch.load(bin_file, map_location="cuda")
            for name, param in state.items():
                yield name, param
            del state
            torch.cuda.empty_cache()


def _get_resolved_weight_or_index_file(model_name_or_path):
    if Path(model_name_or_path).exists():  # local
        weight_or_index_file = glob.glob(str(Path(model_name_or_path).absolute()/ '*.index.json'))
        weight_or_index_file += glob.glob(str(Path(model_name_or_path).absolute()/ '*.safetensors'))
        weight_or_index_file += glob.glob(str(Path(model_name_or_path).absolute()/ 'pytorch_model*.bin'))
        if weight_or_index_file: 
            weight_or_index_file = weight_or_index_file[0]
            
        else:
            raise FileNotFoundError("model weight is not found")
    else:
        for possible_index_name in ["model.safetensors.index.json", "pytorch_model.bin.index.json"]:
            weight_or_index_file = BaseQuantizeConfig.get_resolved_base_dir(model_name_or_path, possible_index_name)
            if weight_or_index_file:break
        if not weight_or_index_file:
            for possible_weight_file in ["model.safetensors", "pytorch_model.bin"]:
                weight_or_index_file = cached_file(model_name_or_path, possible_weight_file)
                if weight_or_index_file:break
    return str(weight_or_index_file)


def _load_check_point(model_name_or_path, disable_mmap: bool = False):
    from transformers.utils.hub import cached_file
    import concurrent
    weight_or_index_file = _get_resolved_weight_or_index_file(model_name_or_path)
    if weight_or_index_file.endswith(".index.json"):
        with open(weight_or_index_file, "r") as f:
            index = json.loads(f.read())
        if "weight_map" in index:
            index = index["weight_map"]
        checkpoint_files = sorted(list(set(index.values())))
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_checkpoint_files = {executor.submit(cached_file, model_name_or_path, f): f for f in checkpoint_files}
            checkpoint_files = [future.result() for future in concurrent.futures.as_completed(future_to_checkpoint_files)]
        #checkpoint_files = [cached_file(model_name_or_path, f) for f in checkpoint_files]
    else:
        checkpoint_files = [weight_or_index_file]

    if len(checkpoint_files) > 0:
        for i in tqdm(range(len(checkpoint_files)), desc="loading weights"):
            if not checkpoint_files[i].endswith("safetensors"):
                weights = torch.load(checkpoint_files[i], map_location="cuda", weights_only=True)
                yield weights
            else:
                if disable_mmap:# or os.name == "nt":
                    # weights = safetensors.torch.load_file(checkpoint_files[i], device="cpu")
                    # yield weights
                    with open(checkpoint_files[i], "rb") as f:
                        result = safetensors.torch.load(f.read())
                        for name, param in result.items():
                            yield name, param
                else:
                    with safetensors.safe_open(checkpoint_files[i], framework="pt", device="cpu") as f:
                        for name in f.keys():
                            yield name, f.get_tensor(name)
    else:
        raise ValueError(f"{model_name_or_path} is not a folder containing weights or safetensors")

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

ModelRegistry = {
    "Qwen3ForCausalLM": Qwen3ForCausalLM,
    "LlamaForCausalLMEagle": LlamaForCausalLMEagle,
    "LlamaForCausalLMEagle3": LlamaForCausalLMEagle3,
}

def initialize_model(model_name:str, dtype:str="auto"):
    """
    Initialize the ForwardModel with the model name or path.
    
    Args:
        model_name: The name or path of the model to load
    """
    config = transformers.AutoConfig.from_pretrained(model_name)
    if conf_dtype := getattr(config, "dtype", None):
        conf_dtype = getattr(config, "torch_dtype", None)
        assert conf_dtype is not None, "config dtype is None"
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
    architectures = config.architectures[0]
    logger.info(f"Loading model: {model_name} {architectures} with dtype: {dtype}")
    before_avail_memory, _ = torch.cuda.mem_get_info(0)
    with TorchDefaultReset(dtype, device="cuda"):
        model = ModelRegistry[architectures](config)
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