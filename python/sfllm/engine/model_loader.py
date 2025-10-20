import concurrent
from contextlib import ContextDecorator
import glob
import json
from pathlib import Path
import safetensors
import torch
from tqdm import tqdm
import transformers


from sfllm.models.modeling_qwen3 import Qwen3ForCausalLM

MODEL_PATH = "/root/work/gemma-3-4b-it"
MODEL_PATH = "D:\\work\\gemma-3-4b-it"

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


def _load_check_point(model, model_name_or_path, get_keys_only: bool = False):
    weight_or_index_file = _get_resolved_weight_or_index_file(model_name_or_path)
    all_keys = set()
    all_missing_keys = []
    all_unexpected_keys = []
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
                weights = torch.load(checkpoint_files[i], map_location="cpu", weights_only=True)
            else:
                weights = safetensors.torch.load_file(checkpoint_files[i], device="cpu")
            if get_keys_only:
                all_keys.update(weights.keys())
                del weights
            else:
                ret = model.load_state_dict(weights, strict=False)
                del weights
                all_missing_keys.extend(ret.missing_keys)
                all_unexpected_keys.extend(ret.unexpected_keys)
    else:
        raise ValueError(f"{model_name_or_path} is not a folder containing weights or safetensors")

    if get_keys_only:
        return all_keys
    return all_missing_keys, all_unexpected_keys

class TorchDefaultDtype(ContextDecorator):
    def __init__(self, dtype):
        if not isinstance(dtype, torch.dtype):
            raise TypeError("dtype must be a torch.dtype")
        self.new_dtype = dtype
        self._prev = None

    def __enter__(self):
        self._prev = torch.get_default_dtype()
        torch.set_default_dtype(self.new_dtype)
        return self

    def __exit__(self, exc_type, exc, tb):
        torch.set_default_dtype(self._prev)
        return False
class ForwardModel:
    def __init__(self, model_name=MODEL_PATH):
        """
        Initialize the ForwardModel with the model name or path.
        
        Args:
            model_name: The name or path of the model to load
        """
        self.model = None
        self.tokenizer = None
        
        # Load the model and tokenizer
        self.load_model(model_name)

    def load_model(self, model_name=MODEL_PATH):
        """
        Load the model and tokenizer
        
        Args:
            model_name: The name or path of the model to load
            
        Returns:
            A dictionary containing model, tokenizer, and processor
        """
        print(f"Loading model: {model_name}")
        config = transformers.AutoConfig.from_pretrained(model_name)
        with TorchDefaultDtype(config.torch_dtype):
            model = Qwen3ForCausalLM(config).cuda()
            _load_check_point(model, model_name)
        self.model = model
        self.model = self.model.eval()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, use_fast=False)

    def tokenize(self, prompt, messages=None):
        """
        Tokenize the prompt and messages for the model.
        
        Args:
            prompt: The prompt to tokenize
            messages: The messages to tokenize
            
        Returns:
            The tokenized inputs
        """
        return self.tokenizer.tokenize(prompt, messages)


    def detokenize(self, tokens):
        """
        Detokenize the tokens to get the original text.
        
        Args:
            tokens: The tokens to detokenize
            
        Returns:
            The detokenized text
        """
        return self.tokenizer.detokenize(tokens)