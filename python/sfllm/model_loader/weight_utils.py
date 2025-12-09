# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/model_executor/model_loader/weight_utils.py

"""Utilities for downloading and initializing model weights."""
import concurrent.futures
import fnmatch
import glob
import hashlib
import json
import logging
import os
import glob
import json
from pathlib import Path
import safetensors
import re
from tqdm import tqdm
import tempfile
from collections import defaultdict
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)

import filelock
import huggingface_hub.constants
import numpy as np
import safetensors.torch
import torch
from huggingface_hub import HfFileSystem, hf_hub_download, snapshot_download
from pydantic import BaseModel, ConfigDict, ValidationInfo, model_validator
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

# use system-level temp directory for file locks, so that multiple users
# can share the same lock without error.
# lock files in the temp directory will be automatically deleted when the
# system reboots, so users will not complain about annoying lock files
temp_dir = tempfile.gettempdir()

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


def get_layer_id(weight_name):
    # example weight name: model.layers.10.self_attn.qkv_proj.weight
    match = re.search(r"layers\.(\d+)\.", weight_name)
    if match:
        return int(match.group(1))
    return None


def default_weight_loader(
    param: torch.Tensor, loaded_weight: torch.Tensor, shard_id: Union[int, str] = None
) -> None:
    """Default weight loader."""
    if shard_id is not None:
        qkv_map = {"q": 0, "k": 1, "v": 2}
        if shard_id in qkv_map:
            offset = param.offset if hasattr(param, "offset") else 0
            start = 0 if shard_id=='q' else offset[qkv_map[shard_id]-1]
            end = offset[qkv_map[shard_id]]
        else:
            start = shard_id * loaded_weight.size(0)
            end = (shard_id + 1) * loaded_weight.size(0)
        param.data[start : end].copy_(loaded_weight)
        return
    try:
        if param.numel() == 1 and loaded_weight.numel() == 1:
            # Sometimes scalar values aren't considered tensors with shapes
            # so if both param and loaded_weight are a scalar,
            # "broadcast" instead of copy
            param.data.fill_(loaded_weight.item())
        else:
            assert param.size() == loaded_weight.size(), (
                f"Attempted to load weight ({loaded_weight.size()}) "
                f"into parameter ({param.size()})"
            )

            param.data.copy_(loaded_weight)
    except Exception:
        # NOTE: This exception is added for the purpose of setting breakpoint to
        # debug weight loading issues.
        raise