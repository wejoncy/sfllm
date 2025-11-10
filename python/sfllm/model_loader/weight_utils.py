# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/model_executor/model_loader/weight_utils.py

"""Utilities for downloading and initializing model weights."""
import concurrent.futures
import fnmatch
import glob
import hashlib
import json
import logging
import os
import re
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