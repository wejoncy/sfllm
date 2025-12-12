# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Common utilities."""
from __future__ import annotations
import numpy as np
import logging
from contextlib import contextmanager

import torch

logger = logging.getLogger(__name__)

@contextmanager
def device_context(device: torch.device):
    module = torch.get_device_module(device)
    if module is not None:
        with module.device(device.index):
            yield
    else:
        raise ValueError(f"Unknown device module: {device}")

def add_prefix(name: str, prefix: str) -> str:
    """Add a weight path prefix to a module name.

    Args:
        name: base module name.
        prefix: weight prefix str to added to the front of `name` concatenated with `.`.

    Returns:
        The string `prefix.name` if prefix is non-empty, otherwise just `name`.
    """
    return name if not prefix else f"{prefix}.{name}"

def get_tensor_model_parallel_rank():
    """Get the tensor model parallel rank."""
    # Placeholder implementation
    return 0
def get_tensor_model_parallel_world_size():
    """Get the tensor model parallel world size."""
    # Placeholder implementation
    return 1
def tensor_model_parallel_all_gather(tensor: torch.Tensor) -> torch.Tensor:
    """All-gather the tensor across tensor model parallel ranks."""
    # Placeholder implementation
    return tensor

def tensor_model_parallel_all_reduce(tensor: torch.Tensor) -> torch.Tensor:
    """All-reduce the tensor across tensor model parallel ranks."""
    # Placeholder implementation
    return tensor

def make_layers_non_pp(
    num_hidden_layers: int,
    layer_fn,
    prefix:str="",
) -> torch.nn.ModuleList:

    layers = torch.nn.ModuleList(
            (
                layer_fn(idx=idx, prefix=f"{prefix}.{idx}")
                for idx in range(num_hidden_layers)
            )
        )
    return layers,0, num_hidden_layers
