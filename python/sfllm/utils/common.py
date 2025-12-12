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
import os
from typing import Callable, List, Optional
import numpy as np
import logging
from contextlib import contextmanager
from torch.library import Library

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


def direct_register_custom_op(
    op_name: str,
    op_func: Callable,
    mutates_args: List[str],
    fake_impl: Optional[Callable] = None,
    target_lib: Optional[Library] = None,
):
    """
    `torch.library.custom_op` can have significant overhead because it
    needs to consider complicated dispatching logic. This function
    directly registers a custom op and dispatches it to the CUDA backend.
    See https://gist.github.com/youkaichao/ecbea9ec9fc79a45d2adce1784d7a9a5
    for more details.

    By default, the custom op is registered to the vLLM library. If you
    want to register it to a different library, you can pass the library
    object to the `target_lib` argument.

    IMPORTANT: the lifetime of the operator is tied to the lifetime of the
    library object. If you want to bind the operator to a different library,
    make sure the library object is alive when the operator is used.

    Note: This function will silently skip registration if the operator
    with the same name is already registered to avoid RuntimeError in
    multi-engine scenarios (e.g., VERL framework).
    """
    sglang_lib = Library("sglang", "FRAGMENT")  # noqa
    my_lib = target_lib or sglang_lib

    # Check if operator is already registered to avoid duplicate registration
    # This is important for scenarios where multiple SGLang engines run in the same process
    try:
        # Try to access the operator to see if it's already registered
        lib_name = my_lib.m.name if hasattr(my_lib.m, "name") else "sglang"
        if hasattr(torch.ops, lib_name) and hasattr(
            getattr(torch.ops, lib_name), op_name
        ):
            # Operator already exists, skip registration
            return
    except (AttributeError, RuntimeError):
        # Operator doesn't exist, proceed with registration
        pass

    if hasattr(torch.library, "infer_schema"):
        schema_str = torch.library.infer_schema(op_func, mutates_args=mutates_args)
    else:
        # for pytorch 2.4
        import torch._custom_op.impl

        schema_str = torch._custom_op.impl.infer_schema(op_func, mutates_args)

    try:
        my_lib.define(op_name + schema_str)
        my_lib.impl(op_name, op_func, "CUDA")
        if fake_impl is not None:
            my_lib._register_fake(op_name, fake_impl)
    except RuntimeError as error:
        if "Tried to register an operator" in str(error) and "multiple times" in str(
            error
        ):
            # Silently ignore duplicate registration errors
            # This can happen in multi-engine scenarios
            pass
        else:
            # Re-raise other RuntimeErrors
            raise error
    except AttributeError as error:
        # Always re-raise AttributeError as it indicates missing dependencies
        raise error


def get_bool_env_var(name: str, default: str = "false") -> bool:
    # FIXME: move your environment variable to sglang.srt.environ
    value = os.getenv(name, default)
    value = value.lower()

    truthy_values = ("true", "1")
    falsy_values = ("false", "0")

    return value in truthy_values