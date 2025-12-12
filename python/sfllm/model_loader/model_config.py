
import json
import logging
import math
import os
from enum import Enum, IntEnum, auto
from typing import Any, List, Optional, Set, Union

import torch
from transformers import PretrainedConfig
import transformers


class ModelConfig:
    def __init__(
        self,
        model_path: str,
        trust_remote_code: bool = True,
        revision: Optional[str] = None,
        context_length: Optional[int] = None,
        is_embedding: Optional[bool] = None,
        dtype: str = "auto",
        quantization: Optional[str] = None,
        is_draft_model: bool = False,
    ) -> None:
        # Parse args
        self.model_path = model_path
        self.revision = revision
        self.quantization = quantization
        self.is_draft_model = is_draft_model
        self.hf_config = transformers.AutoConfig.from_pretrained(model_path)

        conf_dtype = getattr(self.hf_config, "dtype", None)
        if conf_dtype is None:
            conf_dtype = getattr(self.hf_config, "torch_dtype", None)
            assert conf_dtype is not None, "config dtype is None"
        conf_dtype = getattr(torch, conf_dtype) if isinstance(conf_dtype, str) else conf_dtype
        self.dtype = conf_dtype if dtype == "auto" else getattr(torch, dtype)
        self.hf_config.dtype = self.dtype

