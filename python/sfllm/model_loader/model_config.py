
import json
import logging
import math
import os
from enum import Enum, IntEnum, auto
from typing import Any, List, Optional, Set, Union

import torch
from transformers import PretrainedConfig
import transformers


def get_pool_index_layers(config: PretrainedConfig) -> List[int]:
    """Return KV-bearing text-model layer IDs in execution order."""
    layer_types = vars(config).get("layer_types")
    if layer_types is None:
        return list(range(config.num_hidden_layers))
    return [
        i for i, layer_type in enumerate(layer_types)
        if layer_type in ("full_attention", "sliding_attention")
    ]


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
        raw_config, _ = PretrainedConfig.get_config_dict(model_path)
        if raw_config.get("speculators_model_type") == "dflash2":
            # Speculators counts hidden states from the embedding output (0).
            dflash_config = dict(raw_config)
            dflash_config["target_layer_ids"] = [
                layer_id - 1 for layer_id in raw_config["aux_hidden_state_layer_ids"]
            ]
            self.hf_config = transformers.AutoConfig.for_model(
                **raw_config["transformer_layer_config"],
                architectures=raw_config["architectures"],
                dtype=raw_config["dtype"],
                dflash_config=dflash_config,
            )
        else:
            self.hf_config = transformers.AutoConfig.from_pretrained(model_path)

        conf_dtype = self.hf_config.dtype or self.hf_config.get_text_config().dtype
        assert conf_dtype is not None, "config dtype is None"
        dtypes = {"half": torch.float16, "float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        conf_dtype = dtypes[conf_dtype] if isinstance(conf_dtype, str) else conf_dtype
        self.dtype = conf_dtype if dtype == "auto" else dtypes[dtype]
        self.hf_config.dtype = self.dtype
