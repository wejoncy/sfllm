import logging
import torch
import bisect
import tqdm
from typing import Dict, List, Any
import transformers

from sfllm.model_loader.model_loader import initialize_model
from sfllm.engine.shedule_batch import ScheduleBatch,BatchResult
from sfllm.engine.forward_params import ForwardMode, ForwardBatch
from sfllm.engine.model_runner import ModelRunner
from sfllm.layers.sampler import Sampler, SamplingBatchInfo
from sfllm.server_args import ServerArgs
from sfllm.utils.nutils import DEFAULT_CUDA_GRAPH_BATCH_SIZES, MAX_PROCESSED_TOKENS

logger = logging.getLogger(__name__)

class ModelWorker:
    def __init__(self, server_args: ServerArgs):
        self.model_runner = ModelRunner(server_args)
        server_args.model_config = self.model_runner.config
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(server_args.model_path)

        #===init
        self.model_runner.profile_run()
        self.model_runner.init_memory_pool()

    def tokenize(self, prompt):
        return self.tokenizer.encode(prompt)

    def detokenize(self, tokens):
        return self.tokenizer.decode(
            tokens, skip_special_tokens=True, spaces_between_special_tokens=True
        )
