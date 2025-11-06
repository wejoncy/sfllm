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
        self.model_runner: ModelRunner = ModelRunner(server_args)
        server_args.model_config = self.model_runner.model.config
        self.detokenize = self.model_runner.detokenize
        self.tokenizer = self.model_runner.tokenizer

        #===init
        self.model_runner.profile_run()
        self.model_runner.init_memory_pool()

    @property
    def main_mem_pool(self):
        return self.model_runner.block_memory_manager
    
    def init_capture_graph(self):
        self.model_runner.init_capture_graph()

    def forward(self, scheduled_batch: ScheduleBatch) -> BatchResult:
        return self.model_runner.forward(scheduled_batch)