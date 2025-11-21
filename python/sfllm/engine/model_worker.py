import logging
from sfllm.engine.schedule_batch import ScheduleBatch,BatchResult
from sfllm.engine.model_runner import ModelRunner
from sfllm.server_args import ServerArgs

logger = logging.getLogger(__name__)

class ModelWorker:
    def __init__(self, server_args: ServerArgs):
        self.model_runner: ModelRunner = ModelRunner(server_args)
        server_args.model_config = self.model_runner.model.config
        self.detokenize = self.model_runner.detokenize
        self.tokenizer = self.model_runner.tokenizer
        self.compute_stream = self.model_runner.compute_stream

        #===init
        self.model_runner.profile_run()
        self.model_runner.init_memory_pool()

    @property
    def main_mem_pool(self):
        return self.model_runner.block_memory_manager
    
    def init_capture_cudagraph(self):
        self.model_runner.init_capture_cudagraph()

    def forward(self, scheduled_batch: ScheduleBatch) -> BatchResult:
        return self.model_runner.forward(scheduled_batch)