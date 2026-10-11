import logging
from sfllm.engine.schedule_batch import ScheduleBatch,BatchResult
from sfllm.engine.forward_params import ForwardMode
from sfllm.engine.model_runner import ModelRunner
from sfllm.engine.cache.manager import CacheManager
from sfllm.models.interfaces import HasBatchState
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
        self.cache = CacheManager([self.model_runner])

    @property
    def main_mem_pool(self):
        return self.model_runner.block_memory_manager
    
    def init_capture_cudagraph(self):
        self.model_runner.init_capture_cudagraph()

    def forward(self, scheduled_batch: ScheduleBatch) -> BatchResult:
        model = self.model_runner.model
        if isinstance(model, HasBatchState):
            model.prepare_batch_state(scheduled_batch)
        self.cache.before_forward(scheduled_batch, self.model_runner.prefill_graph_runner)
        if scheduled_batch.prefix_restores and scheduled_batch.prefill_plan is None:
            return self.model_runner.forward_from_hidden(
                scheduled_batch.forward_batch.prefill_output.cached_hidden, scheduled_batch.forward_batch,
            )
        result = self.model_runner.forward(scheduled_batch)
        if scheduled_batch.forward_batch.forward_mode == ForwardMode.EXTEND:
            self.cache.after_prefill(scheduled_batch)
        return result
