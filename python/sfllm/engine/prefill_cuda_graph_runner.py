import bisect
import logging

import torch
import tqdm

from sfllm.engine.forward_params import ForwardBatch
from sfllm.engine.schedule_batch import ScheduleBatch
from sfllm.models.interfaces import HasBatchState

logger = logging.getLogger(__name__)
# Finer buckets reduce padding for small prefills.
DEFAULT_PREFILL_CAPTURE_SIZES = (
    list(range(32, 257, 32))
    + list(range(320, 513, 64))
    + list(range(640, 2049, 128))
)


class PrefillCudaGraphRunner:
    """Full prefill graphs with fixed token capacity and dynamic sequence lengths."""

    def __init__(self, model_runner):
        self.runner = model_runner
        args = model_runner.server_args
        self.capture_sizes = sorted(set(
            args.prefill_cuda_graph_sizes or DEFAULT_PREFILL_CAPTURE_SIZES
        ))
        if self.capture_sizes[0] <= 0:
            raise ValueError("Prefill CUDA Graph sizes must be positive.")
        self.max_tokens = self.capture_sizes[-1]
        self.max_requests = args.max_running_requests
        pool = model_runner.block_memory_manager
        if not pool.can_alloc(self.max_tokens):
            raise ValueError("Insufficient KV cache for prefill CUDA Graph padding.")
        # Padding writes must never alias a live request or another padded token.
        self.scratch_locations = torch.tensor(
            pool.persist_alloc_block_from_rear(self.max_tokens),
            dtype=torch.int64, device=model_runner.device_id,
        )
        self.input_ids = torch.zeros_like(self.scratch_locations)
        self.positions = torch.zeros_like(self.input_ids)
        self.locations = self.scratch_locations.clone()
        self.qo_indptr = torch.zeros(
            self.max_requests + 1, dtype=torch.int32, device=self.input_ids.device,
        )
        self.kv_indptr = torch.zeros_like(self.qo_indptr)
        self.kv_indices = model_runner.kv_indices_buffer
        self.graphs = {}
        self.outputs = {}
        self.capture()

    def forward(self, size, batch):
        self.runner.prepare_attention(batch, size)
        return self.runner.model(self.input_ids[:size], self.positions[:size], batch)

    @torch.inference_mode()
    def capture(self):
        runner = self.runner
        pool = runner.block_memory_manager
        if isinstance(runner.model, HasBatchState):
            # Capture has no live requests and must not write their recurrent states.
            runner.model.prepare_batch_state(ScheduleBatch([], pool))
        stream = runner.compute_stream
        for size in tqdm.tqdm(list(reversed(self.capture_sizes)), desc="Capturing prefill CUDA Graphs"):
            # One query per dummy request also fits small context limits.
            self.qo_indptr.copy_(torch.arange(
                self.max_requests + 1, dtype=torch.int32, device=self.input_ids.device,
            ).clamp_max(size))
            batch = ForwardBatch(pool)
            batch.qo_indptr = self.qo_indptr
            batch.kv_indptr = self.kv_indptr
            batch.kv_indices = self.kv_indices
            batch.out_cache_loc = self.locations[:size]
            batch.max_extend_len = size
            runner.bind_cuda_graph_logits_buffer(batch, self.max_requests)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    self.forward(size, batch)
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream, pool=runner.graph_pool):
                output = self.forward(size, batch)
            self.graphs[size] = graph
            self.outputs[size] = output
        torch.cuda.current_stream().wait_stream(stream)
        logger.info("Captured prefill CUDA Graph token sizes: %s", self.capture_sizes)

    def can_run(self, batch):
        tokens = batch.input_ids.numel()
        if not (
            0 < tokens <= self.max_tokens
            and 0 < len(batch) <= self.max_requests
            and batch.forward_batch.kv_indices.numel() <= self.kv_indices.numel()
            and batch.forward_batch.custom_mask is None
        ):
            return False
        size = self.capture_sizes[bisect.bisect_left(self.capture_sizes, tokens)]
        return tokens <= 1024 or size - tokens <= 16

    def replay(self, scheduled_batch):
        batch = scheduled_batch.forward_batch
        tokens = scheduled_batch.input_ids.numel()
        requests = len(scheduled_batch)
        size = self.capture_sizes[bisect.bisect_left(self.capture_sizes, tokens)]
        self.input_ids[:tokens].copy_(scheduled_batch.input_ids)
        self.input_ids[tokens:size].zero_()
        self.positions[:tokens].copy_(scheduled_batch.position_ids)
        self.positions[tokens:size].zero_()
        self.locations[:tokens].copy_(batch.out_cache_loc)
        self.locations[tokens:size].copy_(self.scratch_locations[tokens:size])
        # Repeated offsets make every unused request empty.
        self.qo_indptr[:requests + 1].copy_(batch.qo_indptr)
        self.qo_indptr[requests + 1:].fill_(tokens)
        self.kv_indptr[:requests + 1].copy_(batch.kv_indptr)
        self.kv_indptr[requests + 1:].copy_(batch.kv_indptr[-1:])
        self.kv_indices[:batch.kv_indices.numel()].copy_(batch.kv_indices)
        self.graphs[size].replay()
        logits, hidden = self.outputs[size]
        if hidden is not None:
            hidden = [h[:tokens] for h in hidden]
        return logits[:requests], hidden
