
"""
nsys profile  --force-overwrite=true  -o baseline-report  --trace=cuda,nvtx,osrt,cudnn --cuda-graph-trace=node  python python/sfllm/engine/inference_engine.py
HIP_TRACE_API=1 
export HSA_TOOLS_LIB=/opt/rocm/lib/librocm-debug-agent.so.2
export HSA_ENABLE_DEBUG=1 is useful for ROCm error tracing
"""
import logging
import torch
import queue
from typing import Dict, Any, List, Tuple, Generator, Union

from sfllm.engine.model_worker import ModelWorker
from sfllm.spec_decoding.eagle_worker import EagleWorker
from sfllm.engine.scheduler import Scheduler
from sfllm.engine.sampling_params import SamplingParams
from sfllm.engine.sequence import RequestSequence, SequenceStatus, AbortSequence
from sfllm.engine.schedule_batch import ScheduleBatch, BatchResult
from sfllm.server_args import ServerArgs
from sfllm.utils.nutils import configure_logger,resolve_future_token_ids
from sfllm.kernels.triton_utils import split_tokens_async

logger = logging.getLogger(__name__)
class InferenceEngine:
    """Worker that processes inference requests from a queue."""
    
    def __init__(self, server_args: ServerArgs):
        """
        Initialize the inference worker.
        """
        configure_logger(server_args)
        self.model_worker = ModelWorker(server_args) if server_args.speculative_algorithm != "eagle3" else EagleWorker(server_args)
        self.server_args = server_args
        self.running = False
        self.scheduler = Scheduler(server_args, self.model_worker)
        self.output_batch_queue = queue.Queue()
        self.model_worker.init_capture_cudagraph()
        self.enable_overlap = not server_args.disable_overlap

    @property
    def is_spec_algo(self) -> bool:
        return self.server_args.speculative_algorithm is not None

    def post_forward(
        self,
        schedule_batch: ScheduleBatch,
        token_ids: Union[List[int], BatchResult],
        failed_sequences: List[RequestSequence],
    ) -> None:
        """Post-process the model outputs and update the sequences."""
        if isinstance(token_ids, BatchResult):
            batch_result = token_ids
            token_ids = batch_result.next_token_ids.tolist()
        for idx, sequence in enumerate(schedule_batch):
            if self.enable_overlap:
                if sequence.status.is_active():
                    assert sequence.tokens[sequence.last_generated_token_pos] < 0, (
                        "Last token should be placeholder"
                    )
                    assert token_ids[idx] >= 0, "Generated token should be valid"
                    sequence.tokens[sequence.last_generated_token_pos] = token_ids[idx]
                    sequence.generated_tokens[0] = token_ids[idx]
                    sequence.last_generated_token_pos += 1
            else:
                if self.is_spec_algo:
                    #TODO parallel decoding with speculative decoding, multitoken would be decoded in a single step
                    pos_offset = (batch_result.spec_info.accept_length_cpu+1).cumsum(dim=0).tolist()
                    pos_offset = [0] + pos_offset
                    sequence.new_tokens = token_ids[pos_offset[idx]: pos_offset[idx + 1]]
                    sequence.generated_tokens = sequence.new_tokens.copy()
                    sequence.tokens.extend(sequence.new_tokens)
                    sequence.last_generated_token_pos += len(sequence.generated_tokens)
                else:
                    sequence.new_tokens = token_ids[idx: idx + 1]
                    sequence.generated_tokens[0] = token_ids[idx]
                    sequence.tokens.extend(sequence.new_tokens)
                    sequence.last_generated_token_pos += 1

            if not sequence.is_done():
                sequence.status = SequenceStatus.RUNNING
                if not self.enable_overlap:
                    self.scheduler.running_queue.put(sequence)
            elif not sequence.status.is_active():
                # a sequence may be calculted one more step after completed
                pass
            else:
                self.scheduler.free_sequence_resources(sequence)
                sequence.status = SequenceStatus.COMPLETED
                # abort request may have req_id added after completed, so we need to check again
                sid = next(iter(self.scheduler.abort_requests), None)
                # 100 should be safe to set as buffer
                if sid is not None and sid + 100 < sequence.sequence_id:
                    self.scheduler.abort_requests.remove(sid)


    def new_request(self, prompt: str|Tuple[str, List[int]], sampling_params: SamplingParams) -> int:
        if isinstance(prompt, str):
            sequence = RequestSequence(prompt, sampling_params)
            sequence.init(self.model_worker.tokenizer)
        else:
            assert isinstance(prompt, tuple), "Prompt must be a string or a tuple of (str, List[int])"
            sequence = RequestSequence(prompt[0], sampling_params, input_ids=prompt[1])

        return sequence

    def add_request(
        self,
        prompt: str | Tuple[str, List[int]] | RequestSequence,
        sampling_params: SamplingParams = SamplingParams(),
    ) -> int:
        """Add a new inference request to the queue."""
        if isinstance(prompt, RequestSequence):
            sequence = prompt
        elif isinstance(prompt, AbortSequence):
            self.scheduler.add_abort_request(prompt.sequence_id)
            return prompt.sequence_id
        else:
            sequence = self.new_request(prompt, sampling_params)
        self.scheduler.add_request(sequence)
        return sequence.sequence_id
    
    def step(self):
        """Process a single inference request."""
        new_batch, failed_sequences = self.scheduler.get_next_batch()
        batch_out = []
        if not new_batch.empty():
            new_batch.prepare_inputs()
            new_batch.prepare_sample()
            batch_out = self.model_worker.forward(new_batch)
            if self.is_spec_algo:
                new_batch = self.model_worker.spec_postprocess(new_batch, batch_out)
        self.post_forward(new_batch, batch_out, failed_sequences)
        new_batch.extend(failed_sequences)
        return new_batch

    def step_overlap(self, timeout: float=None) -> Generator[Dict[str, Any], Any, Any]:
        try:
            new_batch = self.output_batch_queue.get(timeout=timeout)
            return new_batch
        except queue.Empty:
            return []

    def event_loop_overlap(self, event=None):
        """Process a single inference request with overlap."""
        logger.info("Inference engine event loop started.============")
        assert self.enable_overlap, "Overlap must be enabled for event loop."
        failed_sequences = []
        cur_batch = None
        last_batch = ScheduleBatch([], None)
        future_limit = 1024*10
        future_token_stride = 1
        device_id = torch.device("cuda:0")
        if self.is_spec_algo:
            draft_token_len = self.server_args.speculative_num_draft_tokens
            draft_token_steps = self.server_args.speculative_num_steps+1
            future_token_stride = draft_token_steps
            target_mem_pool = self.scheduler.mem_pool
            draft_mem_pool = self.scheduler.draft_memory_pool
            target_overlap_pool = torch.Tensor(target_mem_pool.alloc_block(4096), dtype=torch.int64, device=device_id)
            draft_overlap_pool = torch.Tensor(draft_mem_pool.alloc_block(4096), dtype=torch.int64, device=device_id)
        
        future_tokenid_bufs = torch.empty(future_limit, device=device_id, dtype=torch.int64)
        import time
        compute_stream = self.model_worker.compute_stream
        scheduler_stream = torch.cuda.Stream(device=device_id)
        def notified():
            if event is not None:
                return event.is_set()
            return False
        while not notified():
            new_batch, failed_seq = self.scheduler.get_next_batch_async(last_batch=last_batch)
            failed_sequences.extend(failed_seq)
            if new_batch.empty() and last_batch.empty():
                if event is None:
                    break
                time.sleep(0.1)
                continue
            cur_batch = new_batch

            if not cur_batch.empty():
                with torch.cuda.stream(scheduler_stream):
                    cur_batch.prepare_inputs()
                    cur_batch.prepare_sample()
                # Adjust the cache for speculative decoding
                # why we need to do this? 1. we don't know how many tokens will pass the verify steps
                # 2. the previous one affect the draft extend tokens nums
                # so we have to alloc draft_token_len for verify and draft_token_steps tokens for draft
                # however, this induce a bit impact for the next scheduling, because we don't how many tokens we have 
                # unlike the non-spec one, there is only one token each step
                if self.is_spec_algo and cur_batch.forward_batch.is_decode():
                    for sequence in cur_batch:
                        sequence.out_cache_loc = sequence.out_cache_loc[:-draft_token_len]
                        sequence.out_cache_loc_spec = sequence.out_cache_loc_spec[:-draft_token_steps]
                with torch.cuda.stream(compute_stream):
                    compute_stream.wait_stream(scheduler_stream)
                    if cur_batch.forward_batch.is_decode():
                        resolve_future_token_ids(cur_batch.input_ids, future_tokenid_bufs)
                    model_output = self.model_worker.forward(cur_batch)
                    fake_tokenid_indices = cur_batch.fake_tokenid_indices(future_limit, future_token_stride)
                    cur_batch.add_placeholder_token(future_limit, future_token_stride)
                    if not self.is_spec_algo:
                        assert model_output.next_token_ids.shape[-1] == len(cur_batch)
                        future_tokenid_bufs[fake_tokenid_indices] = model_output.next_token_ids
                    else:
                        accept_length = model_output.spec_info.accept_length+1
                        next_token_ids = model_output.next_token_ids
                        future_token_out_buffer = future_tokenid_bufs[fake_tokenid_indices].view(len(cur_batch), -1)
                        split_tokens_async(next_token_ids, accept_length, future_token_out_buffer)
                        model_output.spec_info.accept_length_cpu = model_output.spec_info.accept_length.to("cpu", non_blocking=True)
                        cur_batch.future_spec_info = model_output.spec_info
                    cur_batch.next_token_ids = model_output.next_token_ids.to("cpu", non_blocking=True)
                    cur_batch.copy_done = torch.cuda.Event()
                    cur_batch.copy_done.record(compute_stream)

            if not last_batch.empty():
                copy_done, next_token_ids = last_batch.copy_done, last_batch.next_token_ids
                copy_done.synchronize()
                token_ids = next_token_ids.tolist()
                self.post_forward(last_batch, token_ids, failed_sequences)
                self.output_batch_queue.put(last_batch)

            last_batch = cur_batch


        logger.info("Inference engine event loop exited.")

    def response(self, new_batch: ScheduleBatch, stream: bool) -> Generator[Dict[str, Any], Any, Any]:
        seq_outputs = {}
        for sequence in new_batch:
            if stream:
                if sequence.status == SequenceStatus.RUNNING:
                    new_token = sequence.generated_tokens
                    generated_text = self.model_worker.detokenize(
                        new_token,
                    )
                    seq_outputs[sequence.sequence_id] = {"prompt": sequence.prompt, "text": generated_text}
                    yield seq_outputs
            elif not sequence.status.is_active():
                sequence.generated_text = self.model_worker.detokenize(
                    sequence.tokens[sequence.prompt_token_len : sequence.last_generated_token_pos],
                )
                yield {
                    sequence.sequence_id: {
                        "prompt": sequence.prompt,
                        "text": sequence.generated_text,
                    }
                }

    def generate(self, prompt: List[str]|str, sampling_params: SamplingParams,
                 stream:bool=False) -> Generator[Dict[str, Any], Any, Any]:
        """Generate text for inference requests."""
        if self.enable_overlap:
            yield from self.generate_overlap(prompt, sampling_params, stream=stream)
            return

        if isinstance(prompt, str):
            prompt = [prompt]
        for p in prompt:
            self.add_request(p, sampling_params)

        while not self.scheduler.is_done():
            yield from self.response(self.step(), stream=stream)
            

    def generate_overlap(self, prompt: List[str]|str, sampling_params: SamplingParams,
                 stream:bool=False) -> Generator[Dict[str, Any], Any, Any]:
        """Generate text for inference requests with overlap."""

        if isinstance(prompt, str):
            prompt = [prompt]
        for p in prompt:
            self.add_request(p, sampling_params)

        import threading
        thread = threading.Thread(target=self.event_loop_overlap)
        thread.start()
        
        while thread.is_alive() or not self.output_batch_queue.empty():
            new_batch = self.step_overlap(timeout=1)
            if len(new_batch) == 0:
                continue
            yield from self.response(new_batch, stream=stream)
        
        thread.join()

    def start_profiler(self):
        """Start profiling the inference engine."""
        from sfllm.utils.profiler import SchedulerProfilerMixin
        self.profiler = SchedulerProfilerMixin()
        self.profiler.start_profiler()
    
    def stop_profiler(self):
        """Stop profiling the inference engine."""
        self.profiler.stop_profiler()
