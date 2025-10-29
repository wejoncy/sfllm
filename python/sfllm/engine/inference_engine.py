
"""
nsys profile  --force-overwrite=true  -o baseline-report  --trace=cuda,nvtx,osrt,cudnn --cuda-graph-trace=node  python python/sfllm/engine/inference_engine.py
"""
import logging
import torch
import queue
from typing import Dict, Any, List, Tuple, Generator

from sfllm.engine.model_runner import ModelRunner
from sfllm.engine.scheduler import Scheduler
from sfllm.engine.sampling_params import SamplingParams
from sfllm.engine.sequence import RequestSequence, SequenceStatus, AbortSequence
from sfllm.engine.shedule_batch import ScheduleBatch
from sfllm.server_args import ServerArgs
from sfllm.utils.nutils import configure_logger,resolve_future_token_ids

logger = logging.getLogger(__name__)
class InferenceEngine:
    """Worker that processes inference requests from a queue."""
    
    def __init__(self, server_args: ServerArgs):
        """
        Initialize the inference worker.
        """
        configure_logger(server_args)
        self.model_runner = ModelRunner(server_args)
        self.model_runner.profile_run()
        self.server_args = server_args
        self.running = False
        self.scheduler = Scheduler(server_args)
        self.output_batch_queue = queue.Queue()
        self.model_runner.init_capture_graph(self.scheduler.block_memory_manager)
        self.enable_overlap = not server_args.disable_overlap

    def post_forward(self, schedule_batch: ScheduleBatch, token_ids: List[int], failed_sequences: List[RequestSequence]) -> None:
        """Post-process the model outputs and update the sequences."""
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
                sequence.new_tokens = token_ids[idx: idx + 1]
                sequence.generated_tokens[0] = token_ids[idx]
                sequence.tokens.extend(sequence.new_tokens)

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
            sequence.init(self.model_runner.model.tokenizer)
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
        token_ids = []
        if not new_batch.empty():
            new_batch.prepare_inputs()
            new_batch.prepare_sample()
            token_ids = self.model_runner.forward(new_batch).tolist()
        self.post_forward(new_batch, token_ids, failed_sequences)
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
        future_limit = 1024
        future_tokenid_bufs = torch.empty(future_limit, device="cuda", dtype=torch.int64)
        import time
        compute_stream = self.model_runner.compute_stream
        copy_in_stream = self.model_runner.copy_in_stream
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
                with torch.cuda.stream(copy_in_stream):
                    cur_batch.prepare_inputs()
                    cur_batch.prepare_sample()
                with torch.cuda.stream(compute_stream):
                    compute_stream.wait_stream(copy_in_stream)
                    if cur_batch.forward_metadata.is_decode():
                        resolve_future_token_ids(cur_batch.input_ids, future_tokenid_bufs)
                    model_output = self.model_runner.forward(cur_batch)
                    fake_tokenid_indices = cur_batch.fake_tokenid_indices(future_limit)
                    assert model_output.shape[-1] == len(cur_batch)
                    cur_batch.add_placeholder_token(future_limit)
                    future_tokenid_bufs[fake_tokenid_indices] = model_output
                    cur_batch.next_token_ids = model_output.to("cpu", non_blocking=True)
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

    def response(self, new_batch: ScheduleBatch, stream: bool) -> List[Dict[str, Any]]:
        seq_outputs = {}
        for sequence in new_batch:
            if stream:
                if sequence.status == SequenceStatus.RUNNING:
                    new_token = sequence.generated_tokens
                    generated_text = self.model_runner.detokenize(
                        new_token,
                    )
                    seq_outputs[sequence.sequence_id] = {"prompt": sequence.prompt, "text": generated_text}
                    yield seq_outputs
            elif not sequence.status.is_active():
                sequence.generated_text = self.model_runner.detokenize(
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
