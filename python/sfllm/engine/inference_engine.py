
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
from sfllm.kernels.triton_utils import split_lastdim_async
from sfllm.kernels.triton_utils import move_neg1_to_tail, compact_accepted_tokens,prune_kv_indices,split_firstdim_async

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
        batch_result: BatchResult,
        failed_sequences: List[RequestSequence],
    ) -> None:
        """Post-process the model outputs and update the sequences."""
        if not isinstance(batch_result, BatchResult):
            assert False, "Only BatchResult is supported now."

        self.scheduler.metrics.update_spec_metrics(batch_result.spec_info)
        self.scheduler.metrics.log_prefill_metrics(schedule_batch)
        self.scheduler.metrics.log_decode_metrics(schedule_batch)
        token_ids = batch_result.next_token_ids.tolist()
        if self.is_spec_algo:
            # TODO parallel decoding with speculative decoding, multitoken would be decoded in a single step
            accept_length_cpu = batch_result.spec_info.accept_length_cpu.clamp(min=0)
            cum_token_cnts = (accept_length_cpu+1).cumsum(dim=0).tolist()
            cum_token_cnts = [0] + cum_token_cnts
        for idx, sequence in enumerate(schedule_batch):
            if self.enable_overlap:
                if sequence.status.is_active():
                    assert sequence.tokens[sequence.last_generated_token_pos] < 0, (
                        "Last token should be placeholder"
                    )
                    assert token_ids[idx] >= 0, "Generated token should be valid"
                    if self.is_spec_algo:
                        draft_token_steps = self.server_args.speculative_num_steps+1
                        new_token = token_ids[cum_token_cnts[idx]: cum_token_cnts[idx + 1]]
                        last_pos = sequence.last_generated_token_pos
                        sequence.tokens[last_pos:last_pos+len(new_token)] = new_token
                        sequence.tokens[last_pos+len(new_token): last_pos+draft_token_steps] = []
                        sequence.generated_tokens = new_token
                        sequence.last_generated_token_pos += len(sequence.generated_tokens)
                    else:
                        sequence.tokens[sequence.last_generated_token_pos] = token_ids[idx]
                        sequence.generated_tokens[0] = token_ids[idx]
                        sequence.last_generated_token_pos += 1
            else:
                if self.is_spec_algo:
                    sequence.new_tokens = token_ids[cum_token_cnts[idx]: cum_token_cnts[idx + 1]]
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
                neg_idx = len(sequence.out_cache_loc)
                while neg_idx > 0 and sequence.out_cache_loc[neg_idx-1] < 0:
                    neg_idx -= 1
                sequence.out_cache_loc = sequence.out_cache_loc[:neg_idx]
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

    @torch.inference_mode()
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
            num_draft_tokens = self.server_args.speculative_num_draft_tokens
            draft_token_steps = self.server_args.speculative_num_steps+1
            future_token_stride = draft_token_steps
            target_mem_pool = self.scheduler.mem_pool
            draft_mem_pool = self.scheduler.draft_memory_pool
            hidden_size = self.server_args.model_config.hidden_size*3
            dtype = self.model_worker.dtype
            hidden_states_buf = torch.empty((128, draft_token_steps, hidden_size), device=device_id, dtype=dtype)

            # target_overlap_pool = torch.tensor(target_mem_pool.alloc_block(4000), dtype=torch.int64, device=device_id)
            # draft_overlap_pool = torch.tensor(draft_mem_pool.alloc_block(4000), dtype=torch.int64, device=device_id)
        
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
                if self.is_spec_algo:
                    cur_batch.overlap_affiliated = (hidden_states_buf, compute_stream)
                with torch.cuda.stream(scheduler_stream):
                    cur_batch.prepare_inputs(is_overlap=True)
                    cur_batch.prepare_sample()
                    # Adjust the cache for speculative decoding
                    # why we need to do this? 1. we don't know how many tokens will pass the verify steps
                    # 2. the previous one affect the draft extend tokens nums
                    # so we have to alloc num_draft_tokens for verify and draft_token_steps tokens for draft
                    # however, this induce a bit impact for the next scheduling, because we don't how many tokens we have 
                    # unlike the non-spec one, there is only one token each step
                    # kv_indptr position_ids seq_lens  mask_indptr, position_ids_extend
                    # kv_indices_mtd kv_indptr 
                    # may verified_id, out_cache_loc
                with torch.cuda.stream(compute_stream):
                    compute_stream.wait_stream(scheduler_stream)
                    if self.is_spec_algo and cur_batch.forward_batch.is_decode():
                        accept_length = cur_batch.spec_info.accept_length.clamp(min=0)+1
                        accept_length_raw = cur_batch.spec_info.accept_length+1
                        extra_length = draft_token_steps - accept_length
                        cum_extra_length = extra_length.cumsum(dim=0)
                        cur_batch.position_ids -= extra_length
                        cur_batch.forward_batch.seq_lens -= extra_length
                        seq_mask_len = num_draft_tokens * (cur_batch.forward_batch.seq_lens + num_draft_tokens)
                        cum_seq_mask_len = torch.cumsum(seq_mask_len, dim=0, dtype=torch.int32)
                        cur_batch.forward_batch.mask_indptr[1:] = cum_seq_mask_len
                        cur_batch.forward_batch_spec.kv_indices = cur_batch.forward_batch_spec.kv_indices_mtd
                        #####
                        x = cur_batch.forward_batch_spec.kv_indices_mtd
                        if cur_batch.spec_info.out_cache_loc is not None:
                            b = cur_batch.forward_batch_spec.kv_indptr
                        else:
                            b = cur_batch.forward_batch_spec.kv_indptr[1:]+torch.arange(1,1+len(cur_batch), device=device_id)
                            b = torch.cat([torch.zeros((1,), dtype=torch.long, device=device_id), b], dim=0)
                        # x1=x.clone()
                        compact_accepted_tokens(x, b, cur_batch.forward_batch.seq_lens)

                        # row_ids = torch.arange(x.shape[0], device=device_id)
                        # seg_ids = torch.searchsorted(b, row_ids, right=True) - 1  # [N]
                        # start = b[seg_ids+1]          # [N]
                        # lower = start - (draft_token_steps-accept_length_raw[seg_ids])           # [N]
                        # upper = start               # [N]
                        # mask = (row_ids >= lower) & (row_ids < upper)
                        # x[mask] = -1
                        # x[:] = move_neg1_to_tail(x)
                        # cur_batch.forward_batch_spec.kv_indices_mtd -= extra_length # can't handle it here, may generate_kv_indices_kernel
                        cur_batch.forward_batch_spec.kv_indptr[1:] -= cum_extra_length
                        if cur_batch.spec_info.out_cache_loc is not None:
                            # even we don't know the accept index yet, we can adjust it in GPU async
                            # would it affect when this is in the scheduler stream?
                            # first token is the root token
                            # TODO check it, we uncheck the root token here as this is known to be accepted
                            resolve_out_cache_loc = cur_batch.spec_info.out_cache_loc
                            # resolve_out_cache_loc[0] = -1
                            # resolve_out_cache_loc[accept_length.cumsum(dim=0)[:-1]] = -1
                            # resolve_out_cache_loc = move_neg1_to_tail(resolve_out_cache_loc)[:draft_token_steps-len(cur_batch)]
                            x = cur_batch.forward_batch.kv_indices
                            prune_kv_indices(resolve_out_cache_loc,x,
                                cur_batch.forward_batch.kv_indptr, accept_length)
                            # kv_indptr = cur_batch.forward_batch.kv_indptr[1:]
                            # kv_indices = cur_batch.forward_batch.kv_indices
                            # cum_accept_length = torch.cat([torch.tensor([0], device=device_id), accept_length.cumsum(dim=0)], dim=0)
                            # for i in range(len(cur_batch)):
                            #     L = accept_length[i] - 1
                            #     dst_end = kv_indptr[i]
                            #     dst_start = dst_end - L
                            #     kv_indices[dst_start:dst_end] = resolve_out_cache_loc[cum_accept_length[i]+1:cum_accept_length[i+1]]
                            # kv_indices[:] = move_neg1_to_tail(kv_indices)
                            # assert torch.allclose(x, kv_indices)

                            # cur_batch.forward_batch.kv_indices[:draft_token_steps-len(cur_batch)] = resolve_out_cache_loc
                            # cur_batch.forward_batch.kv_indices[:] = move_neg1_to_tail(cur_batch.forward_batch.kv_indices)
                            # cur_batch.forward_batch.kv_indices = cur_batch.forward_batch.kv_indices[:len(cur_batch)*num_draft_tokens]
                        cur_batch.forward_batch.kv_indptr[1:] -= cum_extra_length

                        x = cur_batch.forward_batch_spec.position_ids_extend
                        b = torch.arange(0, (len(cur_batch)+1)*draft_token_steps, draft_token_steps, device=device_id)
                        compact_accepted_tokens(x, b, accept_length, fill_value=0)
                        x = cur_batch.forward_batch_spec.out_cache_loc
                        compact_accepted_tokens(x, b, accept_length, fill_value=0)
                        if cur_batch.spec_info.verified_id.shape[0] != cur_batch.forward_batch_spec.position_ids_extend.shape[0]:
                            token_len = cur_batch.spec_info.verified_id.shape[0]
                            cur_batch.forward_batch_spec.position_ids_extend = cur_batch.forward_batch_spec.position_ids_extend[:token_len]
                            cur_batch.forward_batch_spec.out_cache_loc = cur_batch.forward_batch_spec.out_cache_loc[:token_len]


                        # cur_batch.spec_info.verified_id = torch.cat([sequence.verified_id for sequence in cur_batch], dim=0)

                with torch.cuda.stream(compute_stream):
                    compute_stream.wait_stream(scheduler_stream)
                    if cur_batch.forward_batch.is_decode():
                        resolve_future_token_ids(cur_batch.input_ids, future_tokenid_bufs)
                    model_output:BatchResult = self.model_worker.forward(cur_batch)
                    fake_tokenid_indices = cur_batch.fake_tokenid_indices(future_limit, future_token_stride)
                    cur_batch.add_placeholder_token(future_limit, future_token_stride)
                    if not self.is_spec_algo:
                        assert model_output.next_token_ids.shape[-1] == len(cur_batch)
                        future_tokenid_bufs[fake_tokenid_indices] = model_output.next_token_ids
                    else:
                        raw_accept_length = model_output.spec_info.accept_length
                        update_accept_length = raw_accept_length+1
                        next_token_ids = model_output.next_token_ids
                        # assert fake_tokenid_indices[:-1]-fake_tokenid_indices[1:] == -1
                        # must !!!!!!!!!!
                        future_token_out_buffer = future_tokenid_bufs[1:fake_tokenid_indices.shape[0]+1].view(len(cur_batch), -1)
                        split_lastdim_async(next_token_ids, update_accept_length, future_token_out_buffer)

                        # if cur_batch.forward_batch.is_decode():
                        #     # cur_batch.forward_batch.out_cache_loc
                        #     # cur_batch.forward_batch_spec.out_cache_loc
                        #     split_firstdim_async(model_output.spec_info.hidden_states, update_accept_length, hidden_states_buf)
                        #     # cur_batch.spec_info.out_cache_loc = cur_batch.forward_batch.out_cache_loc

                        #     # even we don't know the accept index yet, we can adjust it in GPU async
                        #     # would it affect when this is in the scheduler stream?                            
                        cum_update_accept_length = torch.cat([torch.zeros((1,), dtype=torch.long, device=device_id), update_accept_length.cumsum(dim=0)], dim=0)
                        for idx, seq in enumerate(cur_batch):
                            seq.accept_length = raw_accept_length[idx:idx+1]
                            if cur_batch.forward_batch.is_decode():
                                seq.verified_id = future_token_out_buffer[idx]
                                # sync the below code
                                # seq.verified_id = next_token_ids[cum_update_accept_length[idx]: cum_update_accept_length[idx + 1]]
                                # seq.hidden_states = model_output.spec_info.hidden_states[cum_update_accept_length[idx]: cum_update_accept_length[idx + 1]]
                                seq.hidden_states = (model_output.spec_info.hidden_states, cum_update_accept_length[idx:idx+2])
                                seq.out_cache_loc_lazy = cur_batch.forward_batch.out_cache_loc.view(len(cur_batch), num_draft_tokens)[idx]
                            else:
                                # the first token generated from prefill
                                seq.verified_id = model_output.spec_info.verified_id[idx]
                                seq.hidden_states = (model_output.spec_info.hidden_states[idx:idx+1], torch.arange(0,2, device=device_id))
                        # replace the spec_info with cpu version
                        async_spec_info = model_output.spec_info
                        model_output.spec_info = model_output.spec_info.raw_new()
                        model_output.spec_info.accept_length_cpu = raw_accept_length.to("cpu", non_blocking=True)
                        # if async_spec_info.accept_index is not None:
                        #     model_output.spec_info.accept_index = async_spec_info.accept_index.to("cpu", non_blocking=True)
                        if async_spec_info.out_cache_loc is not None:
                            model_output.spec_info.out_cache_loc = async_spec_info.out_cache_loc.to("cpu", non_blocking=True)
                        model_output.out_cache_loc = cur_batch.forward_batch.out_cache_loc.to("cpu", non_blocking=True)

                    model_output.next_token_ids = model_output.next_token_ids.to("cpu", non_blocking=True)
                    cur_batch.model_output = model_output
                    cur_batch.copy_done = torch.cuda.Event()
                    cur_batch.copy_done.record(compute_stream)

            if not last_batch.empty():
                copy_done, model_output = last_batch.copy_done, last_batch.model_output
                copy_done.synchronize()
                if self.is_spec_algo:
                    self.model_worker.spec_postprocess(last_batch, model_output, async_overlap=True)
                self.post_forward(last_batch, model_output, failed_sequences)
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
        
        if not stream:
            self.event_loop_overlap()
            while not self.output_batch_queue.empty():
                new_batch = self.output_batch_queue.get()
                yield from self.response(new_batch, stream=stream)
            return

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
