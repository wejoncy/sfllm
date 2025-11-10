import time
import queue
import logging
from typing import List, Tuple


from sfllm.engine.sequence import RequestSequence, SequenceStatus
from sfllm.engine.schedule_batch import ScheduleBatch
from sfllm.engine.memory_pool import BlockMemoryManager
from sfllm.spec_decoding.spec_common import SpeculativeAlgorithm
from sfllm.engine.model_worker import ModelWorker


logger = logging.getLogger(__name__)
class RunningMetrics:
    def __init__(
        self,
        waiting_queue: queue.Queue,
        running_queue: queue.Queue,
        block_memory_manager: BlockMemoryManager,
        server_args=None,
    ):
        self.tokens_generated = 0
        self.total_forward_tokens = 0
        self.cum_forward_tokens = 0
        self.total_accept_tokens = 0
        self.cum_accept_tokens = 0
        self.prefill_tokens = 0
        self.last_prefill_refresh_time = time.perf_counter()
        self.last_refresh_time = time.perf_counter()
        self.waiting_queue = waiting_queue
        self.running_queue = running_queue
        self.block_memory_manager = block_memory_manager
        self.server_args = server_args

    def log_prefill_metrics(self, cur_prefill_tokens: int):
        current_time = time.perf_counter()
        elapsed = current_time - self.last_prefill_refresh_time
        log_interval = 1.0  # seconds
        if self.prefill_tokens == 0:
            self.prefill_tokens = cur_prefill_tokens
            self.last_prefill_refresh_time = current_time
            return
        if elapsed > log_interval or cur_prefill_tokens == 0:
            msg = f"Prefill batch. #prefill_tokens: {self.prefill_tokens}. "
            msg += (
                f"Prefill throughput (token/s): {self.prefill_tokens / elapsed:.2f}, "
                f"#queue-req p+d: {self.waiting_queue.qsize()}+{self.running_queue.qsize()}, "
            )
            logger.info(msg)
            self.prefill_tokens = 0
            self.last_prefill_refresh_time = current_time

        self.prefill_tokens += cur_prefill_tokens

    def log_decode_metrics(self, running_batch: ScheduleBatch, is_prefill: bool=False):
        seq_group = running_batch.sequences
        decode_tokens = len(seq_group)
        prefill_mask = int(not is_prefill)
        if running_batch.spec_info is not None:
            # each sequence at least generate one token
            accept_tokens = running_batch.spec_info.accept_length_cpu.sum().item()
            decode_tokens += max(0, accept_tokens)
            self.total_accept_tokens += decode_tokens
            self.total_forward_tokens += len(seq_group)
            self.cum_accept_tokens += decode_tokens
            self.cum_forward_tokens += len(seq_group)

        current_time = time.perf_counter()
        elapsed = current_time - self.last_refresh_time

        refresh_interval = 2.0  # seconds
        if self.tokens_generated == 0:
            self.last_refresh_time = current_time
            self.tokens_generated = decode_tokens * prefill_mask
            return

        tps = self.tokens_generated / elapsed
        if (tps > 0 and elapsed > refresh_interval) or (tps == 0 and elapsed > refresh_interval * 10) or is_prefill:
            msg = f"Decode batch. #running-req: {len(seq_group)}. "
            cache_usage = self.block_memory_manager.get_usage()
            msg += (
                f"gen throughput (token/s): {tps:.2f}, "
                f"#queue-req p+d: {self.waiting_queue.qsize()}+{self.running_queue.qsize()}, "
                f"cache usage: {cache_usage:.2f}%"
            )
            if running_batch.spec_info is not None:
                avg_accept_lengths = self.total_accept_tokens / self.total_forward_tokens
                msg += f", accept len: {avg_accept_lengths:.2f}"
            logger.info(msg)
            self.last_refresh_time = current_time
            self.tokens_generated = 0
            self.total_accept_tokens = self.total_forward_tokens = 0
            self.cum_accept_tokens = self.cum_forward_tokens = 0

        self.tokens_generated += decode_tokens * prefill_mask


class SchedulerPolicy:
    def __init__(self, memory_pool: BlockMemoryManager):
        self.memory_pool = memory_pool
        self.total_token_used = 0
        self.cur_token_used = 0
    
    @property
    def total_remain_tokens(self) -> int:
        return self.memory_pool.num_available_blocks() - self.total_token_used
    
    @property
    def cur_remain_tokens(self) -> int:
        return self.memory_pool.num_available_blocks() - self.cur_token_used

    def add_prefill_req(self, sequence: RequestSequence):
        token_len = len(sequence.tokens)
        self.total_token_used += sequence.max_possible_length
        self.cur_token_used += token_len

    def add_decode_req(self, sequence: RequestSequence):
        self.cur_token_used += 1
    
    def release_req(self, sequence: RequestSequence):
        token_len = len(sequence.out_cache_loc)
        self.cur_token_used -= token_len
        self.total_token_used -= sequence.max_possible_length

    def can_add_prefill_req(self, sequence: RequestSequence) -> bool:
        token_len = len(sequence.tokens)
        return self.total_remain_tokens >= token_len + sequence.sampling_params.max_new_tokens

class Scheduler:
    def __init__(self, server_args, model_worker: ModelWorker):
        self.waiting_queue = queue.Queue()
        self.running_queue = queue.Queue()
        
        self.max_context_length = server_args.max_context_length
        self.max_prefill_tokens = min(self.max_context_length, 8192)
        self.max_running_req = server_args.cuda_graph_max_bs
        self.abort_requests = set()

        self.server_args = server_args
        self.spec_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        self.block_memory_manager = model_worker.main_mem_pool
        self.draft_memory_pool = model_worker.draft_mem_pool if self.spec_algorithm.is_none() is False else None
        self.metrics = RunningMetrics(self.waiting_queue, self.running_queue, self.block_memory_manager, server_args=server_args)
        self.scheduler_policy = SchedulerPolicy(self.block_memory_manager)

        # overlap Scheduling
        self.enable_overlap = not server_args.disable_overlap
        self.flying_batch = ScheduleBatch([], self.block_memory_manager)

    
    def add_request(self, sequence: RequestSequence):
        self.waiting_queue.put(sequence)

    def swap_req_to_waiting(self, sequence: RequestSequence):
        self.free_sequence_resources(sequence)
        sequence.out_cache_loc = []
        sequence.status = "WAITING"
        sequence.new_tokens = sequence.tokens.copy()
        self.waiting_queue.put(sequence)

    def add_abort_request(self, sequence_id: int):
        self.abort_requests.add(sequence_id)

    def get_next_batch(self, last_batch: ScheduleBatch=None) -> Tuple[ScheduleBatch, List[RequestSequence]]:
        running_sequences = []
        failed_sequences = []
        # schedule prefill first
        prefill_tokens = 0

        overlap_running_size = self.max_running_req
        while not self.waiting_queue.empty():
            # check abort requests first
            if self.waiting_queue.queue[0].sequence_id in self.abort_requests:
                sequence = self.waiting_queue.get()
                sequence.status = SequenceStatus.CANCELLED
                self.free_sequence_resources(sequence)
                continue
            tokens = self.waiting_queue.queue[0].tokens
            if not self.scheduler_policy.can_add_prefill_req(self.waiting_queue.queue[0]):
                break
            if prefill_tokens + len(tokens) > self.max_prefill_tokens:
                break
            assert self.block_memory_manager.can_alloc(len(tokens))
            sequence = self.waiting_queue.get()
            sequence.status = SequenceStatus.RUNNING
            self.scheduler_policy.add_prefill_req(sequence)
            running_sequences.append(sequence)
            running_sequences[-1].out_cache_loc.extend(self.block_memory_manager.alloc_block(tokens, hashv=0))
            if not self.spec_algorithm.is_none():
                running_sequences[-1].out_cache_loc_spec.extend(self.draft_memory_pool.alloc_block(tokens, hashv=0))
            prefill_tokens += len(tokens)

        running_batch = ScheduleBatch(running_sequences, self.block_memory_manager, self.draft_memory_pool)
        if self.enable_overlap:
            # remove finished sequences from flying batch
            last_batch.filter()
            self.flying_batch.merge(last_batch)
            if len(running_sequences) == 0:
                # if there is no prefill request, schedule decode requests
                for seq in self.flying_batch.sequences:
                    self.scheduler_policy.add_decode_req(seq)
                    seq.out_cache_loc.extend(
                        self.block_memory_manager.alloc_block([-1], hashv=0)
                    )
                running_batch.merge(self.flying_batch)
                self.flying_batch = ScheduleBatch([], self.block_memory_manager)
        # if there is no prefill request, schedule decode requests
        elif len(running_sequences) == 0:
            while not self.running_queue.empty() and len(running_sequences) < overlap_running_size:
                if self.running_queue.queue[0].sequence_id in self.abort_requests:
                    sequence = self.running_queue.get()
                    self.free_sequence_resources(sequence)
                    continue
                # TODO: hmm add_decode_req may not be accurate here during speculative decoding enabled
                self.scheduler_policy.add_decode_req(self.running_queue.queue[0])
                running_sequences.append(self.running_queue.get())
                if not self.spec_algorithm.is_none():
                    target_verify_len = self.server_args.speculative_num_draft_tokens
                    assert self.block_memory_manager.can_alloc(target_verify_len)  # the future token ids are unknown
                    # we will only reserver a few of them, the rest will be free after verification
                    running_sequences[-1].out_cache_loc.extend(
                        self.block_memory_manager.alloc_block([-1]*target_verify_len, hashv=0)
                    )
                    # for draft model
                    total_draft_len = running_sequences[-1].accept_length_cpu[0]+1 # the first run to setup kv cache for last verify tokens
                    # the "+1" is for the last bonus token
                    assert self.draft_memory_pool.can_alloc(total_draft_len)
                    running_sequences[-1].out_cache_loc_spec.extend(
                        self.draft_memory_pool.alloc_block([-1]*total_draft_len, hashv=0)
                    )
                else:
                    assert self.block_memory_manager.can_alloc(1)  # the future token ids are unknown
                    running_sequences[-1].out_cache_loc.extend(self.block_memory_manager.alloc_block([-1], hashv=0))
            running_batch.spec_info = self.flying_batch.spec_info
            self.flying_batch = running_batch
        else:
            running_batch.spec_info = self.flying_batch.spec_info
            self.flying_batch = running_batch

        if len(running_sequences) == 0:
            if not self.running_queue.empty():
                logger.warning("swapping one running req to waiting for free memory.")
                self.swap_req_to_waiting(self.running_queue.get())
            elif not self.waiting_queue.empty() and not self.enable_overlap:
                sequence = self.waiting_queue.get()
                logger.warning(f"the request's token is too long. has tokens: {len(sequence.tokens)}, max context length: {self.max_context_length}. Marking as FAILED.")
                sequence.status = "FAILED"
                failed_sequences.append(sequence)
                self.free_sequence_resources(sequence)

        self.metrics.log_prefill_metrics(prefill_tokens)
        self.metrics.log_decode_metrics(running_batch, is_prefill=prefill_tokens > 0)

        return running_batch, failed_sequences

    def get_next_batch_async(self, last_batch: ScheduleBatch) -> Tuple[ScheduleBatch, List[RequestSequence]]:
        return self.get_next_batch(last_batch)

    def free_sequence_resources(self, sequence: RequestSequence):
        if sequence.sequence_id in self.abort_requests:
            self.abort_requests.remove(sequence.sequence_id)

        if not len(sequence.out_cache_loc):
            return
        self.block_memory_manager.free_block(sequence.out_cache_loc)
        self.scheduler_policy.release_req(sequence)

    def is_done(self) -> bool:
        return self.waiting_queue.empty() and self.running_queue.empty()