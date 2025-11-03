import time
import queue
import logging
from sfllm.engine.sequence import RequestSequence, SequenceStatus
from sfllm.engine.shedule_batch import ScheduleBatch
from sfllm.engine.memory_pool import BlockMemoryManager
from typing import List, Tuple

logger = logging.getLogger(__name__)
class RunningMetrics:
    def __init__(
        self,
        waiting_queue: queue.Queue,
        running_queue: queue.Queue,
        block_memory_manager: BlockMemoryManager,
    ):
        self.tokens_generated = 0
        self.prefill_tokens = 0
        self.last_prefill_refresh_time = time.perf_counter()
        self.last_refresh_time = time.perf_counter()
        self.waiting_queue = waiting_queue
        self.running_queue = running_queue
        self.block_memory_manager = block_memory_manager

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

    def log_decode_metrics(self, seq_group: List[RequestSequence], is_prefill: bool=False):
        current_time = time.perf_counter()            
        elapsed = current_time - self.last_refresh_time

        refresh_interval = 2.0  # seconds

        if self.tokens_generated == 0:
            self.last_refresh_time = current_time
            self.tokens_generated = len(seq_group)*int(not is_prefill)
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
            logger.info(msg)
            self.last_refresh_time = current_time
            self.tokens_generated = 0

        self.tokens_generated += len(seq_group)*int(not is_prefill)


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
    def __init__(self, server_args, memory_pool: BlockMemoryManager):
        self.waiting_queue = queue.Queue()
        self.running_queue = queue.Queue()
        self.flying_queue = []
        self.block_memory_manager = memory_pool
        self.metrics = RunningMetrics(self.waiting_queue, self.running_queue, self.block_memory_manager)
        self.max_context_length = server_args.max_context_length
        self.max_prefill_tokens = min(self.max_context_length, 8192)
        self.max_decode_tokens = server_args.cuda_graph_max_bs
        self.scheduler_policy = SchedulerPolicy(self.block_memory_manager)
        self.abort_requests = set()
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

    def get_next_batch(self) -> Tuple[ScheduleBatch, List[RequestSequence]]:
        running_sequences = []
        failed_sequences = []
        # schedule prefill first
        running_size = self.running_queue.qsize()  # reserve some blocks for running sequences
        prefill_tokens = 0

        overlap_running_size = self.max_decode_tokens
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
            running_sequences[-1].out_cache_loc.extend(self.block_memory_manager.alloc_block(
                tokens, hashv=0
            ))
            prefill_tokens += len(tokens)
        # if there is no prefill request, schedule decode requests
        if len(running_sequences) == 0:
            while not self.running_queue.empty() and len(running_sequences) < overlap_running_size:
                if self.running_queue.queue[0].sequence_id in self.abort_requests:
                    sequence = self.running_queue.get()
                    self.free_sequence_resources(sequence)
                    continue
                assert self.block_memory_manager.can_alloc(1)  # the future token ids are unknown
                self.scheduler_policy.add_decode_req(self.running_queue.queue[0])
                running_sequences.append(self.running_queue.get())
                assert self.block_memory_manager.can_alloc(1)
                running_sequences[-1].out_cache_loc.extend(
                    self.block_memory_manager.alloc_block([-1], hashv=0)
                )

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
        self.metrics.log_decode_metrics(running_sequences, is_prefill=prefill_tokens>0)
        self.flying_batch = ScheduleBatch(running_sequences, self.block_memory_manager)
        return self.flying_batch, failed_sequences

    def get_next_batch_async(self, last_batch: ScheduleBatch) -> Tuple[ScheduleBatch, List[RequestSequence]]:
        running_sequences = []
        failed_sequences = []
        # schedule prefill first
        prefill_tokens = 0

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
            running_sequences[-1].out_cache_loc.extend(self.block_memory_manager.alloc_block(
                tokens, hashv=0
            ))
            prefill_tokens += len(tokens)
        
        # remove finished sequences from flying batch
        last_batch.filter()
        running_batch = ScheduleBatch(running_sequences, self.block_memory_manager)
        if len(running_sequences) > 0:
            self.flying_batch.merge(last_batch)
        else:
            self.flying_batch.merge(last_batch)
            # if there is no prefill request, schedule decode requests
            for seq in self.flying_batch.sequences:
                self.scheduler_policy.add_decode_req(seq)
                seq.out_cache_loc.extend(
                    self.block_memory_manager.alloc_block([-1], hashv=0)
                )
            running_batch.merge(self.flying_batch)
            self.flying_batch = ScheduleBatch([], self.block_memory_manager)

        self.metrics.log_prefill_metrics(prefill_tokens)
        self.metrics.log_decode_metrics(running_batch.sequences, is_prefill=prefill_tokens > 0)
        return running_batch, failed_sequences

    def free_sequence_resources(self, sequence: RequestSequence):
        if sequence.sequence_id in self.abort_requests:
            self.abort_requests.remove(sequence.sequence_id)

        if not len(sequence.out_cache_loc):
            return
        self.block_memory_manager.free_block(sequence.out_cache_loc)
        self.scheduler_policy.release_req(sequence)

    def is_done(self) -> bool:
        return self.waiting_queue.empty() and self.running_queue.empty()