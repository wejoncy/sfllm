import time
import queue
import logging
from sfllm.engine.sequence import Sequence,SequenceGroup
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
        self.last_prefill_refresh_time = current_time
        log_interval = 4.0  # seconds
        if self.prefill_tokens == 0:
            self.prefill_tokens = cur_prefill_tokens
            return

        msg = f"Prefill batch. #prefill_tokens: {self.prefill_tokens}. "
        msg += (
            f"gen throughput (token/s): {self.prefill_tokens / elapsed:.2f}, "
            f"#queue-req p+d: {self.waiting_queue.qsize()}+{self.running_queue.qsize()}, "
        )
        if current_time - self.last_refresh_time > log_interval:
            logger.info(msg)
        self.prefill_tokens = cur_prefill_tokens

    def log_decode_metrics(self, seq_group: List[Sequence]):
        current_time = time.perf_counter()
        elapsed = current_time - self.last_refresh_time
        refresh_interval = 4.0  # seconds
        self.tokens_generated += len(seq_group)

        if elapsed < refresh_interval:
            return
        msg = f"Decode batch. #running-req: {len(seq_group)}. "
        tps = self.tokens_generated / elapsed if elapsed > 0 else 0.0
        self.tokens_generated = 0
        self.last_refresh_time = current_time
        cache_usage = self.block_memory_manager.get_usage()
        msg += (
            f"gen throughput (token/s): {tps:.2f}, "
            f"#queue-req p+d: {self.waiting_queue.qsize()}+{self.running_queue.qsize()}, "
            f"cache usage: {cache_usage:.2f}%"
        )
        logger.info(msg)

class SchedulerPolicy:
    def __init__(self, memory_pool: BlockMemoryManager):
        self.memory_pool = memory_pool
        self.total_token_used = 0
        self.cur_token_used = 0
    
    @property
    def total_remain_tokens(self) -> int:
        return len(self.memory_pool.free_block_ids) - self.total_token_used
    
    @property
    def cur_remain_tokens(self) -> int:
        return len(self.memory_pool.free_block_ids) - self.cur_token_used

    def add_prefill_req(self, sequence: Sequence):
        token_len = len(sequence.tokens)
        self.total_token_used += token_len + sequence.sampling_params.max_new_tokens
        self.cur_token_used += token_len

    def add_decode_req(self, sequence: Sequence):
        self.cur_token_used += 1
    
    def release_req(self, sequence: Sequence):
        token_len = len(sequence.tokens)
        self.cur_token_used -= token_len
        self.total_token_used -= token_len

    def can_add_prefill_req(self, sequence: Sequence) -> bool:
        token_len = len(sequence.tokens)
        return self.total_remain_tokens >= token_len + sequence.sampling_params.max_new_tokens

class Scheduler:
    def __init__(self, server_args,
                 max_context_length: int = 4096, 
                 max_running_tokens: int = 10240,
                 ):
        self.waiting_queue = queue.Queue()
        self.running_queue = queue.Queue()
        self.block_memory_manager = BlockMemoryManager(num_blocks=max_running_tokens)
        self.metrics = RunningMetrics(self.waiting_queue, self.running_queue, self.block_memory_manager)
        self.max_context_length = max_context_length
        self.max_prefill_tokens = min(max_context_length, 4096)
        self.max_decode_tokens = server_args.cuda_graph_max_bs
        self.scheduler_policy = SchedulerPolicy(self.block_memory_manager)

    
    def add_request(self, sequence: Sequence):
        self.waiting_queue.put(sequence)

    def swap_req_to_waiting(self, sequence: Sequence):
        self.free_sequence_resources(sequence)
        sequence.cache_loc_ids = []
        sequence.status = "WAITING"
        sequence.new_tokens = sequence.tokens.copy()
        self.waiting_queue.put(sequence)

    def schedule(self) -> Tuple[SequenceGroup, List[Sequence]]:
        running_sequences = []
        failed_sequences = []
        # schedule prefill first
        running_size = self.running_queue.qsize()  # reserve some blocks for running sequences
        prefill_tokens = 0
        while not self.waiting_queue.empty() and running_size < self.max_decode_tokens:
            tokens = self.waiting_queue.queue[0].tokens
            if not self.scheduler_policy.can_add_prefill_req(self.waiting_queue.queue[0]): 
                break
            if prefill_tokens + len(tokens) > self.max_prefill_tokens:
                break
            self.scheduler_policy.add_prefill_req(self.waiting_queue.queue[0])
            running_sequences.append(self.waiting_queue.get())
            running_sequences[-1].cache_loc_ids.extend(self.block_memory_manager.alloc_block(
                tokens, hashv=0
            ))
            prefill_tokens += len(tokens)
        # if there is no prefill request, schedule decode requests
        if len(running_sequences) == 0:
            while not self.running_queue.empty() and len(running_sequences) < self.max_decode_tokens:
                assert self.block_memory_manager.can_alloc(1)  # the future token ids are unknown
                self.scheduler_policy.add_decode_req(self.running_queue.queue[0])
                running_sequences.append(self.running_queue.get())
                running_sequences[-1].cache_loc_ids.extend(
                    self.block_memory_manager.alloc_block([-1], hashv=0)
                )

        if len(running_sequences) == 0:
            if not self.running_queue.empty():
                logger.warning("swapping one running req to waiting for free memory.")
                self.swap_req_to_waiting(self.running_queue.get())
            elif not self.waiting_queue.empty():
                sequence = self.waiting_queue.get()
                logger.warning(f"the request's token is too long. has tokens: {len(sequence.tokens)}, max context length: {self.max_context_length}. Marking as FAILED.")
                sequence.status = "FAILED"
                failed_sequences.append(sequence)
        self.metrics.log_decode_metrics(running_sequences)
        self.metrics.log_prefill_metrics(prefill_tokens)
        return SequenceGroup(running_sequences), failed_sequences

    def free_sequence_resources(self, sequence: Sequence):
        self.block_memory_manager.free_block(sequence.cache_loc_ids)
        self.scheduler_policy.release_req(sequence)

    def is_done(self) -> bool:
        return self.waiting_queue.empty() and self.running_queue.empty()