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
        self.last_refresh_time = time.perf_counter()
        self.waiting_queue = waiting_queue
        self.running_queue = running_queue
        self.block_memory_manager = block_memory_manager

    def refresh(self, seq_group: SequenceGroup):
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

class Scheduler:
    def __init__(self, max_context_length: int = 4096):
        self.waiting_queue = queue.Queue()
        self.running_queue = queue.Queue()
        self.block_memory_manager = BlockMemoryManager(num_blocks=10240)
        self.metrics = RunningMetrics(self.waiting_queue, self.running_queue, self.block_memory_manager)
        self.max_context_length = max_context_length
        self.max_prefill_tokens = min(max_context_length, 4096)

    
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
        reserved_blocks = self.running_queue.qsize()  # reserve some blocks for running sequences
        prefill_tokens = 0
        while not self.waiting_queue.empty():
            tokens = self.waiting_queue.queue[0].tokens
            if self.block_memory_manager.can_alloc(
                len(tokens) + reserved_blocks*1000
            ) and prefill_tokens + len(tokens) < self.max_prefill_tokens:  # the future token ids are unknown
                running_sequences.append(self.waiting_queue.get())
                running_sequences[-1].cache_loc_ids.extend(self.block_memory_manager.alloc_block(
                    tokens, hashv=0
                ))
                prefill_tokens += len(tokens)
            else:
                break
        # if there is no prefill request, schedule decode requests
        if len(running_sequences) == 0:
            while not self.running_queue.empty():
                if self.block_memory_manager.can_alloc(1):  # the future token ids are unknown
                    running_sequences.append(self.running_queue.get())
                    running_sequences[-1].cache_loc_ids.extend(
                        self.block_memory_manager.alloc_block([-1], hashv=0)
                    )
                else:
                    break
        if len(running_sequences) == 0:
            if not self.running_queue.empty():
                logger.warning("swapping one running req to waiting for free memory.")
                self.swap_req_to_waiting(self.running_queue.get())
            elif not self.waiting_queue.empty():
                sequence = self.waiting_queue.get()
                logger.warning(f"the request's token is too long. has tokens: {len(sequence.tokens)}, max context length: {self.max_context_length}. Marking as FAILED.")
                sequence.status = "FAILED"
                failed_sequences.append(sequence)
            # exit(-1)
        return SequenceGroup(running_sequences),failed_sequences

    def free_sequence_resources(self, sequence: Sequence):
        self.block_memory_manager.free_block(sequence.cache_loc_ids)

    def is_done(self) -> bool:
        return self.waiting_queue.empty() and self.running_queue.empty()