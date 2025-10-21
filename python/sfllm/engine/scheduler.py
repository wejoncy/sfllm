import time
import queue
import logging
from sfllm.engine.sequence import Sequence,SequenceGroup
from sfllm.engine.memory_pool import BlockMemoryManager

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
            f"#queue-req prefill: {(self.waiting_queue.qsize())}, "
            f"#queue-req decode: {(self.running_queue.qsize())}, "
            f"cache usage: {cache_usage:.2f}%"
        )
        logger.info(msg)

class Scheduler:
    def __init__(self):
        self.waiting_queue = queue.Queue()
        self.running_queue = queue.Queue()
        self.block_memory_manager = BlockMemoryManager(num_blocks=10240)
        self.metrics = RunningMetrics(self.waiting_queue, self.running_queue, self.block_memory_manager)

    
    def add_request(self, sequence: Sequence):
        self.waiting_queue.put(sequence)

    def drop_request_to_waiting(self, sequence: Sequence):
        self.running_queue.queue.remove(sequence)
        self.free_sequence_resources(sequence)
        sequence.cache_loc_ids = []
        sequence.status = "WAITING"
        sequence.tokens = sequence.tokens[: sequence.prompt_token_len]
        sequence.new_tokens = sequence.tokens.copy()
        self.waiting_queue.put(sequence)

    def schedule(self) -> SequenceGroup:
        running_sequences = []

        # schedule prefill first
        reserved_blocks = self.running_queue.qsize()  # reserve some blocks for running sequences
        while not self.waiting_queue.empty():
            tokens = self.waiting_queue.queue[0].tokens
            if self.block_memory_manager.can_alloc(
                len(tokens) + reserved_blocks*1000
            ):  # the future token ids are unknown
                running_sequences.append(self.waiting_queue.get())
                running_sequences[-1].cache_loc_ids.extend(self.block_memory_manager.alloc_block(
                    tokens, hashv=0
                ))
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
        if len(running_sequences) == 0 and (not self.waiting_queue.empty() or not self.running_queue.empty()):
            logger.error("deadlock detected: Memory full, cannot schedule any sequence.")
            exit(-1)
        return SequenceGroup(running_sequences)

    def free_sequence_resources(self, sequence: Sequence):
        self.block_memory_manager.free_block(sequence.cache_loc_ids)

    def is_done(self) -> bool:
        return self.waiting_queue.empty() and self.running_queue.empty()