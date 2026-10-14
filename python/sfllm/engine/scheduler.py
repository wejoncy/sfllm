import queue
import logging
from collections import deque
from typing import List, Tuple


from sfllm.engine.sequence import RequestSequence, SequenceStatus
from sfllm.engine.schedule_batch import ScheduleBatch
from sfllm.engine.cache.pool import KVPool
from sfllm.spec_decoding.spec_common import SpeculativeAlgorithm
from sfllm.engine.model_worker import ModelWorker
from sfllm.engine.scheduler_metrics import RunningMetrics


logger = logging.getLogger(__name__)

class SchedulerPolicy:
    def __init__(self, memory_pool: KVPool):
        self.memory_pool = memory_pool
        self.total_token_used = 0
        self.cur_token_used = 0
        self.new_token_ratio = 0.7  # reserve 70% for new tokens
    
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

    def prefill_required_tokens(self, sequence: RequestSequence) -> float:
        token_len = len(sequence.tokens)
        return token_len + sequence.sampling_params.max_new_tokens + self.total_token_used*self.new_token_ratio

class Scheduler:
    def __init__(self, server_args, model_worker: ModelWorker):
        self.waiting_queue = queue.Queue()
        self.running_queue = queue.Queue()
        
        self.max_context_length = server_args.max_context_length
        self.max_prefill_tokens = min(self.max_context_length, 16 * 1024)
        self.max_running_req = server_args.max_running_requests
        # Stable slots are shared by overlap token handoff and model-owned state.
        self.free_request_indices = deque(range(self.max_running_req))
        self.abort_requests = set()

        self.server_args = server_args
        self.spec_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        self.mem_pool = model_worker.main_mem_pool
        self.draft_memory_pool = model_worker.draft_mem_pool if self.spec_algorithm.is_none() is False else None
        self.metrics = RunningMetrics(
            self.waiting_queue, self.running_queue, self.mem_pool,
            server_args=server_args, prefix_reuse=model_worker.cache.prefix,
        )
        self.scheduler_policy = SchedulerPolicy(self.mem_pool)
        self.cache = model_worker.cache

        # overlap Scheduling
        self.enable_overlap = not server_args.disable_overlap
        self.flying_batch = ScheduleBatch([], self.mem_pool)

    
    def add_request(self, sequence: RequestSequence):
        self.waiting_queue.put(sequence)

    def swap_req_to_waiting(self, sequence: RequestSequence):
        self._release_sequence(sequence)
        del sequence.out_cache_loc[:]
        del sequence.out_cache_loc_spec[:]
        sequence.status = SequenceStatus.PENDING
        sequence.new_tokens = sequence.tokens.copy()
        self.waiting_queue.put(sequence)

    def add_abort_request(self, sequence_id: int):
        self.abort_requests.add(sequence_id)

    def get_next_batch(self, last_batch: ScheduleBatch=None) -> Tuple[ScheduleBatch, List[RequestSequence]]:
        running_sequences = []
        failed_sequences = []
        if self.enable_overlap:
            self.flying_batch.merge(last_batch)
            self.flying_batch.filter()
        # schedule prefill first
        prefill_tokens = 0
        while not self.waiting_queue.empty() and self.free_request_indices:
            # check abort requests first
            if self.waiting_queue.queue[0].sequence_id in self.abort_requests:
                sequence = self.waiting_queue.get()
                sequence.status = SequenceStatus.CANCELLED
                self.free_sequence_resources(sequence)
                continue
            tokens = self.waiting_queue.queue[0].tokens
            if len(tokens) > self.max_context_length:
                sequence = self.waiting_queue.get()
                logger.warning(
                    f"Request has {len(tokens)} prompt tokens, exceeding "
                    f"the {self.max_context_length} token context limit."
                )
                sequence.status = SequenceStatus.FAILED
                sequence.generated_tokens.clear()
                failed_sequences.append(sequence)
                continue
            start, end, node = self.cache.select(self.waiting_queue.queue[0])
            # A prompt above the batch budget runs alone, up to the context limit.
            if (running_sequences
                    and prefill_tokens + end - start > self.max_prefill_tokens):
                break
            required_tokens = self.scheduler_policy.prefill_required_tokens(self.waiting_queue.queue[0])
            if not self.cache.can_admit(self.waiting_queue.queue[0], start, node, required_tokens):
                break
            sequence = self.waiting_queue.get()
            assert sequence.request_index == -1
            sequence.request_index = self.free_request_indices.popleft()
            self.cache.admit(sequence, start, end, node)
            sequence.status = SequenceStatus.RUNNING
            self.scheduler_policy.add_prefill_req(sequence)
            running_sequences.append(sequence)
            prefill_tokens += len(sequence.new_tokens)

        running_batch = ScheduleBatch(running_sequences, self.mem_pool, self.draft_memory_pool)
        self.cache.prepare_batch(running_batch)
        if self.enable_overlap:
            if len(running_sequences) == 0:
                # if there is no prefill request, schedule decode requests
                self._allocate_decode(self.flying_batch.sequences)
                running_batch.merge(self.flying_batch)
                running_batch.spec_info = self.flying_batch.spec_info
                self.flying_batch = ScheduleBatch([], self.mem_pool)
        # if there is no prefill request, schedule decode requests
        elif len(running_sequences) == 0: # no overlap
            while not self.running_queue.empty() and len(running_sequences) < self.max_running_req:
                if self.running_queue.queue[0].sequence_id in self.abort_requests:
                    sequence = self.running_queue.get()
                    self.free_sequence_resources(sequence)
                    continue
                running_sequences.append(self.running_queue.get())
            self._allocate_decode(running_sequences)
            running_batch.spec_info = self.flying_batch.spec_info
            self.flying_batch = running_batch
        else:
            running_batch.spec_info = self.flying_batch.spec_info
            self.flying_batch = running_batch

        if len(running_sequences) == 0:
            if not self.running_queue.empty():
                logger.warning("swapping one running req to waiting for free memory.")
                self.swap_req_to_waiting(self.running_queue.get())

        return running_batch, failed_sequences

    def get_next_batch_async(self, last_batch: ScheduleBatch) -> Tuple[ScheduleBatch, List[RequestSequence]]:
        return self.get_next_batch(last_batch)

    def _allocate_decode(self, sequences):
        width = 1 if self.spec_algorithm.is_none() else self.server_args.speculative_num_draft_tokens
        counts = [[width] * len(sequences)]
        if not self.spec_algorithm.is_none():
            counts.append([
                self.server_args.speculative_num_steps + 1
                if self.enable_overlap and len(seq.tokens) > seq.last_generated_token_pos
                else int(seq.accept_length_cpu[0]) + 1
                for seq in sequences
            ])
        self.cache.allocate_batch(sequences, counts)
        for sequence in sequences:
            self.scheduler_policy.add_decode_req(sequence)

    def free_sequence_resources(self, sequence: RequestSequence):
        self.abort_requests.discard(sequence.sequence_id)
        self._release_sequence(sequence)

    def _release_sequence(self, sequence):
        self.cache.release_request(sequence)
        if sequence.request_index >= 0:
            self.scheduler_policy.release_req(sequence)
            self.free_request_indices.append(sequence.request_index)
            sequence.request_index = -1

    def complete_batch(self, batch):
        self.cache.complete_batch(batch)

    def is_done(self) -> bool:
        return self.waiting_queue.empty() and self.running_queue.empty()
