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
        mem_pool: BlockMemoryManager,
        server_args=None,
    ):
        self.num_generated_tokens = 0
        self.spec_num_forward_ct = 0
        self.cum_forward_ct = 0
        self.spec_num_accepted_tokens = 0
        self.cum_spec_accept_tokens = 0
        self.prefill_tokens = 0
        self.last_prefill_refresh_time = time.perf_counter()
        self.last_refresh_time = time.perf_counter()
        self.waiting_queue = waiting_queue
        self.running_queue = running_queue
        self.mem_pool = mem_pool
        self.server_args = server_args

    def update_spec_metrics(self, spec_info):
        if spec_info is None:
            return
        bs, num_accepted_tokens = len(spec_info.accept_length_cpu), spec_info.accept_length_cpu.sum().item()
        if num_accepted_tokens < 0:
            return
        self.spec_num_accepted_tokens += num_accepted_tokens + bs
        self.spec_num_forward_ct += bs
        self.num_generated_tokens += num_accepted_tokens
        self.cum_spec_accept_tokens += bs+num_accepted_tokens

    def log_prefill_metrics(self, schedule_batch: ScheduleBatch):
        if not schedule_batch.forward_batch.is_decode():
            cur_prefill_tokens = len(schedule_batch.input_ids)
        else:
            cur_prefill_tokens = 0
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

    def log_decode_metrics(self, schedule_batch: ScheduleBatch):
        is_prefill = not schedule_batch.forward_batch.is_decode()
        batch_size = len(schedule_batch)
        decode_tokens = batch_size
        prefill_mask = int(not is_prefill)
        current_time = time.perf_counter()
        elapsed = current_time - self.last_refresh_time

        self.cum_forward_ct += decode_tokens * prefill_mask

        refresh_interval = 2.0  # seconds
        if self.num_generated_tokens == 0:
            self.last_refresh_time = current_time
            self.num_generated_tokens = decode_tokens * prefill_mask
            return

        tps = self.num_generated_tokens / elapsed
        if (tps > 0 and elapsed > refresh_interval) or (tps == 0 and elapsed > refresh_interval * 10) or is_prefill:
            msg = f"Decode batch. #running-req: {batch_size}. "
            cache_usage = self.mem_pool.get_usage()
            msg += (
                f"gen throughput (token/s): {tps:.2f}, "
                f"cache usage: {cache_usage:.2f}%"
            )
            if schedule_batch.spec_info is not None:
                avg_accept_lengths = self.spec_num_accepted_tokens / self.spec_num_forward_ct
                msg += f", accept len: {avg_accept_lengths:.2f}"
            logger.info(msg)
            self.last_refresh_time = current_time
            self.num_generated_tokens = 0
            self.spec_num_accepted_tokens = self.spec_num_forward_ct = 0

        self.num_generated_tokens += decode_tokens * prefill_mask