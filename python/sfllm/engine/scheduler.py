import queue
from sfllm.engine.sequence import Sequence,SequenceGroup
from sfllm.engine.memory_pool import BlockMemoryManager

class Scheduler:
    def __init__(self):
        self.waiting_queue = queue.Queue()
        self.running_queue = queue.Queue()
        self.block_memory_manager = BlockMemoryManager(num_blocks=10240)

    
    def add_request(self, sequence: Sequence):
        self.waiting_queue.put(sequence)

    def schedule(self) -> SequenceGroup:
        running_sequences = []

        # schedule prefill first
        while not self.waiting_queue.empty():
            tokens = self.waiting_queue.queue[0].tokens
            if self.block_memory_manager.can_alloc(
                tokens
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
                if self.block_memory_manager.can_alloc([-1]):  # the future token ids are unknown
                    running_sequences.append(self.running_queue.get())
                    running_sequences[-1].cache_loc_ids.extend(
                        self.block_memory_manager.alloc_block([-1], hashv=0)
                    )
                else:
                    break        
        return SequenceGroup(running_sequences)
    
    def is_done(self) -> bool:
        return self.waiting_queue.empty() and self.running_queue.empty()