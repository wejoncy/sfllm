
from sfllm.engine.sampling_params import SamplingParams
from enum import IntEnum, auto
import threading
from typing import List, Tuple


class SequenceStatus(IntEnum):
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()

    def __str__(self):
        return self.name

class StrictCounter:
    def __init__(self, start=0):
        self._value = start
        self._lock = threading.Lock()

    def next(self):
        with self._lock:
            self._value += 1
            return self._value

strict_counter = StrictCounter()
def _get_next_sequence_id():
    return strict_counter.next()

class Sequence:
    def __init__(
        self,
        prompt: str,
        sampling_params: SamplingParams = SamplingParams(),
        input_ids: List[int] = None,
    ):
        self.sequence_id = _get_next_sequence_id()
        self.prompt = prompt
        self.prompt_token_len = 0
        self.sampling_params = sampling_params
        self.tokens = []
        self.new_tokens = []
        self.generated_text = ""
        self.status = SequenceStatus.PENDING
        self.cache_loc_ids = []
        if input_ids is not None:
            self.tokens = input_ids
            self.prompt_token_len = len(input_ids)
            self.new_tokens = input_ids.copy()

class SequenceGroup:
    def __init__(self, sequences: list[Sequence]):
        self.sequences = sequences
    
    def __iter__(self):
        return iter(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx]
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def empty(self) -> bool:
        return len(self.sequences) == 0

    def append(self, sequence_list: list[Sequence]) -> None:
        self.sequences.extend(sequence_list)