
from sfllm.engine.sampling_params import SamplingParams
from enum import IntEnum, auto
import threading
from typing import List, Callable


class SequenceStatus(IntEnum):
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    CANCELLED = auto()
    FAILED = auto()

    def __str__(self):
        return self.name
    
    def is_active(self):
        return self in {SequenceStatus.PENDING, SequenceStatus.RUNNING}

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


class AbortSequence:
    def __init__(self, sequence_id: int):
        self.sequence_id = sequence_id

class DecodeSequence:
    def __init__(self, request_sequence: 'RequestSequence'):
        self.sequence_id = request_sequence.sequence_id
        self.tokens = request_sequence.generated_tokens
        self.text = ""
        self.status = request_sequence.status
        self.completion_tokens = (
            len(request_sequence.tokens) - request_sequence.prompt_token_len
        )

class RequestSequence:
    def __init__(
        self,
        prompt: str,
        sampling_params: SamplingParams = SamplingParams(),
        input_ids: List[int] = None,
    ):
        self.sequence_id = _get_next_sequence_id()
        self.prompt = prompt
        self.prompt_token_len = 0
        self.max_possible_length = 0
        self.sampling_params = sampling_params
        self.tokens = []
        self.new_tokens = []
        self.generated_tokens = [-1]  # for overlap generation
        self.last_generated_token_pos = 0
        self.generated_text = ""
        self.status = SequenceStatus.PENDING
        self.out_cache_loc = []
        self.out_cache_loc_spec = []
        if input_ids is not None:
            self.tokens = input_ids
            self.prompt_token_len = len(input_ids)
            self.last_generated_token_pos = self.prompt_token_len
            self.new_tokens = input_ids.copy()
            self.max_possible_length = sampling_params.max_new_tokens + self.prompt_token_len
    
    def init(self, tokenizer_or_ids: Callable | list):
        if isinstance(tokenizer_or_ids, list):
            self.tokens = tokenizer_or_ids
            self.prompt_token_len = len(tokenizer_or_ids)
            self.new_tokens = tokenizer_or_ids.copy()
            self.max_possible_length = self.sampling_params.max_new_tokens + self.prompt_token_len
            return
        if self.tokens:
            return
        self.tokens = tokenizer_or_ids.encode(self.prompt)
        self.prompt_token_len = len(self.tokens)
        self.new_tokens = self.tokens.copy()
        self.max_possible_length = self.sampling_params.max_new_tokens + self.prompt_token_len
        self.last_generated_token_pos = self.prompt_token_len
    
    def is_done(self) -> bool:
        return (
            not self.status.is_active()
            or len(self.tokens) - self.prompt_token_len
            >= self.sampling_params.max_new_tokens
        )
