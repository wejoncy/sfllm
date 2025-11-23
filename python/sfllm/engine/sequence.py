
from sfllm.engine.sampling_params import SamplingParams
from enum import IntEnum, auto
import threading
from typing import List, Callable, Optional
from dataclasses import dataclass, field


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


@dataclass
class RawSequence:
    sequence_id: int = field(default_factory=_get_next_sequence_id)
    prompt: str = ""
    prompt_token_len: int = 0
    max_possible_length: int = 0
    sampling_params: Optional[SamplingParams] = None
    tokens: list = field(default_factory=list)
    new_tokens: list = field(default_factory=list)
    generated_tokens: list = field(default_factory=lambda: [-1])
    last_generated_token_pos: int = 0
    generated_text: str = ""
    status: SequenceStatus = SequenceStatus.PENDING


class RequestSequence(RawSequence):
    def __init__(
        self,
        prompt: str,
        sampling_params: Optional[SamplingParams] = None,
        input_ids: List[int] = None,
    ):
        if sampling_params is None:
            sampling_params = SamplingParams()
        super().__init__(prompt=prompt, sampling_params=sampling_params)
        self.out_cache_loc = []
        self.out_cache_loc_spec = []
        self.out_cache_loc_lazy = None # tensor on cuda
        self.marked = False # marked for draft token handling, fill with -1 for the future accepted tokens
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
            self.last_generated_token_pos = self.prompt_token_len
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
            or self.last_generated_token_pos - self.prompt_token_len
            >= self.sampling_params.max_new_tokens
        )

    def export_raw_sequence(self) -> RawSequence:
        return RawSequence(
            sequence_id=self.sequence_id,
            prompt=self.prompt,
            prompt_token_len=self.prompt_token_len,
            max_possible_length=self.max_possible_length,
            sampling_params=self.sampling_params,
            tokens=self.tokens.copy(),
            new_tokens=self.new_tokens.copy(),
            generated_tokens=self.generated_tokens.copy(),
            last_generated_token_pos=self.last_generated_token_pos,
            generated_text=self.generated_text,
            status=self.status,
        )