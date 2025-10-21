
from sfllm.engine.sampling_params import SamplingParams


class SequenceStatus:
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

_cur_sequence_id = 0
def _get_next_sequence_id():
    global _cur_sequence_id
    _cur_sequence_id += 1
    return _cur_sequence_id

class Sequence:
    def __init__(self, prompt:str, sampling_params:SamplingParams = SamplingParams()):
        self.sequence_id = _get_next_sequence_id()
        self.prompt = prompt
        self.prompt_token_len = 0
        self.sampling_params = sampling_params
        self.tokens = []
        self.new_tokens = []
        self.generated_text = ""
        self.status = SequenceStatus.PENDING
        self.cache_loc_ids = []

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