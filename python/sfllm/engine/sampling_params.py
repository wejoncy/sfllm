
from typing import Dict, Any

class SamplingParams:
    def __init__(
        self,
        max_new_tokens=50,
        temperature=0.8,
        top_p=0.95,
        top_k=1073741824,
        stop_token_ids=None,
        stop=None,
    ):
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        if isinstance(stop_token_ids, int):
            stop_token_ids = (stop_token_ids,)
        self.stop_token_ids = frozenset(stop_token_ids or ())
        if isinstance(stop, str):
            stop = (stop,)
        self.stop = tuple(stop or ())
        self.stop_token_sequences = ()
        self.is_greedy = temperature == 0 or top_k <= 1

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> "SamplingParams":
        return cls(
            max_new_tokens=params.get("max_new_tokens", 50),
            temperature=params.get("temperature", 0.8),
            top_p=params.get("top_p", 0.95),
            top_k=params.get("top_k", 1073741824),
            stop_token_ids=params.get("stop_token_ids"),
            stop=params.get("stop"),
        )
