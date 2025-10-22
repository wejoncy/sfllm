
from typing import Dict, Any

class SamplingParams:
    def __init__(self, max_new_tokens=50, temperature=0.8, top_p=0.95, top_k=1073741824):
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.is_greedy = top_k <=1

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> "SamplingParams":
        return cls(
            max_new_tokens=params.get("max_new_tokens", 50),
            temperature=params.get("temperature", 0.8),
            top_p=params.get("top_p", 0.95),
            top_k=params.get("top_k", 1073741824),
        )
