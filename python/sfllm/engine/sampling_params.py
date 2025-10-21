
class SamplingParams:
    def __init__(self, max_tokens=50, temperature=0.8, top_p=0.95, top_k=1073741824):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.is_greedy = top_k <=1
