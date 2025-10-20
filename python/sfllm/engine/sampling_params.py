
class SamplingParams:
    def __init__(self, max_tokens=50, temperature=1.0, top_p=1.0):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p