import torch
from torch import nn
import platform

def maybe_compile(fn):
    """Only compile if not on Windows."""
    if platform.system() == "Windows":
        print("[torch.compile disabled on Windows]")
        return fn
    else:
        print("[torch.compile enabled]")
        return torch.compile(fn) 

class Sampler(nn.Module):
    def __init__(self):
        super().__init__()

    @maybe_compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(
            torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        ).argmax(dim=-1)
        return sample_tokens
