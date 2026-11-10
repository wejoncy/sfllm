import torch
from torch import nn
import platform
import dataclasses

from sfllm.kernels.topk import pruned_topk


@dataclasses.dataclass
class SamplingBatchInfo:
    # Basic batched sampling params
    temperatures: torch.Tensor
    top_ps: torch.Tensor
    top_ks: torch.Tensor
    min_ps: torch.Tensor
    is_all_greedy: bool
    # Computed from request parameters on the CPU, without reading GPU tensors.
    max_top_k: int | None = None


def maybe_compile(fn):
    def wrapper(*args, **kwargs):
        """Only compile if not on Windows."""
        if platform.system() == "Windows":
            return fn(*args, **kwargs)
        else:
            return torch.compile(fn)(*args, **kwargs)

    return wrapper

class Sampler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config.vocab_size
    
    def top_k_top_p_min_p_sampling_from_probs_torch(
        self,
        probs: torch.Tensor,
        top_ks: torch.Tensor,
        top_ps: torch.Tensor,
        min_ps: torch.Tensor,
        need_min_p_sampling: bool,
        max_top_k: int | None = None,
    ):
        """Sample after filtering full-softmax probabilities by top-k, top-p and min-p."""
        if max_top_k is not None and 0 < max_top_k <= 64 and max_top_k < probs.shape[-1]:
            # Only these entries survive the top-k mask. Keep full-softmax
            # probabilities so the existing top-p cutoff has the same meaning.
            probs_sort, probs_idx = pruned_topk(probs, max_top_k)
        else:
            probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        probs_sort[
            torch.arange(0, probs_sort.shape[-1], device=probs.device).view(1, -1)
            >= top_ks.view(-1, 1)
        ] = 0.0
        probs_sort[(probs_sum - probs_sort) > top_ps.view(-1, 1)] = 0.0

        if need_min_p_sampling:
            min_p_thresholds = probs_sort[:, 0] * min_ps
            probs_sort[probs_sort < min_p_thresholds.view(-1, 1)] = 0.0

        sampled_index = torch.multinomial(probs_sort, num_samples=1)
        # int32 range is enough to represent the token ids
        # probs_idx = probs_idx.to(torch.int32)
        batch_next_token_ids = torch.gather(probs_idx, dim=1, index=sampled_index).view(-1)
        return batch_next_token_ids


    # @maybe_compile
    def forward(self, logits: torch.Tensor, sampling_batch_info: SamplingBatchInfo):
        if sampling_batch_info is None:
            return logits
        elif sampling_batch_info.is_all_greedy:
            return torch.argmax(logits, dim=-1)
        else:
            logits = logits.float().div_(sampling_batch_info.temperatures.unsqueeze(dim=1))
            probs = torch.softmax(logits, dim=-1)
            return self.top_k_top_p_min_p_sampling_from_probs_torch(
                probs,
                sampling_batch_info.top_ks,
                sampling_batch_info.top_ps,
                sampling_batch_info.min_ps,
                False,
                sampling_batch_info.max_top_k,
            )
        # sample_tokens = probs.div_(
        #     torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        # ).argmax(dim=-1)
        # return sample_tokens
