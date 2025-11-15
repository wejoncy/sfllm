import logging
from typing import Optional
from sfllm.engine.forward_params import ForwardBatch,ForwardMode

import torch
from torch import nn

logger = logging.getLogger(__name__)

class LogitsProcessor(nn.Module):
    def __init__(
        self, config, logit_scale: Optional[float] = None
    ):
        super().__init__()
        self.config = config
        self.logit_scale = logit_scale
        self.use_fp32_lm_head = False

    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: nn.Embedding,
        embedding_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get logits from hidden_states.

        If sampled_logits_only is True, it means hidden_states only contain the
        last position (e.g., extend without input logprobs). The caller should
        guarantee the given hidden_states follow this constraint.
        """
        if hasattr(lm_head, "weight"):
            if self.use_fp32_lm_head:
                logits = torch.matmul(
                    hidden_states.to(torch.float32), lm_head.weight.to(torch.float32).T
                )
            else:
                logits = torch.matmul(
                    hidden_states.to(lm_head.weight.dtype), lm_head.weight.T
                )
        else:
            # GGUF models
            # TODO: use weight_packed_linear for GGUF models
            if self.use_fp32_lm_head:
                with torch.cuda.amp.autocast(enabled=False):
                    logits = lm_head(hidden_states.to(torch.float32), embedding_bias
                    )
            else:
                logits = lm_head(hidden_states, embedding_bias)

        if self.logit_scale is not None:
            logits.mul_(self.logit_scale)
        return logits

    def forward(
        self,
        hidden_states,
        lm_head: nn.Embedding,
        aux_hidden_states: Optional[torch.Tensor] = None,
        forward_batch: Optional[ForwardBatch] = None,
    ) -> torch.Tensor:
        """Compute logits from hidden states.

        Args:
            input_ids: Input token IDs.
            hidden_states: Hidden states from the model.
            lm_head: Language model head (embedding layer).
            aux_hidden_states: Auxiliary hidden states for additional processing.

        Returns:
            Logits tensor.
        """
        if forward_batch.forward_mode == ForwardMode.EXTEND:
            prune_hidden_states = hidden_states[forward_batch.qo_indptr[1:] - 1]
        else:
            prune_hidden_states = hidden_states
        logits = self._get_logits(prune_hidden_states, lm_head)
        return logits, aux_hidden_states
