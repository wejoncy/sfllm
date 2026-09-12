"""Qwen3 DSpark draft: shared target-KV backbone with a Markov proposal head."""

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn

from sfllm.kernels.dflash2 import dflash2_selector_greedy_walk
from sfllm.kernels.sampling import logits_argmax
from sfllm.models.dflash2 import DFlash2Attention, DFlash2DraftModel


@dataclass(frozen=True)
class DSparkConfig:
    # Number of proposals, excluding the anchor in the target verify block.
    block_size: int
    mask_token_id: int
    target_layer_ids: Tuple[int, ...]
    markov_rank: int
    markov_head_type: str
    conv_kernel_size: int = 0
    conv_group_size: int = 0

    @classmethod
    def from_hf_config(cls, config):
        raw = config.to_dict() if hasattr(config, "to_dict") else dict(config)
        fields = {**raw, **raw.get("dflash_config", {}), **raw.get("dspark_config", {})}
        for key in ("block_size", "markov_rank", "markov_head_type", "target_layer_ids"):
            if raw.get(f"dspark_{key}") is not None:
                fields[key] = raw[f"dspark_{key}"]
        if raw.get("dspark_noise_token_id") is not None:
            fields["mask_token_id"] = raw["dspark_noise_token_id"]
        try:
            parsed = cls(
                block_size=int(fields["block_size"]),
                mask_token_id=int(fields["mask_token_id"]),
                target_layer_ids=tuple(int(x) for x in fields["target_layer_ids"]),
                markov_rank=int(fields["markov_rank"]),
                markov_head_type=str(fields["markov_head_type"]).lower(),
                conv_kernel_size=int(fields.get("conv_kernel_size", 0)),
                conv_group_size=int(fields.get("conv_group_size", 0)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid DSpark checkpoint config.") from exc
        if raw.get("model_type") != "qwen3" or raw.get("attention_bias", False):
            raise ValueError("DSpark requires a Qwen3 draft with attention_bias=false.")
        if parsed.block_size < 1 or parsed.markov_rank < 1:
            raise ValueError("DSpark block_size and markov_rank must be positive.")
        if parsed.markov_head_type not in ("vanilla", "gated", "rnn"):
            raise ValueError(f"Unsupported DSpark Markov head: {parsed.markov_head_type}.")
        if not 0 <= parsed.mask_token_id < int(raw["vocab_size"]):
            raise ValueError("DSpark mask_token_id is outside the vocabulary.")
        ids = parsed.target_layer_ids
        if not ids or min(ids) < 0 or tuple(sorted(set(ids))) != ids:
            raise ValueError("DSpark target_layer_ids must be non-negative, unique and sorted.")
        taps, group = parsed.conv_kernel_size, parsed.conv_group_size
        if taps < 0 or group < 0 or bool(taps) != bool(group):
            raise ValueError("DSpark convolution sizes must both be positive or both zero.")
        if group and int(raw["hidden_size"]) % group:
            raise ValueError("DSpark conv_group_size must divide hidden_size.")
        layers = raw.get("layer_types")
        if (
            not layers or len(layers) != int(raw["num_hidden_layers"])
            or any(t not in ("full_attention", "sliding_attention") for t in layers)
        ):
            raise ValueError("DSpark requires one full/sliding attention type per draft layer.")
        if "sliding_attention" in layers and int(raw.get("sliding_window") or 0) <= 0:
            raise ValueError("DSpark sliding attention requires a positive window.")
        return parsed


class DSparkAttention(DFlash2Attention):
    @staticmethod
    def attention_is_causal(config, sliding):
        return getattr(config, "is_causal", sliding)


class DSparkMarkovHead(nn.Module):
    def __init__(self, hidden_size, vocab_size, rank, head_type):
        super().__init__()
        self.head_type = head_type
        self.markov_w1 = nn.Embedding(vocab_size, rank)
        self.markov_w2 = nn.Linear(rank, vocab_size, bias=False)
        if head_type == "gated":
            self.gate_proj = nn.Linear(hidden_size + rank, rank)
        elif head_type == "rnn":
            self.joint_proj = nn.Linear(hidden_size + 2 * rank, 3 * rank)
        elif head_type != "vanilla":
            raise ValueError(f"Unsupported DSpark Markov head: {head_type}.")

    def sample_candidates(self, candidate_ids, unary_logits, anchor_tokens, out):
        """Build all top-k Markov edges together, then walk the small lattice."""
        if self.head_type != "vanilla":
            raise ValueError("DSpark top-k proposals require a vanilla Markov head.")
        top_k = candidate_ids.shape[-1]
        predecessors = torch.cat([
            anchor_tokens[:, None, None].expand(-1, 1, top_k),
            candidate_ids[:, :-1],
        ], dim=1)
        predecessor_latent = self.markov_w1(predecessors)
        successor_latent = self.markov_w2.weight[candidate_ids]
        transitions = torch.einsum(
            "blpr,blcr->blpc", predecessor_latent, successor_latent
        )
        dflash2_selector_greedy_walk(candidate_ids, unary_logits, transitions, out)

    def sample(self, base_logits, hidden_states, anchor_tokens, out):
        previous = anchor_tokens
        state = None
        for step in range(base_logits.shape[1]):
            latent = self.markov_w1(previous)
            if self.head_type == "gated":
                gate = torch.sigmoid(self.gate_proj(torch.cat(
                    [hidden_states[:, step], latent], dim=-1
                )))
                latent = gate * latent
            elif self.head_type == "rnn":
                if state is None:
                    state = torch.zeros_like(latent)
                gate, candidate, output = self.joint_proj(torch.cat(
                    [state, latent, hidden_states[:, step]], dim=-1
                )).chunk(3, dim=-1)
                gate = torch.sigmoid(gate)
                state = gate * state + (1 - gate) * torch.tanh(candidate)
                latent = torch.tanh(output)
            # Preserve checkpoint dtype rounding in both projection and addition.
            previous = logits_argmax(
                base_logits[:, step], self.markov_w2(latent), out=out[:, step]
            )


class DSparkDraftModel(DFlash2DraftModel):
    config_cls = DSparkConfig
    attention_cls = DSparkAttention

    def _init_proposal_head(self):
        self.proposal_top_k = -1
        self.markov_head = DSparkMarkovHead(
            int(self.config.hidden_size), int(self.config.vocab_size),
            self.dflash_config.markov_rank, self.dflash_config.markov_head_type,
        )

    def sample_proposals(self, hidden_states, target_head_weight, anchor_tokens, out):
        if self.proposal_top_k > 0:
            ids, unary = self.compute_candidates(
                hidden_states, target_head_weight,
                top_k=self.proposal_top_k,
            )
            self.markov_head.sample_candidates(
                ids, unary.to(target_head_weight.dtype), anchor_tokens, out
            )
            return
        # Dropping the anchor leaves gaps between requests. Flatten explicitly
        # so matmul uses one GEMM instead of rereading the head for each request.
        batch, slots, hidden_size = hidden_states.shape
        flat_hidden = hidden_states.reshape(-1, hidden_size).to(target_head_weight.dtype)
        logits = torch.matmul(flat_hidden, target_head_weight.T).view(batch, slots, -1)
        self.markov_head.sample(logits, hidden_states, anchor_tokens, out)

    def load_weights(self, weights):
        def backbone_and_markov_weights():
            for name, weight in weights:
                name = name.removeprefix("model.")
                # Fixed-width greedy decoding does not use the confidence planner.
                if name in ("confidence_head.proj.weight", "confidence_head.proj.bias"):
                    continue
                name = {
                    "encoder.fc.weight": "fc.weight",
                    "encoder.output_norm_enc.weight": "hidden_norm.weight",
                }.get(name, name)
                yield name, weight
        super().load_weights(backbone_and_markov_weights())


class Qwen3DSparkModel(DSparkDraftModel):
    pass


EntryClass = [DSparkDraftModel, Qwen3DSparkModel]
