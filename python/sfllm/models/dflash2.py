"""Qwen3 DFlash2 draft model.

The draft owns its transformer, block-local convolutions, context projection,
and candidate selector. Token embeddings and the vocabulary head are supplied
by the target model at execution time.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

try:
    from flashinfer import top_k as _flashinfer_top_k
except ImportError:
    _flashinfer_top_k = None

from sfllm.engine.forward_params import ForwardBatch
from sfllm.layers.layernorm import RMSNorm
from sfllm.model_loader.weight_utils import default_weight_loader
from sfllm.models.qwen3 import Qwen3Attention, Qwen3MLP


@dataclass(frozen=True)
class DFlash2Config:
    """The runtime fields emitted by the DFlash-to-DFlash2 initializer."""

    block_size: int
    mask_token_id: int
    conv_kernel_size: int
    conv_group_size: int
    selector_rank: int
    selector_top_k: int
    target_layer_ids: Tuple[int, ...]

    @classmethod
    def from_hf_config(cls, config: Any) -> "DFlash2Config":
        raw = config.to_dict() if hasattr(config, "to_dict") else dict(config)
        if raw.get("model_type") != "qwen3":
            raise ValueError("DFlash2 currently supports Qwen3 draft checkpoints.")
        if bool(raw.get("attention_bias", False)):
            raise ValueError(
                "DFlash2 target-KV materialization currently requires "
                "attention_bias=false."
            )

        section = raw.get("dflash_config")
        if not isinstance(section, Mapping):
            raise ValueError("DFlash2 checkpoint is missing dflash_config.")
        try:
            parsed = cls(
                block_size=int(section["block_size"]),
                mask_token_id=int(section["mask_token_id"]),
                conv_kernel_size=int(section["conv_kernel_size"]),
                conv_group_size=int(section["conv_group_size"]),
                selector_rank=int(section["selector_rank"]),
                selector_top_k=int(section["selector_top_k"]),
                target_layer_ids=tuple(int(x) for x in section["target_layer_ids"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid canonical DFlash2 checkpoint config.") from exc

        if parsed.block_size < 2:
            raise ValueError("DFlash2 block_size must be at least two.")
        if parsed.conv_kernel_size <= 0 or parsed.conv_group_size <= 0:
            raise ValueError("DFlash2 convolution sizes must be positive.")
        if parsed.selector_rank <= 0 or parsed.selector_top_k <= 0:
            raise ValueError("DFlash2 selector rank and top-k must be positive.")
        if int(raw["hidden_size"]) % parsed.conv_group_size:
            raise ValueError("DFlash2 conv_group_size must divide hidden_size.")
        if section.get("sample_from_anchor", False) is not False:
            raise ValueError("DFlash2 currently requires sample_from_anchor=false.")
        vocab_size = int(raw["vocab_size"])
        if not 0 <= parsed.mask_token_id < vocab_size:
            raise ValueError("DFlash2 mask_token_id is outside the vocabulary.")
        if parsed.selector_top_k > vocab_size:
            raise ValueError("DFlash2 selector_top_k exceeds the vocabulary size.")
        if (
            not parsed.target_layer_ids
            or any(x < 0 for x in parsed.target_layer_ids)
            or tuple(sorted(set(parsed.target_layer_ids))) != parsed.target_layer_ids
        ):
            raise ValueError(
                "DFlash2 target_layer_ids must be non-negative, unique, and sorted."
            )
        num_hidden_layers = int(raw["num_hidden_layers"])
        layer_types = raw.get("layer_types")
        if (
            num_hidden_layers <= 0
            or not isinstance(layer_types, list)
            or len(layer_types) != num_hidden_layers
        ):
            raise ValueError("DFlash2 requires one layer_types entry per draft layer.")
        if len(set(layer_types)) != 1 or layer_types[0] not in (
            "full_attention", "sliding_attention"
        ):
            raise ValueError(
                "DFlash2 requires uniformly full or sliding attention layers."
            )
        if layer_types[0] == "sliding_attention" and int(raw.get("sliding_window") or 0) <= 0:
            raise ValueError("DFlash2 sliding attention requires a positive window.")
        return parsed


class DFlash2Attention(Qwen3Attention):
    """Non-causal Qwen3 block attention with target-KV materialization."""

    def __init__(self, config, layer_id: int, quant_config=None, prefix: str = ""):
        rope = getattr(config, "rope_parameters", None) or {}
        sliding = config.layer_types[layer_id] == "sliding_attention"
        is_causal = self.attention_is_causal(config, sliding)
        window = int(config.sliding_window) - 1 if sliding else -1
        super().__init__(
            hidden_size=int(config.hidden_size),
            num_heads=int(config.num_attention_heads),
            num_kv_heads=int(config.num_key_value_heads),
            layer_id=layer_id,
            rope_theta=float(rope.get("rope_theta", getattr(config, "rope_theta", 1_000_000))),
            rope_scaling=rope or getattr(config, "rope_scaling", None),
            head_dim=int(getattr(config, "head_dim", 0) or 0) or None,
            max_position_embeddings=int(config.max_position_embeddings),
            quant_config=quant_config,
            rms_norm_eps=float(config.rms_norm_eps),
            attention_bias=bool(config.attention_bias),
            prefix=prefix,
            alt_stream=None,
            is_causal=is_causal,
            window_size=(window, 0 if sliding and is_causal else window),
        )

    @staticmethod
    def attention_is_causal(config, sliding):
        return sliding and getattr(
            config, "is_causal",
            not getattr(config, "dflash_config", {}).get("sliding_window_non_causal", False),
        )

    def materialize_kv(
        self,
        raw_kv: torch.Tensor,
        positions: torch.Tensor,
        cache_locs: torch.Tensor,
        kv_buffer: Tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        k, v = raw_kv.split([self.kv_size, self.kv_size], dim=-1)
        k = k.view(k.shape[0], self.num_kv_heads, self.head_dim)
        v = v.view_as(k)
        # Target KV materialization has no query heads.
        torch.ops.sfkernels.qk_norm_rope_and_cache(
            k[:, :0], k, v,
            self.q_norm.weight, self.k_norm.weight,
            self.rotary_emb.cos_sin_cache,
            positions,
            not self.rotary_emb.is_neox_style,
            kv_buffer[0], kv_buffer[1], cache_locs,
            self.k_norm.variance_epsilon,
        )


@torch.compile(dynamic=True)
def _grouped_conv(
    hidden_states: torch.Tensor,
    delta: torch.Tensor,
    base: torch.Tensor,
    block_size: int,
    num_groups: int,
    group_size: int,
    taps: int,
) -> torch.Tensor:
    blocks = hidden_states.unflatten(-1, (num_groups, group_size))
    coefficients = base.view(1, taps, num_groups, group_size) + delta.unsqueeze(-1)
    output = coefficients[:, 0] * blocks
    positions = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    if block_size & (block_size - 1) == 0:
        positions = positions & (block_size - 1)
    else:
        positions = positions % block_size
    for tap in range(1, taps):
        shifted = F.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))
        output = output + coefficients[:, tap] * shifted * (
            positions >= tap
        ).view(-1, 1, 1)
    return output.flatten(-2)


class DFlash2GroupedConv(nn.Module):
    """Block-local dynamic depthwise convolution used by DFlash2."""

    def __init__(
        self, hidden_size: int, block_size: int, taps: int, group_size: int
    ) -> None:
        super().__init__()
        self.block_size = int(block_size)
        self.taps = int(taps)
        self.group_size = int(group_size)
        self.num_groups = int(hidden_size) // self.group_size

        base_kernel = torch.zeros(2, self.taps, int(hidden_size))
        base_kernel[:, 0] = 1
        self.base_kernel = nn.Parameter(base_kernel)
        self.kernel_projection = nn.Linear(
            int(hidden_size), 2 * self.taps * self.num_groups, bias=False
        )

    def _convolve(
        self, hidden_states: torch.Tensor, coefficients: torch.Tensor, side: int
    ) -> torch.Tensor:
        return _grouped_conv(
            hidden_states,
            coefficients,
            self.base_kernel[side],
            self.block_size,
            self.num_groups,
            self.group_size,
            self.taps,
        )

    def prepare(self, hidden_states: torch.Tensor):
        coefficients = self.kernel_projection(hidden_states).reshape(
            *hidden_states.shape[:-1], 2, self.taps, self.num_groups
        )
        return (
            self._convolve(hidden_states, coefficients[..., 0, :, :], side=0),
            coefficients[..., 1, :, :],
        )

    def finish(
        self, hidden_states: torch.Tensor, coefficients: torch.Tensor
    ) -> torch.Tensor:
        return self._convolve(hidden_states, coefficients, side=1)


class DFlash2DecoderLayer(nn.Module):
    def __init__(self, config, layer_id: int, quant_config=None) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        self.input_layernorm = RMSNorm(hidden_size, eps=float(config.rms_norm_eps))
        self.self_attn = DFlash2Attention(
            config,
            layer_id,
            quant_config=quant_config,
            prefix=f"layers.{layer_id}.self_attn",
        )
        self.post_attention_layernorm = RMSNorm(
            hidden_size, eps=float(config.rms_norm_eps)
        )
        self.mlp = Qwen3MLP(
            hidden_size=hidden_size,
            intermediate_size=int(config.intermediate_size),
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"layers.{layer_id}.mlp",
        )
        conv_args = (
            hidden_size,
            int(config.dflash_config["block_size"]),
            int(config.dflash_config["conv_kernel_size"]),
            int(config.dflash_config["conv_group_size"]),
        )
        self.attention_conv = DFlash2GroupedConv(*conv_args)
        self.mlp_conv = DFlash2GroupedConv(*conv_args)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states, attention_kernel = self.attention_conv.prepare(hidden_states)
        hidden_states = self.self_attn(positions, hidden_states, forward_batch)
        hidden_states = self.attention_conv.finish(hidden_states, attention_kernel)

        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual
        )
        hidden_states, mlp_kernel = self.mlp_conv.prepare(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.mlp_conv.finish(hidden_states, mlp_kernel)
        return hidden_states, residual


class DFlash2CandidateSelector(nn.Module):
    def __init__(
        self, hidden_size: int, vocab_size: int, state_rank: int, top_k: int
    ) -> None:
        super().__init__()
        self.top_k = int(top_k)
        self.predecessor_codebook = nn.Parameter(
            torch.zeros(vocab_size, state_rank), requires_grad=False
        )
        self.successor_codebook = nn.Parameter(
            torch.zeros(vocab_size, state_rank), requires_grad=False
        )
        self.hidden_projection = nn.Linear(hidden_size, state_rank, bias=False)

    def build_lattice(
        self,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        anchor_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.hidden_projection(hidden_states)
        successors = self.successor_codebook[candidate_ids]
        predecessor_ids = torch.cat(
            (
                anchor_token_ids[:, None, None].expand(-1, 1, self.top_k),
                candidate_ids[:, :-1],
            ),
            dim=1,
        )
        predecessors = self.predecessor_codebook[predecessor_ids]
        pairwise = torch.einsum(
            "blpr,blcr->blpc", predecessors * hidden[:, :, None], successors
        )
        return unary_logits[:, :, None, :] + pairwise.float()


class DFlash2DraftModel(nn.Module):
    """Qwen3 DFlash2 backbone and low-rank path selector."""

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__()
        del prefix
        self.config = config
        self.dflash_config = DFlash2Config.from_hf_config(config)
        hidden_size = int(config.hidden_size)
        self.layers = nn.ModuleList(
            [
                DFlash2DecoderLayer(config, layer_id, quant_config=quant_config)
                for layer_id in range(int(config.num_hidden_layers))
            ]
        )
        self.norm = RMSNorm(hidden_size, eps=float(config.rms_norm_eps))
        self.fc = nn.Linear(
            len(self.dflash_config.target_layer_ids) * hidden_size,
            hidden_size,
            bias=False,
        )
        self.speculative_hidden_size = self.fc.out_features
        self.hidden_norm = RMSNorm(hidden_size, eps=float(config.rms_norm_eps))
        self.candidate_selector = DFlash2CandidateSelector(
            hidden_size,
            int(config.vocab_size),
            self.dflash_config.selector_rank,
            self.dflash_config.selector_top_k,
        )
        self.register_buffer("_flat_kv_weight_t", None, persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del input_ids
        if input_embeds is None:
            raise ValueError("DFlash2 must receive embeddings from the target model.")
        hidden_states = input_embeds
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions, hidden_states, forward_batch, residual
            )
        if residual is None:
            return self.norm(hidden_states)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def compute_candidates(
        self, hidden_states: torch.Tensor, target_head_weight: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = torch.matmul(
            hidden_states.to(target_head_weight.dtype), target_head_weight.T
        )
        if _flashinfer_top_k is None:
            values, ids = torch.topk(
                logits, self.dflash_config.selector_top_k, dim=-1, sorted=True
            )
        else:
            values, ids = _flashinfer_top_k(
                logits,
                self.dflash_config.selector_top_k,
                sorted=True,
                deterministic=True,
            )
        return ids.to(torch.int64), values.float()

    def project_target_hidden(self, target_hidden: torch.Tensor) -> torch.Tensor:
        if target_hidden.ndim != 2 or target_hidden.shape[1] != self.fc.in_features:
            raise ValueError(
                "DFlash2 target hidden feature mismatch: expected [N, "
                f"{self.fc.in_features}], got {tuple(target_hidden.shape)}."
            )
        return self.hidden_norm(self.fc(target_hidden))

    def _finalize_kv_projection(self) -> None:
        kv_weights = []
        for layer in self.layers:
            attention = layer.self_attn
            weight = attention.qkv_proj.weight
            kv_weights.append(
                weight[attention.q_size : attention.q_size + 2 * attention.kv_size]
            )
        stacked = torch.stack(kv_weights).reshape(-1, int(self.config.hidden_size))
        self._flat_kv_weight_t = stacked.T.contiguous()

    def materialize_target_kv(
        self,
        target_hidden: torch.Tensor,
        positions: torch.Tensor,
        cache_locs: torch.Tensor,
        kv_buffers,
    ) -> None:
        self.materialize_context_kv(
            self.project_target_hidden(target_hidden),
            positions,
            cache_locs,
            kv_buffers,
        )

    def materialize_context_kv(
        self,
        context: torch.Tensor,
        positions: torch.Tensor,
        cache_locs: torch.Tensor,
        kv_buffers,
    ) -> None:
        if self._flat_kv_weight_t is None:
            raise RuntimeError("DFlash2 KV projection was not finalized after loading.")
        if context.ndim != 2 or context.shape[1] != int(self.config.hidden_size):
            raise ValueError(
                "DFlash2 projected context must have shape [N, hidden_size]."
            )
        raw_kv = torch.matmul(context, self._flat_kv_weight_t)
        raw_kv = raw_kv.view(
            context.shape[0], len(self.layers), 2 * self.layers[0].self_attn.kv_size
        )
        for layer_id, layer in enumerate(self.layers):
            layer.self_attn.materialize_kv(
                raw_kv[:, layer_id], positions, cache_locs, kv_buffers[layer_id]
            )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        stacked_params = (
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        )
        ignored = {"embed_tokens.weight", "lm_head.weight"}
        params = dict(self.named_parameters())
        loaded_directly = set()
        loaded_shards = defaultdict(set)
        required_shards = {
            "qkv_proj": {"q", "k", "v"},
            "gate_up_proj": {0, 1},
        }

        for name, loaded_weight in weights:
            if name.startswith("model."):
                name = name[len("model.") :]
            if name in ignored:
                continue
            for packed_name, checkpoint_name, shard_id in stacked_params:
                if f".{checkpoint_name}." not in f".{name}":
                    continue
                mapped_name = name.replace(checkpoint_name, packed_name)
                if mapped_name not in params:
                    raise KeyError(f"Unexpected DFlash2 weight {name!r}.")
                if mapped_name in loaded_directly:
                    raise RuntimeError(
                        f"DFlash2 parameter {mapped_name!r} is present both fused "
                        "and as separate checkpoint shards."
                    )
                if shard_id in loaded_shards[mapped_name]:
                    raise RuntimeError(
                        f"Duplicate DFlash2 shard {shard_id!r} for {mapped_name!r}."
                    )
                parameter = params[mapped_name]
                loader = getattr(parameter, "weight_loader", default_weight_loader)
                loader(parameter, loaded_weight, shard_id)
                loaded_shards[mapped_name].add(shard_id)
                break
            else:
                if name not in params:
                    raise KeyError(f"Unexpected DFlash2 weight {name!r}.")
                if name in loaded_directly or name in loaded_shards:
                    raise RuntimeError(f"Duplicate DFlash2 parameter {name!r}.")
                parameter = params[name]
                loader = getattr(parameter, "weight_loader", default_weight_loader)
                loader(parameter, loaded_weight)
                loaded_directly.add(name)

        incomplete = []
        for name, shards in loaded_shards.items():
            packed_name = next(
                packed for packed in required_shards if f".{packed}." in f".{name}"
            )
            expected = required_shards[packed_name]
            if shards != expected:
                incomplete.append(
                    f"{name}: missing={sorted(expected - shards, key=str)}, "
                    f"loaded={sorted(shards, key=str)}"
                )
        if incomplete:
            raise RuntimeError(
                "DFlash2 checkpoint has incomplete packed parameters: "
                + "; ".join(sorted(incomplete))
            )

        loaded = loaded_directly | set(loaded_shards)
        missing = sorted(set(params) - loaded)
        if missing:
            raise RuntimeError(f"DFlash2 checkpoint is missing weights: {missing}.")
        self._finalize_kv_projection()


EntryClass = DFlash2DraftModel
