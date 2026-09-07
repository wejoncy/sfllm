"""Inference-only, text-path implementation of dense Qwen3.5.

Qwen3.5 alternates full attention with Gated DeltaNet linear-attention
blocks.  The public checkpoint is a multimodal wrapper; SFLLM intentionally
loads only its language model here and skips the vision encoder and MTP head.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import torch
from torch import nn
import sf_kernel

from sfllm.engine.forward_params import ForwardBatch, ForwardMode
from sfllm.kernels.gdn import GatedDeltaNetBackend, gated_rmsnorm
from sfllm.layers.layernorm import GemmaRMSNorm
from sfllm.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sfllm.layers.logits_processor import LogitsProcessor
from sfllm.layers.quantization import QuantizationConfig
from sfllm.layers.radix_attention import RadixAttention
from sfllm.layers.rotary_embedding import get_rope
from sfllm.model_loader.model_config import get_pool_index_layers
from sfllm.model_loader.weight_utils import default_weight_loader
from sfllm.models.interfaces import HasBatchState
from sfllm.models.qwen2 import Qwen2MLP
from sfllm.server_args import get_global_server_args
from sfllm.utils import add_prefix, make_layers_non_pp

class Qwen3_5GatedDeltaNet(nn.Module):
    def __init__(
        self,
        config,
        conv_states: torch.Tensor,
        ssm_states: torch.Tensor,
        state_indices: torch.Tensor,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_k_heads = config.linear_num_key_heads
        self.num_v_heads = config.linear_num_value_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.rms_norm_eps = config.rms_norm_eps
        self.conv_states = conv_states
        self.ssm_states = ssm_states
        self.state_indices = state_indices

        self.in_proj_qkvzba = MergedColumnParallelLinear(
            self.hidden_size,
            [
                self.key_dim,
                self.key_dim,
                self.value_dim,
                self.value_dim,
                self.num_v_heads,
                self.num_v_heads,
            ],
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("in_proj_qkvzba", prefix),
        )
        self.conv_weight = nn.Parameter(
            torch.empty(self.conv_dim, 1, self.conv_kernel_size), requires_grad=False
        )
        self.A_log = nn.Parameter(
            torch.empty(self.num_v_heads, dtype=torch.float32), requires_grad=False
        )
        self.dt_bias = nn.Parameter(
            torch.empty(self.num_v_heads), requires_grad=False
        )
        self.norm = nn.Parameter(
            torch.ones(self.head_v_dim, dtype=torch.float32), requires_grad=False
        )
        self.out_proj = RowParallelLinear(
            self.value_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("out_proj", prefix),
        )
        server_args = get_global_server_args()
        self.backend = GatedDeltaNetBackend(
            prefill_backend=(
                server_args.linear_attn_prefill_backend
                or server_args.linear_attn_backend
            ),
            decode_backend=(
                server_args.linear_attn_decode_backend
                or server_args.linear_attn_backend
            ),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        query_start_loc: Optional[torch.Tensor],
        query_start_loc_i64: Optional[torch.Tensor],
    ):
        num_sequences = (
            hidden_states.shape[0]
            if forward_batch.forward_mode == ForwardMode.DECODE
            else query_start_loc.shape[0] - 1
        )
        state_indices = self.state_indices[:num_sequences]

        projected, _ = self.in_proj_qkvzba(hidden_states)
        projected_qkvz, projected_ba = projected.split(
            (self.conv_dim + self.value_dim, self.num_v_heads * 2), dim=-1
        )

        common = dict(
            conv_weight=self.conv_weight.view(self.conv_dim, self.conv_kernel_size),
            conv_states=self.conv_states,
            ssm_states=self.ssm_states,
            state_indices=state_indices,
            a_log=self.A_log,
            dt_bias=self.dt_bias,
            num_k_heads=self.num_k_heads,
            num_v_heads=self.num_v_heads,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
        )
        if forward_batch.forward_mode == ForwardMode.DECODE:
            core, z = self.backend.decode(projected_qkvz, projected_ba, **common)
        elif forward_batch.forward_mode == ForwardMode.EXTEND:
            mixed_qkv = projected_qkvz[:, : self.conv_dim]
            z = projected_qkvz[:, self.conv_dim :].view(
                -1, self.num_v_heads, self.head_v_dim
            )
            b, a = projected_ba.split(self.num_v_heads, dim=-1)
            core = self.backend.prefill(
                mixed_qkv,
                a,
                b,
                query_start_loc=query_start_loc,
                query_start_loc_i64=query_start_loc_i64,
                **common,
            )
        else:
            raise NotImplementedError(
                f"Qwen3.5 GDN does not support {forward_batch.forward_mode.name} yet"
            )

        core = gated_rmsnorm(core, z, self.norm, self.rms_norm_eps).view(
            hidden_states.shape[0], self.value_dim
        )
        output, _ = self.out_proj(core)
        return output


class Qwen3_5Attention(nn.Module):
    def __init__(
        self,
        config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.attn_output_gate = config.attn_output_gate
        rope = config.rope_parameters

        q_heads = self.num_heads * (2 if self.attn_output_gate else 1)
        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            q_heads,
            self.num_kv_heads,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.q_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=False,
            prefix=add_prefix("o_proj", prefix),
        )
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=rope.get("rope_theta", 10000000),
            is_neox_style=True,
            rope_scaling=rope,
            dtype=torch.get_default_dtype(),
            partial_rotary_factor=rope["partial_rotary_factor"],
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            prefix=add_prefix("attn", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        if self.attn_output_gate:
            q_gate, k, v = qkv.split(
                (self.q_size * 2, self.kv_size, self.kv_size), dim=-1
            )
        else:
            q_gate, k, v = qkv.split(
                (self.q_size, self.kv_size, self.kv_size), dim=-1
            )

        tokens = q_gate.shape[0]
        q_gate = q_gate.view(
            tokens, self.num_heads, self.head_dim * (2 if self.attn_output_gate else 1)
        )
        q = q_gate[..., :self.head_dim]
        gate = q_gate[..., self.head_dim:] if self.attn_output_gate else None
        k = k.view(tokens, self.num_kv_heads, self.head_dim)
        q_out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        k_out = torch.empty(k.shape, dtype=k.dtype, device=k.device)
        k_cache = v_cache = None
        if forward_batch.past_key_values is not None:
            k_cache, v_cache = forward_batch.past_key_values[self.attn.layer_id]

        sf_kernel.gemma_qk_norm_rope(
            q,
            k,
            v.view(tokens, self.num_kv_heads, self.head_dim),
            q_out,
            k_out,
            self.q_norm.weight,
            self.k_norm.weight,
            self.rotary_emb.cos_sin_cache,
            positions,
            k_cache,
            v_cache,
            forward_batch.out_cache_loc,
            self.q_norm.variance_epsilon,
        )
        output = self.attn(
            q_out.view(tokens, -1), k_out.view(tokens, -1), v,
            forward_batch, save_kv_cache=False,
        )
        if gate is not None:
            output = output.view(-1, self.q_size)
            sf_kernel.fused_sigmoid_mul(output, gate)
        output, _ = self.o_proj(output)
        return output


class Qwen3_5DecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        layer_id: int,
        cache_layer_id: Optional[int],
        state_layer_id: Optional[int],
        state_buffers: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        layer_type = config.layer_types[layer_id]
        self.layer_type = layer_type
        if layer_type == "full_attention":
            self.self_attn = Qwen3_5Attention(
                config,
                cache_layer_id,
                quant_config,
                add_prefix("self_attn", prefix),
            )
        elif layer_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(
                config,
                state_buffers[0][state_layer_id],
                state_buffers[1][state_layer_id],
                state_buffers[2],
                quant_config,
                add_prefix("linear_attn", prefix),
            )
        else:
            raise ValueError(f"Unsupported Qwen3.5 layer type: {layer_type}")
        self.mlp = Qwen2MLP(
            config.hidden_size,
            config.intermediate_size,
            config.hidden_act,
            quant_config,
            add_prefix("mlp", prefix),
        )
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        query_start_loc: Optional[torch.Tensor],
        query_start_loc_i64: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        if self.layer_type == "full_attention":
            hidden_states = self.self_attn(positions, hidden_states, forward_batch)
        else:
            hidden_states = self.linear_attn(
                hidden_states,
                forward_batch,
                query_start_loc,
                query_start_loc_i64,
            )
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual
        )
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3_5Model(nn.Module):
    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, _freeze=True)
        server_args = get_global_server_args()
        linear_attention_layer_ids = [
            i for i, layer_type in enumerate(config.layer_types)
            if layer_type == "linear_attention"
        ]
        cache_layer_ids = {
            layer_id: cache_id
            for cache_id, layer_id in enumerate(get_pool_index_layers(config))
        }
        state_layer_ids = {
            layer_id: state_id
            for state_id, layer_id in enumerate(linear_attention_layer_ids)
        }

        max_state_rows = int(server_args.max_running_requests)
        conv_dim = (
            2 * config.linear_num_key_heads * config.linear_key_head_dim
            + config.linear_num_value_heads * config.linear_value_head_dim
        )
        self.register_buffer(
            "conv_states",
            torch.zeros(
                len(linear_attention_layer_ids),
                max_state_rows + 1,
                conv_dim,
                config.linear_conv_kernel_dim - 1,
            ),
            persistent=False,
        )
        self.register_buffer(
            "ssm_states",
            torch.zeros(
                len(linear_attention_layer_ids),
                max_state_rows + 1,
                config.linear_num_value_heads,
                config.linear_value_head_dim,
                config.linear_key_head_dim,
                dtype={"float32": torch.float32, "bfloat16": torch.bfloat16}[
                    vars(config).get("mamba_ssm_dtype", "float32")
                ],
            ),
            persistent=False,
        )
        self.register_buffer(
            "state_indices",
            torch.arange(1, max_state_rows + 1, dtype=torch.int32),
            persistent=False,
        )
        self.prepared_state_batch = None
        state_buffers = (
            self.conv_states,
            self.ssm_states,
            self.state_indices,
        )
        self.layers, self.start_layer, self.end_layer = make_layers_non_pp(
            config.num_hidden_layers,
            lambda idx, prefix: Qwen3_5DecoderLayer(
                config,
                idx,
                cache_layer_ids.get(idx),
                state_layer_ids.get(idx),
                state_buffers,
                quant_config,
                prefix,
            ),
            prefix=add_prefix("layers", prefix),
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def prepare_batch_state(self, scheduled_batch) -> None:
        batch_size = len(scheduled_batch)
        padded_batch_size = batch_size + scheduled_batch.forward_batch.padded_token
        indices = [sequence.request_index + 1 for sequence in scheduled_batch]
        batch_key = (padded_batch_size, tuple(indices))
        if batch_key == self.prepared_state_batch:
            return
        indices.extend([-1] * scheduled_batch.forward_batch.padded_token)
        # The pinned allocator retains storage until the async copy completes.
        self.state_indices[:padded_batch_size].copy_(
            torch.tensor(indices, dtype=torch.int32, device="cpu", pin_memory=True),
            non_blocking=True,
        )
        self.prepared_state_batch = batch_key

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids) if input_embeds is None else input_embeds
        residual = None
        query_start_loc = None
        query_start_loc_i64 = None
        if forward_batch.forward_mode == ForwardMode.EXTEND:
            query_start_loc = forward_batch.qo_indptr
            if query_start_loc.shape[0] == 1:
                query_start_loc = query_start_loc.new_tensor(
                    (0, hidden_states.shape[0])
                )
            query_start_loc_i64 = query_start_loc.to(torch.int64)
        for layer in self.layers:
            hidden_states, residual = layer(
                positions,
                hidden_states,
                forward_batch,
                residual,
                query_start_loc,
                query_start_loc_i64,
            )
        if residual is not None:
            hidden_states, _ = self.norm(hidden_states, residual)
        else:
            hidden_states = self.norm(hidden_states)
        return hidden_states


class Qwen3_5ForConditionalGeneration(nn.Module, HasBatchState):
    """Text-only serving view of a dense Qwen3.5 multimodal checkpoint."""

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkvzba": [
            "in_proj_qkv",
            "in_proj_z",
            "in_proj_b",
            "in_proj_a",
        ],
    }

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if quant_config is not None:
            raise ValueError("Initial Qwen3.5 support is BF16-only")
        text_config = config.text_config
        self.config = text_config
        self.model = Qwen3_5Model(text_config, quant_config, add_prefix("model", prefix))
        self.lm_head = self.model.embed_tokens if text_config.tie_word_embeddings else ReplicatedLinear(
            text_config.hidden_size, text_config.vocab_size, bias=False
        )
        self.logits_processor = LogitsProcessor(text_config)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def prepare_batch_state(self, scheduled_batch) -> None:
        self.model.prepare_batch_state(scheduled_batch)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        get_embedding: bool = False,
    ):
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        if get_embedding:
            return hidden_states, forward_batch
        return self.logits_processor(hidden_states, self.lm_head, None, forward_batch)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params = dict(self.named_parameters())
        loaded = set()

        for checkpoint_name, loaded_weight in weights:
            if checkpoint_name.startswith("model.visual.") or checkpoint_name.startswith("mtp."):
                continue
            name = checkpoint_name.replace("model.language_model.", "model.", 1)
            if self.config.tie_word_embeddings and name == "lm_head.weight":
                continue
            if ".linear_attn.norm.weight" in name:
                name = name.replace(".linear_attn.norm.weight", ".linear_attn.norm")
            if ".linear_attn.conv1d.weight" in name:
                name = name.replace(".linear_attn.conv1d.weight", ".linear_attn.conv_weight")

            mappings = []
            if ".linear_attn.in_proj_qkv.weight" in name:
                chunks = loaded_weight.split(
                    [
                        self.config.linear_num_key_heads * self.config.linear_key_head_dim,
                        self.config.linear_num_key_heads * self.config.linear_key_head_dim,
                        self.config.linear_num_value_heads * self.config.linear_value_head_dim,
                    ],
                    dim=0,
                )
                target = name.replace("in_proj_qkv", "in_proj_qkvzba")
                mappings = [(target, chunk, idx) for idx, chunk in enumerate(chunks)]
            elif ".linear_attn.in_proj_z.weight" in name:
                mappings = [(name.replace("in_proj_z", "in_proj_qkvzba"), loaded_weight, 3)]
            elif ".linear_attn.in_proj_b.weight" in name:
                mappings = [(name.replace("in_proj_b", "in_proj_qkvzba"), loaded_weight, 4)]
            elif ".linear_attn.in_proj_a.weight" in name:
                mappings = [(name.replace("in_proj_a", "in_proj_qkvzba"), loaded_weight, 5)]
            else:
                stacked = [
                    ("qkv_proj", "q_proj", "q"),
                    ("qkv_proj", "k_proj", "k"),
                    ("qkv_proj", "v_proj", "v"),
                    ("gate_up_proj", "gate_proj", 0),
                    ("gate_up_proj", "up_proj", 1),
                ]
                for packed_name, shard_name, shard_id in stacked:
                    if shard_name in name:
                        mappings = [(name.replace(shard_name, packed_name), loaded_weight, shard_id)]
                        break
                if not mappings:
                    mappings = [(name, loaded_weight, None)]

            for target_name, tensor, shard_id in mappings:
                param = params.get(target_name)
                if param is None:
                    raise KeyError(
                        f"Qwen3.5 language weight has no destination: {checkpoint_name} -> {target_name}"
                    )
                loader = vars(param).get("_weight_loader", default_weight_loader)
                if shard_id is None:
                    loader(param, tensor)
                else:
                    loader(param, tensor, shard_id)
                loaded.add(target_name)

        missing = set(params) - loaded
        # Tied embeddings appear only once in named_parameters().
        if self.config.tie_word_embeddings:
            missing.discard("lm_head.weight")
        if missing:
            raise RuntimeError(f"Missing Qwen3.5 language weights: {sorted(missing)}")


EntryClass = Qwen3_5ForConditionalGeneration
