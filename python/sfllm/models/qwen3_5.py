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
from sfllm.kernels.gdn import (
    GatedDeltaNetBackend, gated_rmsnorm, scatter_recurrent_state,
    update_recurrent_state_indices,
)
from sfllm.layers.layernorm import GemmaRMSNorm
from sfllm.layers.quantization.fp8_kernel import (
    rmsnorm_silu_gate_quant_fp8,
    sigmoid_mul_quant_fp8,
)
from sfllm.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sfllm.layers.logits_processor import LogitsProcessor
from sfllm.layers.quantization import Fp8Config, QuantizationConfig
from sfllm.layers.quantization.utils import is_layer_skipped
from sfllm.layers.radix_attention import RadixAttention
from sfllm.layers.rotary_embedding import get_rope
from sfllm.model_loader.model_config import get_pool_index_layers
from sfllm.model_loader.weight_utils import default_weight_loader, get_layer_id
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
        self.intermediate_conv = None
        self.ssm_state_indices = None
        self.ssm_output_indices = None
        self.fused_in_proj = (
            quant_config is None or quant_config.weight_strategy == "channel"
        )
        if quant_config is not None and self.fused_in_proj:
            self.fused_in_proj = is_layer_skipped(
                add_prefix("in_proj_qkvz", prefix), quant_config.ignored_layers,
                quant_config.packed_modules_mapping,
            ) == is_layer_skipped(
                add_prefix("in_proj_ba", prefix), quant_config.ignored_layers,
                quant_config.packed_modules_mapping,
            )

        qkvz_sizes = [self.key_dim, self.key_dim, self.value_dim, self.value_dim]
        ba_sizes = [self.num_v_heads, self.num_v_heads]
        if self.fused_in_proj:
            self.in_proj_qkvzba = MergedColumnParallelLinear(
                self.hidden_size, qkvz_sizes + ba_sizes, bias=False,
                quant_config=quant_config,
                prefix=add_prefix("in_proj_qkvzba", prefix),
            )
        else:
            # ModelOpt keeps the recurrent gates in BF16 by default.
            self.in_proj_qkvz = MergedColumnParallelLinear(
                self.hidden_size, qkvz_sizes, bias=False, quant_config=quant_config,
                prefix=add_prefix("in_proj_qkvz", prefix),
            )
            self.in_proj_ba = MergedColumnParallelLinear(
                self.hidden_size, ba_sizes, bias=False, quant_config=quant_config,
                prefix=add_prefix("in_proj_ba", prefix),
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
        self.fp8_output = self.out_proj.quant_method.input_dtype is not None
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
        mode = forward_batch.forward_mode
        if self.fused_in_proj:
            projected, _ = self.in_proj_qkvzba(hidden_states)
            projected_qkvz, projected_ba = projected.split(
                (self.conv_dim + self.value_dim, self.num_v_heads * 2), dim=-1
            )
        else:
            qkvz_input, ba_input = hidden_states
            projected_qkvz, _ = self.in_proj_qkvz(qkvz_input)
            projected_ba, _ = self.in_proj_ba(ba_input)

        num_tokens = projected_qkvz.shape[0]
        num_sequences = num_tokens
        if mode == ForwardMode.EXTEND:
            num_sequences = query_start_loc.shape[0] - 1
        elif mode == ForwardMode.TARGET_VERIFY:
            num_sequences //= forward_batch.max_extend_len
        state_indices = self.state_indices[:num_sequences]

        common = dict(
            conv_weight=self.conv_weight.view(self.conv_dim, self.conv_kernel_size),
            conv_states=self.conv_states,
            ssm_states=self.ssm_states,
            state_indices=state_indices,
            ssm_state_indices=(
                self.ssm_state_indices[:num_sequences]
                if self.ssm_state_indices is not None else None
            ),
            a_log=self.A_log,
            dt_bias=self.dt_bias,
            num_k_heads=self.num_k_heads,
            num_v_heads=self.num_v_heads,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
        )
        if forward_batch.forward_mode == ForwardMode.DECODE:
            core, z = self.backend.decode(projected_qkvz, projected_ba, **common)
        elif mode == ForwardMode.TARGET_VERIFY:
            core, z = self.backend.decode(
                projected_qkvz, projected_ba,
                intermediate_conv=self.intermediate_conv[:num_sequences],
                ssm_output_indices=self.ssm_output_indices[:num_sequences],
                **common,
            )
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

        if self.fp8_output:
            core = rmsnorm_silu_gate_quant_fp8(
                core, weight=self.norm, gate=z, eps=self.rms_norm_eps,
                input_scale=self.out_proj.quant_method.input_scale,
            )
        else:
            core = gated_rmsnorm(core, z, self.norm, self.rms_norm_eps).view(
                num_tokens, self.value_dim
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
        self.fp8_output = self.o_proj.quant_method.input_dtype is not None
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
            if self.fp8_output:
                output = sigmoid_mul_quant_fp8(
                    output, gate=gate, input_scale=self.o_proj.quant_method.input_scale,
                )
            else:
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
            input_projections = (self.self_attn.qkv_proj,)
        elif layer_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(
                config,
                state_buffers[0][state_layer_id],
                state_buffers[1][state_layer_id],
                state_buffers[2],
                quant_config,
                add_prefix("linear_attn", prefix),
            )
            input_projections = (
                (self.linear_attn.in_proj_qkvzba,) if self.linear_attn.fused_in_proj
                else (self.linear_attn.in_proj_qkvz, self.linear_attn.in_proj_ba)
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
        self.input_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps,
            quant_methods=tuple(p.quant_method for p in input_projections),
        )
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps,
            quant_methods=(self.mlp.gate_up_proj.quant_method,),
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
        spec_steps = (
            server_args.speculative_num_draft_tokens
            if server_args.speculative_algorithm == "dflash2" else 0
        )
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
                max_state_rows * (spec_steps + 1) + 1,
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
        self.layers_to_capture = []
        self.register_buffer("ssm_output_indices", None, persistent=False)
        if spec_steps:
            self.register_buffer(
                "intermediate_conv",
                self.conv_states.new_empty((
                    self.conv_states.shape[0], max_state_rows, spec_steps,
                    *self.conv_states.shape[2:],
                )),
                persistent=False,
            )
            self.register_buffer(
                "ssm_current_slots",
                (torch.arange(max_state_rows + 1, dtype=torch.int32) - 1)
                * (spec_steps + 1) + 1,
                persistent=False,
            )
            self.register_buffer(
                "ssm_state_indices", torch.empty_like(self.state_indices), persistent=False,
            )
            self.ssm_output_indices = torch.empty(
                (max_state_rows, spec_steps), dtype=torch.int32
            )
            for layer_id, state_id in state_layer_ids.items():
                attn = self.layers[layer_id].linear_attn
                attn.intermediate_conv = self.intermediate_conv[state_id]
                attn.ssm_state_indices = self.ssm_state_indices
                attn.ssm_output_indices = self.ssm_output_indices

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
        aux_hidden_states = []
        query_start_loc = None
        query_start_loc_i64 = None
        if forward_batch.forward_mode == ForwardMode.EXTEND:
            query_start_loc = forward_batch.qo_indptr
            if query_start_loc.shape[0] == 1:
                query_start_loc = query_start_loc.new_tensor(
                    (0, hidden_states.shape[0])
                )
            query_start_loc_i64 = query_start_loc.to(torch.int64)
        if self.ssm_output_indices is not None:
            if forward_batch.forward_mode == ForwardMode.EXTEND:
                batch_size = query_start_loc.shape[0] - 1
            elif forward_batch.forward_mode == ForwardMode.TARGET_VERIFY:
                batch_size = hidden_states.shape[0] // forward_batch.max_extend_len
            else:
                batch_size = hidden_states.shape[0]
            update_recurrent_state_indices(
                self.ssm_current_slots, self.state_indices[:batch_size],
                self.ssm_state_indices, self.ssm_output_indices,
                prefill=forward_batch.forward_mode == ForwardMode.EXTEND,
            )
        for layer_id, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                positions,
                hidden_states,
                forward_batch,
                residual,
                query_start_loc,
                query_start_loc_i64,
            )
            if layer_id in self.layers_to_capture:
                aux_hidden_states.append(hidden_states + residual)
        if residual is not None:
            hidden_states, _ = self.norm(hidden_states, residual)
        else:
            hidden_states = self.norm(hidden_states)
        return (hidden_states, aux_hidden_states) if aux_hidden_states else hidden_states


class Qwen3_5ForConditionalGeneration(nn.Module, HasBatchState):
    """Text-only serving view of a dense Qwen3.5 multimodal checkpoint."""

    remap_prefix = {"model.language_model.": "model."}
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
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
        if quant_config is not None and (
            not isinstance(quant_config, Fp8Config) or quant_config.weight_block_size is not None
        ):
            raise ValueError("Qwen3.5 only supports per-tensor or per-channel FP8 quantization")
        text_config = config.text_config
        self.config = text_config
        self.quant_config = quant_config
        self.model = Qwen3_5Model(text_config, quant_config, add_prefix("model", prefix))
        self.lm_head = self.model.embed_tokens if text_config.tie_word_embeddings else ReplicatedLinear(
            text_config.hidden_size, text_config.vocab_size, bias=False
        )
        self.logits_processor = LogitsProcessor(text_config)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def prepare_batch_state(self, scheduled_batch) -> None:
        self.model.prepare_batch_state(scheduled_batch)

    def set_layers_to_capture(self, layer_ids) -> None:
        self.model.layers_to_capture = layer_ids

    def commit_speculative_state(self, accepted_steps: torch.Tensor) -> None:
        """Commit the anchor and accepted drafts; the bonus is the next anchor."""
        indices = self.model.state_indices[:accepted_steps.shape[0]]
        scatter_recurrent_state(
            self.model.intermediate_conv, self.model.conv_states,
            indices, accepted_steps,
        )
        update_recurrent_state_indices(
            self.model.ssm_current_slots, indices,
            self.model.ssm_state_indices, self.model.ssm_output_indices,
            accepted_steps=accepted_steps,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        get_embedding: bool = False,
    ):
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        aux_hidden_states = None
        if self.model.layers_to_capture:
            hidden_states, aux_hidden_states = hidden_states
        if get_embedding:
            return hidden_states, forward_batch
        return self.logits_processor(hidden_states, self.lm_head, aux_hidden_states, forward_batch)

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
            if ".linear_attn.in_proj_" in name:
                separate = not self.model.layers[get_layer_id(name)].linear_attn.fused_in_proj
                qkvz_proj = "in_proj_qkvz" if separate else "in_proj_qkvzba"
                ba_proj = "in_proj_ba" if separate else "in_proj_qkvzba"
                ba_offset = 0 if separate else 4

            mappings = []
            if ".linear_attn.in_proj_qkv." in name:
                chunks = loaded_weight.split(
                    [
                        self.config.linear_num_key_heads * self.config.linear_key_head_dim,
                        self.config.linear_num_key_heads * self.config.linear_key_head_dim,
                        self.config.linear_num_value_heads * self.config.linear_value_head_dim,
                    ],
                    dim=0,
                ) if name.endswith(".weight") or (
                    name.endswith(".weight_scale")
                    and self.quant_config.weight_strategy == "channel"
                ) else [loaded_weight] * 3
                target = name.replace("in_proj_qkv", qkvz_proj)
                mappings = [(target, chunk, idx) for idx, chunk in enumerate(chunks)]
            elif ".linear_attn.in_proj_z." in name:
                mappings = [(name.replace("in_proj_z", qkvz_proj), loaded_weight, 3)]
            elif ".linear_attn.in_proj_b." in name:
                mappings = [(name.replace("in_proj_b", ba_proj), loaded_weight, ba_offset)]
            elif ".linear_attn.in_proj_a." in name:
                mappings = [(name.replace("in_proj_a", ba_proj), loaded_weight, ba_offset + 1)]
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
