"""Text inference for Qwen3.5 MoE, sharing attention and state with dense Qwen3.5."""

from importlib.util import find_spec

import torch
from torch import nn

from sfllm.layers.moe import CutlassMoE, FusedMoE
from sfllm.model_loader.weight_utils import get_layer_id
from sfllm.models.qwen2 import Qwen2MLP
from sfllm.models.qwen3_5 import (
    Qwen3_5DecoderLayer, Qwen3_5ForConditionalGeneration, Qwen3_5Model,
)
from sfllm.server_args import get_global_server_args


def _cutlass_available():
    return find_spec("flashinfer") is not None


class Qwen3_5SparseMoeBlock(FusedMoE):
    shared_components = (
        "shared_expert.gate_proj.weight", "shared_expert.up_proj.weight",
        "shared_expert.down_proj.weight", "shared_expert_gate.weight",
    )

    def __init__(self, config, prefix=""):
        if config.hidden_act != "silu":
            raise ValueError("Qwen3.5 MoE experts require SiLU")
        super().__init__(
            config.hidden_size, config.moe_intermediate_size,
            config.num_experts, config.num_experts_per_tok,
        )
        self.shared_expert = Qwen2MLP(
            config.hidden_size, config.shared_expert_intermediate_size,
            config.hidden_act, prefix=f"{prefix}.shared_expert",
        )
        self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)
        self.requires_grad_(False)

    def forward(self, hidden_states):
        routed = super().forward(hidden_states)
        if not hidden_states.shape[0]:
            return routed
        shared = self.shared_expert(hidden_states)
        return routed + torch.sigmoid(self.shared_expert_gate(hidden_states)) * shared


class Qwen3_5MoeDecoderLayer(Qwen3_5DecoderLayer):
    def build_mlp(self, config, quant_config, prefix):
        backend = get_global_server_args().moe_runner_backend
        use_cutlass = backend == "flashinfer_cutlass" or (
            backend == "auto" and torch.cuda.is_available()
            and torch.cuda.get_device_capability()[0] == 9
            and config.shared_expert_intermediate_size == config.moe_intermediate_size
            and _cutlass_available()
        )
        if use_cutlass:
            if not _cutlass_available():
                raise ImportError(
                    "flashinfer_cutlass requires sfllm[flashinfer] "
                    "(FlashInfer). "
                    "Select --moe-runner-backend triton_kernel for bundled Triton kernels."
                )
            if config.hidden_act != "silu" or config.shared_expert_intermediate_size != config.moe_intermediate_size:
                raise ValueError("Fused CUTLASS requires SiLU and equal shared/routed expert widths")
            return CutlassMoE(
                config.hidden_size, config.moe_intermediate_size,
                config.num_experts, config.num_experts_per_tok,
            ), ()
        return Qwen3_5SparseMoeBlock(config, prefix), ()


class Qwen3_5MoeModel(Qwen3_5Model):
    decoder_layer_class = Qwen3_5MoeDecoderLayer

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        first = self.layers[0].mlp
        if isinstance(first, CutlassMoE):
            for layer in self.layers:
                layer.mlp.backend = first.backend


class Qwen3_5MoeForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    model_class = Qwen3_5MoeModel

    def __init__(self, config, quant_config=None, prefix=""):
        if quant_config is not None:
            raise ValueError("Qwen3.5 MoE currently supports unquantized BF16/FP16 checkpoints")
        if getattr(config.text_config, "mlp_only_layers", None):
            raise ValueError("Qwen3.5 MoE checkpoints with dense MLP layers are not supported")
        super().__init__(config, quant_config, prefix)

    def load_weights(self, weights):
        shared_loaded = set()

        def track_shared_weights():
            for name, weight in weights:
                if ".mlp." in name and name.startswith("model.language_model."):
                    component = name.split(".mlp.", 1)[1]
                    layer_id = get_layer_id(name)
                    block = self.model.layers[layer_id].mlp
                    if component in Qwen3_5SparseMoeBlock.shared_components:
                        shared_loaded.add((layer_id, component))
                        if isinstance(block, CutlassMoE):
                            block.load_shared_weight(component, weight)
                            continue
                    if isinstance(block, CutlassMoE):
                        name = name.replace(".experts.gate_up_proj", ".up_gate_proj")
                        name = name.replace(".experts.down_proj", ".down_proj")
                yield name, weight

        # Dense Qwen3.5 already handles packed tensors and shared MLP shards.
        super().load_weights(track_shared_weights())
        missing = {
            (layer_id, component)
            for layer_id in range(len(self.model.layers))
            for component in Qwen3_5SparseMoeBlock.shared_components
        } - shared_loaded
        if missing:
            raise RuntimeError(f"Missing Qwen3.5 shared expert weights: {sorted(missing)}")
        first = self.model.layers[0].mlp
        if isinstance(first, CutlassMoE) and first.gate.weight.is_cuda:
            first.backend.prepare(first.up_gate_proj, first.down_proj, first.top_k + 1)


EntryClass = Qwen3_5MoeForConditionalGeneration
