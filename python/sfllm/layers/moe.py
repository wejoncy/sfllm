"""Unquantized MoE layers backed by FlashInfer or vendored Triton kernels."""

import torch
from torch import nn

from sfllm.kernels.moe import fused_experts, route_tokens


class RoutedExperts(nn.Module):
    def __init__(self, num_experts: int, hidden_size: int, intermediate_size: int):
        super().__init__()
        if min(num_experts, hidden_size, intermediate_size) <= 0:
            raise ValueError("Expert counts and dimensions must be positive")
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        # Preserve the checkpoint's packed [gate, up] and [output, input] layouts.
        self.gate_up_proj = nn.Parameter(torch.empty(
            num_experts, 2 * intermediate_size, hidden_size,
        ), requires_grad=False)
        self.down_proj = nn.Parameter(torch.empty(
            num_experts, hidden_size, intermediate_size,
        ), requires_grad=False)

    def forward(self, hidden_states, routing):
        return fused_experts(hidden_states, routing, self.gate_up_proj, self.down_proj)


class FusedMoE(nn.Module):
    backend = "triton_kernel"

    def __init__(self, hidden_size, intermediate_size, num_experts, top_k):
        super().__init__()
        if not 0 < top_k <= num_experts:
            raise ValueError("num_experts_per_tok must be within [1, num_experts]")
        if top_k & (top_k - 1) or top_k > 32:
            raise ValueError("triton_kernel requires power-of-two top_k up to 32")
        self.top_k = top_k
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = RoutedExperts(num_experts, hidden_size, intermediate_size)
        self.requires_grad_(False)

    def forward(self, hidden_states):
        if not hidden_states.is_cuda or hidden_states.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("MoE requires CUDA FP16 or BF16 inputs")
        if hidden_states.ndim != 2 or hidden_states.shape[1] != self.experts.hidden_size:
            raise ValueError("Expected packed [tokens, hidden_size] MoE inputs")
        if not hidden_states.shape[0]:
            return torch.empty_like(hidden_states)
        routing = route_tokens(hidden_states, self.gate.weight, self.top_k)
        return self.experts(hidden_states, routing)


class CutlassMoE(nn.Module):
    """Pack the shared expert into the same library call as routed experts."""

    def __init__(self, hidden_size, intermediate_size, num_experts, top_k):
        super().__init__()
        from sfllm.kernels.moe import CutlassBackend

        if min(hidden_size, intermediate_size, num_experts) <= 0:
            raise ValueError("Expert counts and dimensions must be positive")
        if hidden_size % 8 or intermediate_size % 8:
            raise ValueError("CUTLASS requires hidden and intermediate sizes divisible by 8")
        if not 0 < top_k <= min(num_experts, 32) or top_k & (top_k - 1):
            raise ValueError("CUTLASS routing requires power-of-two top_k up to 32")
        self.top_k = top_k
        self.num_experts = num_experts
        self.intermediate_size = intermediate_size
        # An E+1-wide router makes cuBLAS use unaligned kernels (257 for Qwen).
        # Align storage to 16 columns, while routing only over the real E experts.
        router_width = ((num_experts + 1 + 15) // 16) * 16
        self.gate = nn.Linear(hidden_size, router_width, bias=False)
        self.gate.weight.data[num_experts + 1:].zero_()
        self.gate.weight._weight_loader = self.load_router
        self.up_gate_proj = nn.Parameter(torch.empty(
            num_experts + 1, 2 * intermediate_size, hidden_size,
        ))
        self.down_proj = nn.Parameter(torch.empty(
            num_experts + 1, hidden_size, intermediate_size,
        ))
        self.up_gate_proj._weight_loader = self.load_gate_up
        self.down_proj._weight_loader = self.load_down
        self.backend = CutlassBackend()
        self.requires_grad_(False)

    def load_router(self, param, weight):
        if weight.shape != param[:self.num_experts].shape:
            raise ValueError("Invalid routed router weight shape")
        param.data[:self.num_experts].copy_(weight)

    def load_gate_up(self, param, weight):
        if weight.shape != param[:self.num_experts].shape:
            raise ValueError("Invalid routed gate/up weight shape")
        gate, up = weight.chunk(2, dim=1)
        param.data[:self.num_experts, :self.intermediate_size].copy_(up)
        param.data[:self.num_experts, self.intermediate_size:].copy_(gate)

    def load_down(self, param, weight):
        if weight.shape != param[:self.num_experts].shape:
            raise ValueError("Invalid routed down weight shape")
        param.data[:self.num_experts].copy_(weight)

    def load_shared_weight(self, component, weight):
        width = self.intermediate_size
        target = {
            "shared_expert.gate_proj.weight": self.up_gate_proj[-1, width:],
            "shared_expert.up_proj.weight": self.up_gate_proj[-1, :width],
            "shared_expert.down_proj.weight": self.down_proj[-1],
            "shared_expert_gate.weight": self.gate.weight[self.num_experts:self.num_experts + 1],
        }[component]
        if target.shape != weight.shape:
            raise ValueError(f"Invalid {component} shape")
        target.data.copy_(weight)

    def forward(self, x):
        if not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("MoE requires CUDA FP16 or BF16 inputs")
        if x.ndim != 2 or x.shape[1] != self.gate.in_features:
            raise ValueError("Expected packed [tokens, hidden_size] MoE inputs")
        if not len(x):
            return torch.empty_like(x)
        return self.backend(x, self.gate.weight, self.up_gate_proj, self.down_proj, self.top_k)
