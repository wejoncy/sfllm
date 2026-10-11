"""Reuse attention backends with group-specific addresses and visible lengths."""

from copy import copy

import torch

from sfllm.kernels.cache_indices import resolve_window
from sfllm.layers.radix_attention import create_attention_backend


class GroupedAttentionBackend:
    def __init__(self, runner, pool):
        self.groups = []
        for group in pool.groups:
            metadata = {layer: runner.layer_metadata[layer] for layer in group.layers}
            backend = create_attention_backend(runner, metadata)
            indices = None
            if group.ring is not None:
                indices = torch.empty(
                    runner.server_args.max_running_requests * group.ring.stride,
                    dtype=torch.int64, device=runner.device_id,
                )
            self.groups.append((group, backend, indices))

    def prepare(self, batch, num_tokens):
        layer_batches = {}
        for group, backend, indices in self.groups:
            # Each metadata view owns physical addresses; scheduler/spec retain
            # the original logical IDs and never remap this view's writes.
            view = copy(batch)
            view.cache_groups = None
            view.attn_backend = backend
            if group.ring is not None and view.past_key_values is not None:
                resolve_window(view, group.ring, indices)
            backend.prepare(view, num_tokens)
            for layer in group.layers:
                layer_batches[layer] = view
        batch.cache_groups = layer_batches

    def forward(self, q, k, v, layer, batch, save_kv_cache=True, **kwargs):
        view = batch.cache_groups[layer.layer_id]
        return view.attn_backend.forward(
            q, k, v, layer, view, save_kv_cache, **kwargs,
        )
