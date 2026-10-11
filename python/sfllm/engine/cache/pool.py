"""KV storage, token-slot allocation and fixed execution reservations."""

from array import array
from collections import OrderedDict
from dataclasses import dataclass
import logging
from math import prod

import torch

from .storage import KVStorage, KVWrite, Slots
from .window import WindowLayout

logger = logging.getLogger(__name__)


def group_layers(metadata):
    groups = OrderedDict()
    for layer, m in metadata.items():
        key = (m['num_kv_heads'], m['head_dim'], m['v_head_dim'], m['dtype'],
               m['window_size'], m['is_causal'])
        groups.setdefault(key, []).append(layer)
    return list(groups.items())


def _scratch_count(runner):
    args = runner.server_args
    count = 0
    if runner.is_draft:
        width = (args.speculative_num_draft_tokens if _materialized_context(runner) else
                 (args.speculative_num_steps - 1) * args.speculative_eagle_topk)
        count = args.cuda_graph_max_bs * width
    elif args.enable_prefill_cuda_graph and not args.disable_cuda_graph:
        from sfllm.engine.prefill_cuda_graph_runner import DEFAULT_PREFILL_CAPTURE_SIZES
        count += max(args.prefill_cuda_graph_sizes or DEFAULT_PREFILL_CAPTURE_SIZES)
    return count


def _logical_capacity(args, has_full, full_capacity, scratch):
    # Logical IDs remain live after their SWA storage expires.
    tokens = full_capacity if has_full else args.max_running_requests * (
        args.max_context_length + 2 * args.speculative_num_draft_tokens
    )
    return tokens + scratch + 1


def _materialized_context(runner):
    return runner.is_draft and runner.server_args.speculative_algorithm in ('dflash2', 'dspark')


def _window_shape(args, history, scratch, materialized_context):
    """Use the same geometry for memory budgeting, allocation and addressing."""
    width = history + 2 * max(args.speculative_num_draft_tokens, args.speculative_num_steps + 1)
    return ((args.max_running_requests, width),
            (scratch + 1, 0 if materialized_context else args.max_context_length))


@dataclass
class KVGroup:
    layers: list
    window: int
    slots: Slots | None
    storage: KVStorage
    ring: WindowLayout | None = None


class KVPool:
    """Per-model logical slots and independently allocated KV groups.

    Request admission, prefix policy and transfer submission live above this
    resource owner. Full groups address logical slots directly; SWA groups
    resolve them through their fixed window layout.
    """
    def __init__(self, runner, full_capacity, checkpoint_capacity):
        args = runner.server_args
        self.is_draft = runner.is_draft
        self.materialized_context = _materialized_context(runner)
        definitions = group_layers(runner.layer_metadata)
        self.has_full = any(key[4][0] < 0 for key, _ in definitions)
        scratch_count = _scratch_count(runner)
        self.num_blocks = _logical_capacity(args, self.has_full, full_capacity, scratch_count)
        self.logical = Slots(self.num_blocks, reserved=scratch_count + 1,
                             refcounted=args.enable_prefix_cache)
        self.scratch_locations = torch.arange(
            1, scratch_count + 1, dtype=torch.int64, device=runner.device_id,
        )
        self.groups = []
        self.saved_windows = {}
        self.kv_buffers = [None] * len(runner.layer_metadata)
        self.layer_groups = {}
        self.owners = set()  # Opaque allocation owners, for occupancy accounting.
        self.request_indices = torch.arange(args.max_running_requests, dtype=torch.int64, device=runner.device_id)
        for key, layers in definitions:
            heads, kd, vd, dtype, window, _ = key
            if window[0] < 0:
                token_shape, reserved = (self.num_blocks,), (0, 0)
            else:
                history = min(window[0], args.max_context_length)
                token_shape, reserved = _window_shape(args, history, scratch_count, self.materialized_context)
            storage = KVStorage(len(layers), token_shape, heads, kd, vd, dtype, runner.device_id, reserved)
            group = KVGroup(layers, window[0], self.logical if window[0] < 0 else None, storage)
            if group.window >= 0:
                group.ring = WindowLayout(history, token_shape, reserved[0], self.request_indices)
                if checkpoint_capacity:
                    self.saved_windows[id(group)] = KVStorage(
                        len(layers), (checkpoint_capacity, history), heads, kd, vd, dtype, runner.device_id,
                    )
            key_buffer, value_buffer = storage.buffers
            for local, layer in enumerate(layers):
                self.kv_buffers[layer] = key_buffer[local], value_buffer[local]
                self.layer_groups[layer] = len(self.groups)
            self.groups.append(group)
        self.dtype = self.groups[0].storage.key.dtype
        self.context_buffers = []
        for group in self.groups:
            key_buffer, value_buffer = group.storage.buffers
            start = 0
            while start < len(group.layers):
                end = start + 1
                while end < len(group.layers) and group.layers[end] == group.layers[end - 1] + 1:
                    end += 1
                self.context_buffers.append(KVWrite(
                    slice(group.layers[start], group.layers[end - 1] + 1),
                    key_buffer[start:end], value_buffer[start:end],
                ))
                start = end

    @property
    def has_window(self):
        return any(group.window >= 0 for group in self.groups)

    def can_alloc(self, count):
        return len(self.logical) >= count

    def alloc_block(self, count):
        if not self.can_alloc(count):
            raise MemoryError(f'Cannot allocate {count} token slots')
        ids = self.logical.alloc(count)
        locations = array('q')
        locations.frombytes(ids.tobytes())
        return locations

    def free_block(self, ids, force_sort=False):
        self.logical.release(ids, unique=True)

    def reset_free_order(self):
        # Flush is only allowed after requests and retained prefixes release.
        if len(self.logical) + self.logical.reserved != self.num_blocks:
            raise RuntimeError("Cache still has live token slots")
        self.logical.sort_free()

    def num_available_blocks(self):
        return len(self.logical)

    def get_usage(self):
        allocated = free = 0
        for group in self.groups:
            size = group.storage.nbytes
            allocated += size
            if group.ring is None:
                free += len(group.slots) * (size // group.slots.capacity)
            else:
                tokens = group.storage.buffers[0].shape[1]
                free += max(0, tokens - self.logical.reserved - len(self.owners) * group.ring.stride) * (size // tokens)
        return 100.0 * (1 - free / allocated)


def plan_pools(runners, state_bytes=0, state_workspace_bytes=0, checkpoint_limit=0):
    """One memory budget for every model/group; execution storage is reserved once."""
    args = runners[0].server_args
    free, _ = torch.cuda.mem_get_info(runners[0].device_id)
    budget = int(free * args.mem_fraction)
    hidden_bytes = runners[0].get_config().hidden_size * runners[0].dtype.itemsize
    full_bytes = 0
    window_bytes = checkpoint_window_bytes = 0
    temporary = (state_bytes + args.max_running_requests * hidden_bytes
                 + state_workspace_bytes) if checkpoint_limit else 0
    for runner in runners:
        definitions = group_layers(runner.layer_metadata)
        has_full = any(key[4][0] < 0 for key, _ in definitions)
        scratch = _scratch_count(runner)
        for key, layers in definitions:
            heads, kd, vd, dtype, window, _ = key
            size = len(layers) * heads * (kd + vd) * dtype.itemsize
            if window[0] < 0:
                temporary += (scratch + 1) * size
                full_bytes += size
            else:
                if not _materialized_context(runner):
                    if has_full:
                        full_bytes += 8
                    else:
                        temporary += _logical_capacity(args, False, 0, scratch) * 8
                history = min(window[0], args.max_context_length)
                shape, reserved = _window_shape(args, history, scratch, _materialized_context(runner))
                window_bytes += (prod(shape) + sum(reserved)) * size
                checkpoint_window_bytes += history * size
    for capacity in range(checkpoint_limit, -1, -1):
        available = budget - temporary - window_bytes - capacity * (state_bytes + checkpoint_window_bytes)
        full_capacity = available // full_bytes if full_bytes else 0
        if available >= 0 and (not full_bytes or full_capacity >= args.max_context_length + 2 * args.max_running_requests * max(args.speculative_num_draft_tokens, args.speculative_num_steps + 1)):
            logger.info('Cache pools: %s full token slots, %s checkpoint slots', full_capacity, capacity)
            return int(full_capacity), capacity
    raise MemoryError('KV budget cannot fit one maximum context and execution workspace')
