"""SWA address layout and write plans over pool-owned storage."""

import numpy as np
import torch

from sfllm.kernels.cache_indices import copy_cache_slots, ring_write_locations


def index_tensor(values, device):
    tensor = torch.from_numpy(np.asarray(values, dtype=np.int64))
    return tensor.pin_memory().to(device, non_blocking=True) if tensor.device != torch.device(device) else tensor


def saved_window_slots(saved, width):
    slot, position = saved
    return (slot - 1) * width + np.arange(min(position, width), dtype=np.int64)


class WindowLayout:
    def __init__(self, history, token_shape, base, request_indices):
        self.history = history
        self.stride = token_shape[1]
        self.base = base
        self.temporary_base = base + token_shape[0] * self.stride
        self.request_indices = request_indices
        self.prefill_mapping = None
        self.previous_prefill_ids = None

    def work_slots(self, request_index, positions):
        return self.base + request_index * self.stride + positions % self.stride

    def write_locations(self, batch, locations):
        return ring_write_locations(
            locations, batch.position_ids_extend, batch.qo_indptr,
            self.request_indices, self.base, self.stride,
        )

    def prepare_reuse(self, batch, restores, materialized_context):
        """Plan checkpoint-to-work copies; each saved window is chronological.

        Full prefill needs its initial history. Materialized draft context
        needs only history surviving in a requested final/internal window.
        """
        sources, destinations = [], []
        for row, request in enumerate(batch):
            checkpoint = restores.get(request.sequence_id)
            if checkpoint is not None:
                _, position = checkpoint
                first = max(0, position - self.history)
                if materialized_context:
                    # Draft attention runs after context KV is written. Only
                    # history still needed by a final/internal window survives.
                    records = () if batch.prefill_plan is None else batch.prefill_plan.records
                    end = min((p for r, p, _ in records if r == row), default=request.prefill_end)
                    first = max(first, end - self.history)
                if first >= position:
                    continue
                src = saved_window_slots(checkpoint, self.history)[first - max(0, position - self.history):]
                dst = self.work_slots(request.request_index, np.arange(first, position))
                sources.append(src)
                destinations.append(dst)
        if not sources:
            return None
        return index_tensor(
            (np.concatenate(sources), np.concatenate(destinations)), self.request_indices.device,
        ).unbind()

    def prepare_save(self, batch):
        """Plan fused writes to working tails and selected checkpoint windows.

        New KV rows use the existing context writer; old history must be copied
        into checkpoints before the forward can overwrite the working ring.
        """
        plan = batch.prefill_plan
        offsets = np.cumsum([0, *(r.prefill_end - r.prefill_start for r in batch)])
        paths = ([range(a, b) for a, b in zip(offsets, offsets[1:])]
                 if plan is None else plan.paths)
        rows, destinations, copy_src, copy_dst = [], [], [], []
        records = () if plan is None else plan.records
        writes = [(row, request.prefill_end, None) for row, request in enumerate(batch)]
        writes.extend(records)
        split = 0
        for row, position, checkpoint in writes:
            request = batch[row]
            positions = np.arange(max(0, position - self.history), position, dtype=np.int64)
            destination = (self.work_slots(request.request_index, positions) if checkpoint is None else
                           saved_window_slots((checkpoint, position), self.history))
            path = paths[row]
            start = request.prefill_end - len(path)
            present = positions >= start
            first = max(start, 0, position - self.history)
            rows.append(np.asarray(path[first - start:position - start], dtype=np.int64))
            destinations.append(destination[present])
            if checkpoint is None:
                split += rows[-1].size
            else:
                copy_src.append(self.work_slots(request.request_index, positions[~present]))
                copy_dst.append(destination[~present])
        rows = np.concatenate(rows) if rows else np.empty(0, dtype=np.int64)
        destinations = np.concatenate(destinations) if destinations else np.empty(0, dtype=np.int64)
        copy_src = np.concatenate(copy_src) if copy_src else np.empty(0, dtype=np.int64)
        copy_dst = np.concatenate(copy_dst) if copy_dst else np.empty(0, dtype=np.int64)
        device = self.request_indices.device
        copies = index_tensor((copy_src, copy_dst), device).unbind() if copy_src.size else None
        if rows.size == offsets[-1] and np.array_equal(rows, np.arange(offsets[-1])):
            rows = None
        elif rows.size and np.array_equal(rows, np.arange(rows[0], rows[-1] + 1)):
            rows = slice(int(rows[0]), int(rows[-1]) + 1)
        else:
            rows = index_tensor(rows, device)
        return index_tensor(destinations, device), rows, split, copies


def prepare_window_writes(pool, batch, restores):
    """Build per-pool writes and ordered transfers for CacheManager to submit."""
    copies, mappings = [], []
    if not pool.materialized_context:
        locations = np.concatenate([
            np.asarray((r.out_cache_loc_spec if pool.is_draft else r.out_cache_loc), dtype=np.int64)[r.prefill_start:r.prefill_end]
            for r in batch
        ])
    writes = {}
    for group in pool.groups:
        if group.ring is None:
            continue
        ring = group.ring
        buffers = group.storage.buffers
        saved = pool.saved_windows.get(id(group))
        device = buffers[0].device
        restore = ring.prepare_reuse(batch, restores, pool.materialized_context) if restores else None
        if restore is not None:
            copies.append((saved.buffers, buffers, *restore, None, False))
        dest, rows, split, history_copy = ring.prepare_save(batch)
        secondary = (*saved.buffers, split) if saved is not None and split < dest.numel() else None
        writes[id(group)] = (dest, rows, secondary)
        if history_copy is not None:
            copies.append((buffers, saved.buffers, *history_copy, None, False))
        if not pool.materialized_context:
            # Attention needs the entire new chunk until the forward finishes.
            # Temporary locations are only used for prefill, never decode.
            if ring.prefill_mapping is None:
                ring.prefill_mapping = torch.zeros(pool.num_blocks, dtype=torch.int64, device=device)
            ids = index_tensor(locations, device)
            temporary = torch.arange(len(locations), device=device) + ring.temporary_base
            mappings.append((ring, ids, temporary))
            selected = temporary if rows is None else temporary[rows]
            copies.append((buffers, buffers, selected, dest, secondary, True))
    result = []
    for write in pool.context_buffers:
        group = pool.groups[pool.layer_groups[write.layers.start]]
        locations, rows, secondary = writes.get(id(group), (None, None, None))
        if secondary is not None:
            start = group.layers.index(write.layers.start)
            end = start + write.layers.stop - write.layers.start
            key, value, split = secondary
            secondary = key[start:end], value[start:end], split
        result.append(write._replace(locations=locations, rows=rows, secondary=secondary))
    return result, mappings, copies


def begin_window_writes(plans):
    """Install temporary mappings and preserve history before ring writes."""
    for _, mappings, copies in plans:
        for ring, ids, locations in mappings:
            if ring.previous_prefill_ids is not None:
                ring.prefill_mapping.index_fill_(0, ring.previous_prefill_ids, 0)
            ring.prefill_mapping.index_copy_(0, ids, locations)
            ring.previous_prefill_ids = ids
        for source, destination, rows, locations, secondary, after in copies:
            if not after:
                copy_cache_slots(source, destination, rows, locations, secondary)


def finish_window_writes(plans):
    """Commit temporary prefill KV once attention has finished reading it."""
    for _, _, copies in plans:
        for source, destination, rows, locations, secondary, after in copies:
            if after:
                copy_cache_slots(source, destination, rows, locations, secondary)
