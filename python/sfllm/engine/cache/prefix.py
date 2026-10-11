"""Prefix reuse and shared-prefill planning on the existing model stream."""

from bisect import bisect_left
from dataclasses import dataclass

import numpy as np
import torch

from sfllm.engine.forward_params import ForwardMode, PrefillOutputPlan

from .radix_cache import PrefixCache, common_prefix


_MIN_CHECKPOINT_DISTANCE = 128


@dataclass
class Checkpoint:
    state_slot: int


class SharedPrefill:
    @staticmethod
    def workspace_bytes(state, conv, max_requests, max_tokens, dtype):
        # One layer at a time: request and checkpoint paths, initial/final
        # states, conv taps, and expanded operands. Allocation stays unchanged.
        workspace = 2 * max_requests * (
            state.numel() * (state.element_size() + 2 * torch.float32.itemsize)
            + 2 * conv.numel() * conv.element_size()
        )
        heads, value_dim = state.shape[:2]
        return workspace + 3 * max_tokens * (
            conv.shape[0] + 2 * heads + heads * value_dim
        ) * dtype.itemsize

    def __init__(self, paths, starts, shared, records, lengths):
        self.records = records
        self.paths = paths
        self.output = PrefillOutputPlan()
        self.internal = [(row, p, slot) for row, p, slot in records if p < lengths[row]]
        self.active = any(start != end for start, end in zip(starts, shared)) or bool(self.internal) or any(not path for path in paths)
        if not self.active:
            return
        self.expanded = []
        self.indptr = [0]
        self.owners = (slice(0, sum(map(len, paths)))
                       if all(start == end for start, end in zip(starts, shared)) else [])
        for path, start, end in zip(paths, starts, shared):
            if isinstance(self.owners, list):
                self.owners.extend(range(len(self.expanded) + end - start,
                                         len(self.expanded) + len(path)))
            self.expanded.extend(path)
            self.indptr.append(len(self.expanded))
        self.rows = list(range(len(paths)))
        for row, position, _ in self.internal:
            self.expanded.extend(paths[row][:position - starts[row]])
            self.indptr.append(len(self.expanded))
            self.rows.append(row)
        tokens = sum(length - begin for length, begin in zip(lengths, shared))
        last = [path[-1] if path else tokens + row for row, path in enumerate(paths)]
        output = self.output
        output.output_indices = list(dict.fromkeys(last))
        output.output_mapping = ([output.output_indices.index(i) for i in last]
                                 if len(output.output_indices) != len(last) else None)
        output.checkpoint_indices = [paths[row][p - starts[row] - 1] for row, p, _ in records]

    def prepare(self, device):
        if not self.active:
            return
        def tensor(values, dtype=torch.int64):
            return torch.tensor(values, dtype=dtype, pin_memory=True).to(device, non_blocking=True)
        output = self.output
        output.output_indices = tensor(output.output_indices)
        if output.output_mapping is not None:
            output.output_mapping = tensor(output.output_mapping)
        output.checkpoint_indices = tensor(output.checkpoint_indices)
        self.expanded = tensor(self.expanded)
        if isinstance(self.owners, list):
            self.owners = tensor(self.owners)
        self.rows = tensor(self.rows)
        self.indptr = tensor(self.indptr)
        self.dense_rows = tensor(range(len(self.rows)), torch.int32)
        self.internal_rows = self.dense_rows[len(self.paths):]
        self.internal_slots = tensor([s for _, _, s in self.internal], torch.int32)


class PrefixReuse:
    def __init__(self, pools, state_pool=None):
        self.pools = pools
        self.state_pool = state_pool
        self.tree = PrefixCache([p.logical for p in pools if p.has_full],
                                self.release_checkpoint if state_pool is not None else None)
        self.batch_requests = []
        self.hits = self.misses = self.reused_tokens = 0
        self.prompt_tokens = self.batch_shared_tokens = 0

    def get_metrics(self):
        """Cumulative admission counts since clear; gauges include in-flight slots."""
        requests = self.hits + self.misses
        return {
            'requests': requests,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hits / requests if requests else 0.0,
            'prompt_tokens': self.prompt_tokens,
            'reused_tokens': self.reused_tokens,
            'token_hit_rate': self.reused_tokens / self.prompt_tokens if self.prompt_tokens else 0.0,
            'batch_shared_tokens': self.batch_shared_tokens,
            'entries': len(self.tree.lru),
            'checkpoints': len(self.tree.lru) if self.state_pool is not None else 0,
            'checkpoint_slots_used': self.state_pool.capacity - len(self.state_pool.slots) if self.state_pool is not None else 0,
            'checkpoint_capacity': self.state_pool.capacity if self.state_pool is not None else 0,
        }

    def tensor(self, values):
        return torch.tensor(values, dtype=torch.int64, device='cpu', pin_memory=True).to(self.state_pool.hidden.device, non_blocking=True)

    def release_checkpoint(self, checkpoint):
        self.state_pool.slots.release([checkpoint.state_slot], cache=True)

    def select(self, request):
        _, start, node = self.tree.match(request.tokens)
        if self.state_pool is None:
            # Full KV alone has no logits; recompute the last token privately.
            start = min(start, max(0, len(request.tokens) - 1))
            if not start:
                node = None
        return start, len(request.tokens), node

    def share_prefix(self, request, position, node):
        """Share Full KV and pin any checkpoint needed by stateful resources."""
        checkpoint = node.checkpoint
        full = iter(self.tree.full_prefix(node))
        for pool in self.pools:
            ids = np.asarray(next(full), dtype=np.int64)[:position] if pool.has_full else pool.logical.alloc(position)
            if pool.has_full:
                pool.logical.retain(ids, unique=True)
            locations = request.out_cache_loc_spec if pool.is_draft else request.out_cache_loc
            locations.frombytes(ids.tobytes())
        if checkpoint is not None:
            self.state_pool.slots.retain([checkpoint.state_slot])
            request.prefix_restore = (checkpoint.state_slot, position)
        self.tree.touch(node)

    def admit(self, request, start, end, node):
        request.prefix_restore = None
        if node is not None:
            self.share_prefix(request, start, node)
            self.hits += 1
            self.reused_tokens += start
        else:
            self.misses += 1
        shared, source = start, None
        for row, (other, base, _, _) in enumerate(self.batch_requests):
            # Both paths must begin from the same computed model history.
            if base == start:
                length = common_prefix(other.tokens, request.tokens)
                if self.state_pool is None:
                    length = min(length, end - 1)
                if length > shared:
                    shared, source = length, row
        if source is not None:
            other = self.batch_requests[source][0]
            for pool in self.pools:
                source_slots = other.out_cache_loc_spec if pool.is_draft else other.out_cache_loc
                destination = request.out_cache_loc_spec if pool.is_draft else request.out_cache_loc
                ids = np.asarray(source_slots[start:shared], dtype=np.int64)
                pool.logical.retain(ids, unique=True)
                destination.frombytes(ids.tobytes())
        self.batch_requests.append((request, start, shared, source))
        request.prefill_start, request.prefill_end = shared, end
        request.new_tokens = request.tokens[shared:end]
        self.prompt_tokens += end
        self.batch_shared_tokens += shared - start

    def prepare_batch(self, batch):
        entries, self.batch_requests = self.batch_requests, []
        if not entries or self.state_pool is None:
            return
        # Transfer checkpoint references once; both SWA and GDN read this batch record.
        batch.prefix_restores = {
            request.sequence_id: request.prefix_restore
            for request, _, _, _ in entries if request.prefix_restore is not None
        }
        for request, _, _, _ in entries:
            request.prefix_restore = None
        if all(request.sequence_id in batch.prefix_restores and base == len(request.tokens)
               for request, base, _, _ in entries):
            return
        paths, total, candidates = [], 0, {}
        historical = set()
        starts, shared, lengths = [], [], []
        for row, (request, base, begin, source) in enumerate(entries):
            end = len(request.tokens)
            path = range(total, total + end - begin)
            if source is not None:
                path = list(paths[source][:begin-base]) + list(path)
            paths.append(path)
            total += end - begin
            starts.append(base)
            shared.append(begin)
            lengths.append(end)
            if source is not None and begin > base:
                candidates.setdefault(tuple(request.tokens[:begin]), (source, begin))
            matched = self.tree.match(request.tokens)[0]
            if base < matched < end:
                tokens = tuple(request.tokens[:matched])
                candidates.setdefault(tokens, (row, matched))
                historical.add(tokens)
        ends = {}
        for row, (request, base, _, _) in enumerate(entries):
            if lengths[row] > base:
                ends.setdefault(tuple(request.tokens), (row, lengths[row]))
        # A prompt end already produces its state; it needs no extra GDN path.
        candidates = {
            tokens: item for tokens, item in candidates.items()
            if tokens not in ends and item[1] - starts[item[0]] >= _MIN_CHECKPOINT_DISTANCE
        }
        def benefit(item):
            tokens, (row, position) = item
            length = position - starts[row]
            token = paths[row][length - 1]
            consumers = (tokens in historical) - 1
            # Paths contain increasing projected-row IDs. Count only this
            # boundary's consumers, without scanning every prefill token.
            for path in paths:
                index = bisect_left(path, token)
                consumers += index < len(path) and path[index] == token
            return consumers * length

        # Bound snapshot density along each path, not the number of branches.
        # Independent prefix families need not share checkpoint positions.
        selected = {}
        for tokens, item in sorted(candidates.items(), key=benefit, reverse=True):
            if any(abs(len(tokens) - len(other)) < _MIN_CHECKPOINT_DISTANCE
                   and (tokens[:len(other)] == other or other[:len(tokens)] == tokens)
                   for other in selected):
                continue
            selected[tokens] = item
        selected.update(ends)
        records = []
        for tokens, (row, position) in selected.items():
            if self.tree.match(tokens)[1] == position:
                continue
            # A snapshot retained for this batch's restore cannot yield a free
            # state slot. Keep those cache records instead of emptying the LRU.
            while not len(self.state_pool.slots) and self.tree.evict(
                lambda node: self.state_pool.slots.refs[0, node.checkpoint.state_slot] == 0
            ):
                pass
            if not len(self.state_pool.slots):
                break
            records.append((row, position, int(self.state_pool.slots.alloc(1)[0])))
        batch.prefill_plan = SharedPrefill(paths, starts, shared, records, lengths)

    def before_forward(self, batch):
        if batch.forward_batch.forward_mode != ForwardMode.EXTEND or self.state_pool is None:
            return
        plan = batch.prefill_plan
        output = plan.output if plan is not None else PrefillOutputPlan()
        cached = [restore[0] if (restore := batch.prefix_restores.get(request.sequence_id)) is not None
                  and restore[1] == request.prefill_end else -1 for request in batch]
        if any(slot >= 0 for slot in cached):
            output.cached_hidden = self.state_pool.hidden[self.tensor(cached)]
        output.checkpoint_buffer = self.state_pool.prefill_hidden
        if plan is not None and plan.active:
            plan.prepare(self.state_pool.hidden.device)
        batch.forward_batch.prefill_output = output
        return [(request.request_index + 1, batch.prefix_restores.get(request.sequence_id, (-1, 0))[0])
                for request in batch]

    def after_prefill(self, batch):
        plan = batch.prefill_plan
        if self.state_pool is None:
            records = [(row, request.prefill_end, None) for row, request in enumerate(batch)]
        elif plan is None:
            return
        else:
            records = plan.records
        if self.state_pool is not None and records:
            state_slots = [slot for _, _, slot in records]
            slots = self.tensor(state_slots)
            rows = None if plan.active else self.tensor([row for row, _, _ in records])
            self.state_pool.save_hidden(slots, rows)
            self.state_pool.slots.retain(state_slots, cache=True, unique=True)
            self.state_pool.slots.release(state_slots, unique=True)
        for row, position, slot in records:
            request = batch[row]
            full = [(request.out_cache_loc_spec if pool.is_draft else request.out_cache_loc)[:position] for pool in self.pools if pool.has_full]
            tokens = tuple(request.tokens[:position])
            checkpoint = Checkpoint(slot) if slot is not None else None
            # All cache writes have been submitted on the model stream. Later
            # restores follow them there, so publish without waiting for results.
            self.tree.insert(tokens, full, checkpoint)

    def complete(self, batch):
        if batch.prefix_restores:
            self.state_pool.slots.release([slot for slot, _ in batch.prefix_restores.values()])
            batch.prefix_restores = None

    def release_request(self, request):
        # Once batched, complete() owns the reference even if the request retires.
        restore = getattr(request, 'prefix_restore', None)
        if restore is not None:
            self.state_pool.slots.release([restore[0]])
            request.prefix_restore = None

    def clear(self):
        self.tree.clear()
        self.hits = self.misses = self.reused_tokens = 0
        self.prompt_tokens = self.batch_shared_tokens = 0
