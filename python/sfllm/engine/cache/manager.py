"""Cache resources and lifetime shared by ordinary and prefix-reusing requests."""

import torch

from sfllm.engine.forward_params import ForwardMode

from .attention import GroupedAttentionBackend
from .pool import KVPool, plan_pools
from .prefix import PrefixReuse, SharedPrefill
from .state import StatePool
from .storage import KVWrite
from .window import begin_window_writes, finish_window_writes, prepare_window_writes


class CacheManager:
    def __init__(self, runners):
        runner = runners[0]
        args = runner.server_args
        self.recurrent_state = runner.recurrent_state
        states = (() if self.recurrent_state is None
                  else (self.recurrent_state.ssm, self.recurrent_state.conv))
        state_bytes = workspace_bytes = 0
        checkpoint_limit = args.max_running_requests if args.enable_prefix_cache and (
            states or any(m['window_size'][0] >= 0 for r in runners for m in r.layer_metadata.values())
        ) else 0
        if checkpoint_limit:
            state_bytes = StatePool.entry_bytes(runner.get_config().hidden_size, runner.dtype, states)
            if states:
                workspace_bytes = SharedPrefill.workspace_bytes(
                    states[0][0, 0], states[1][0, 0],
                    args.max_running_requests, args.max_context_length, runner.dtype,
                )
        full_capacity, checkpoints = plan_pools(
            runners, state_bytes, workspace_bytes, checkpoint_limit,
        )
        self.pools = [KVPool(r, full_capacity, checkpoints) for r in runners]
        for r, pool in zip(runners, self.pools):
            r.block_memory_manager = pool
            if pool.has_window:
                r.attn_backend = GroupedAttentionBackend(r, pool)
        self.state_pool = self.prefix = None
        if checkpoint_limit:
            self.state_pool = StatePool(
                checkpoints, args.max_running_requests, runner.get_config().hidden_size,
                runner.dtype, runner.device_id, states,
            )
            runner.prefill_hidden = self.state_pool.prefill_hidden
            if self.recurrent_state is not None:
                self.recurrent_state.bind_checkpoints(self.state_pool.ssm, self.state_pool.conv)
        if args.enable_prefix_cache:
            self.prefix = PrefixReuse(self.pools, self.state_pool)
        self.window_pools = [p for p in self.pools if p.has_window]
        self.request_order = None

    def select(self, request):
        if self.prefix is not None:
            return self.prefix.select(request)
        return 0, len(request.tokens), None

    def _ensure_capacity(self, pool, count):
        """Reclaim retained prefixes without putting eviction policy in a pool."""
        while not pool.can_alloc(count):
            if self.prefix is None or not self.prefix.tree.evict():
                return False
        return True

    def can_admit(self, request, start, node, required_tokens):
        tree = self.prefix.tree if node is not None else None
        if tree is not None:
            tree.protected.add(node)
        try:
            for pool in self.pools:
                reused = start if pool.has_full or request.request_index >= 0 else 0
                required = len(request.tokens) if pool.is_draft else required_tokens
                if (required > pool.num_blocks - pool.logical.reserved
                        or not self._ensure_capacity(pool, required - reused)):
                    return False
            return True
        finally:
            if tree is not None:
                tree.protected.discard(node)

    def admit(self, request, start, end, node):
        if self.prefix is not None:
            self.prefix.admit(request, start, end, node)
            start = request.prefill_start
        else:
            request.prefill_start, request.prefill_end = start, end
        for pool in self.pools:
            pool.owners.add(request.sequence_id)
            locations = request.out_cache_loc_spec if pool.is_draft else request.out_cache_loc
            locations.extend(pool.alloc_block(end - start))

    def prepare_batch(self, batch):
        if self.prefix is not None:
            self.prefix.prepare_batch(batch)
            batch.forward_batch.forward_mode = ForwardMode.EXTEND if batch.sequences else ForwardMode.DECODE

    def allocate_batch(self, requests, token_counts):
        """Reserve each model's tokens once, preserving request/slot order."""
        if not requests:
            return
        for pool, counts in zip(self.pools, token_counts):
            count = sum(counts)
            if self.prefix is not None and not self._ensure_capacity(pool, count):
                raise MemoryError(f'Cannot allocate {count} token slots')
            locations = pool.alloc_block(count)
            start = 0
            for request, count in zip(requests, counts):
                destination = request.out_cache_loc_spec if pool.is_draft else request.out_cache_loc
                destination.extend(locations[start:start + count])
                start += count

    def before_forward(self, batch, prefill_graph=None):
        if batch.forward_batch.forward_mode == ForwardMode.EXTEND and self.window_pools:
            batch.window_writes = {
                pool: prepare_window_writes(pool, batch, batch.prefix_restores)
                for pool in self.window_pools
            }
        if self.window_pools:
            order = tuple(r.request_index for r in batch)
            if order != self.request_order:
                indices = torch.tensor(order, dtype=torch.int64, pin_memory=True)
                for pool in self.window_pools:
                    pool.request_indices[:len(batch)].copy_(indices, non_blocking=True)
                self.request_order = order
        if batch.forward_batch.forward_mode == ForwardMode.EXTEND and batch.window_writes is not None:
            begin_window_writes(batch.window_writes.values())
        if self.prefix is not None:
            restores = self.prefix.before_forward(batch)
            if self.recurrent_state is not None and restores is not None:
                self.recurrent_state.reuse_checkpoints(
                    batch.prefill_plan, restores,
                    use_prefill_graph=prefill_graph is not None and prefill_graph.can_run(batch),
                )

    def after_prefill(self, batch):
        if batch.window_writes is not None:
            finish_window_writes(batch.window_writes.values())
        if self.prefix is not None:
            if self.recurrent_state is not None:
                if batch.prefix_restores:
                    # Exact hits skip forward, so restore their working state here.
                    restores = [(r.request_index + 1, restore[0]) for r in batch
                                if (restore := batch.prefix_restores.get(r.sequence_id)) is not None
                                and restore[1] == r.prefill_end]
                    if restores:
                        self.recurrent_state.reuse_checkpoints(None, restores)
                self.recurrent_state.save_checkpoints(batch)
            self.prefix.after_prefill(batch)

    def context_for(self, pool, batch, locations=None):
        """Use prepared prefill writes, or map newly accepted context tokens."""
        if not pool.has_window:
            return pool.context_buffers
        if locations is None:
            return batch.window_writes[pool][0]
        forward_batch = batch.forward_batch_spec if pool.is_draft else batch.forward_batch
        mapped = {id(g): g.ring.write_locations(forward_batch, locations)
                  for g in pool.groups if g.ring is not None}
        return [KVWrite(write.layers, write.key, write.value, mapped.get(id(pool.groups[pool.layer_groups[write.layers.start]])))
                for write in pool.context_buffers]

    def complete_batch(self, batch):
        if self.prefix is not None:
            self.prefix.complete(batch)

    def release_request(self, request):
        # Pools keep fixed storage. New writes follow the old request's queued
        # accesses on the compute stream; returning IDs needs no event query.
        for pool in self.pools:
            pool.owners.discard(request.sequence_id)
            locations = request.out_cache_loc_spec if pool.is_draft else request.out_cache_loc
            pool.free_block(locations)
        if self.prefix is not None:
            self.prefix.release_request(request)

    def clear(self):
        if any(pool.owners for pool in self.pools):
            raise RuntimeError('Cannot clear a cache with live requests')
        if self.prefix is not None:
            self.prefix.clear()
        for pool in self.pools:
            pool.reset_free_order()
