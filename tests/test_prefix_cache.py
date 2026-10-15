"""Shared prefill, checkpoint recovery, and cache resource ownership."""

import numpy as np
import pytest

from sfllm.engine.forward_params import ForwardMode
from sfllm.engine.cache.radix_cache import PrefixCache
from sfllm.engine.cache.storage import KVWrite, Slots
from sfllm.engine.cache.window import prepare_window_writes, saved_window_slots


def cache_manager(pools, prefix=None, state=None):
    """Exercise the coordinator with small, explicitly sized test resources."""
    from sfllm.engine.cache.manager import CacheManager
    manager = CacheManager.__new__(CacheManager)
    manager.pools = pools
    manager.prefix, manager.recurrent_state = prefix, state
    manager.window_pools = [p for p in pools if p.has_window]
    manager.request_order = None
    return manager


@pytest.mark.parametrize('backend', ['triton', 'flashinfer'])
@pytest.mark.parametrize('dtype_name', ['float32', 'bfloat16'])
@pytest.mark.parametrize('use_initial', [False, True])
def test_shared_prefill_twenty_branches_and_internal_states(backend, dtype_name, use_initial):
    import torch
    from sfllm.engine.cache.prefix import SharedPrefill
    from sfllm.kernels.gdn import GatedDeltaNetBackend

    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    torch.manual_seed(127)
    count, length, heads, hv, dim = 20, 160, 2, 4, 1024
    paths, shared, total = [list(range(length))], [0], length
    for i in range(1, count):
        prefix, end = 3 + i * 7, length + i
        paths.append(paths[0][:prefix] + list(range(total, total + end - prefix)))
        shared.append(prefix)
        total += end - prefix
    records = [(0, p, i + 1) for i, p in enumerate((3, 17, 65, 127))]
    plan = SharedPrefill(paths, [0] * count, shared, records, [len(p) for p in paths])
    plan.prepare('cuda')
    x = torch.randn(total, dim, device='cuda', dtype=torch.bfloat16) * .2
    a, b = [torch.randn(total, hv, device='cuda', dtype=torch.bfloat16) for _ in range(2)]
    weight = torch.randn(dim, 4, device='cuda', dtype=torch.bfloat16) * .2
    alog = torch.randn(hv, device='cuda') * .1
    bias = torch.randn(hv, device='cuda', dtype=torch.bfloat16) * .1
    slots = torch.arange(1, count + 1, device='cuda', dtype=torch.int32)
    dtype = getattr(torch, dtype_name)
    state = torch.zeros(count + 1, hv, 128, 128, device='cuda', dtype=dtype)
    conv = torch.zeros(count + 1, dim, 3, device='cuda', dtype=torch.bfloat16)
    if use_initial:
        # All shared paths start from the same history, in distinct work slots.
        state[1:] = torch.randn_like(state[1]) * .1
        conv[1:] = torch.randn_like(conv[1]) * .2
    initial_state, initial_conv = state[1].clone(), conv[1].clone()
    checkpoint_state = state.new_empty(len(records) + 1, *state.shape[1:])
    checkpoint_conv = conv.new_empty(len(records) + 1, *conv.shape[1:])
    kwargs = dict(conv_weight=weight, state_indices=slots,
                  a_log=alog, dt_bias=bias, num_k_heads=heads, num_v_heads=hv,
                  head_k_dim=128, head_v_dim=128)
    baseline = GatedDeltaNetBackend(backend, 'triton', use_initial_state=use_initial)
    expanded = torch.tensor([i for path in paths for i in path], device='cuda')
    cu = torch.tensor([0, *np.cumsum([len(p) for p in paths])], device='cuda')
    ref_state, ref_conv = state.clone(), conv.clone()
    reference = baseline.prefill(x[expanded], a[expanded], b[expanded],
                                 conv_states=ref_conv, ssm_states=ref_state,
                                 query_start_loc=cu, query_start_loc_i64=cu, **kwargs)
    unique_cu = torch.tensor([0, *np.cumsum([len(p)-s for p, s in zip(paths, shared)])], device='cuda')
    candidate = GatedDeltaNetBackend(backend, 'triton', use_initial_state=use_initial)
    candidate.prefill_plan = plan
    candidate.checkpoint_states = checkpoint_state, checkpoint_conv
    output = candidate.prefill(x, a, b, conv_states=conv, ssm_states=state,
                               query_start_loc=unique_cu, query_start_loc_i64=unique_cu, **kwargs)
    torch.testing.assert_close(output, reference[plan.owners], rtol=.02, atol=3e-4)
    torch.testing.assert_close(state, ref_state, rtol=.02, atol=3e-4)
    torch.testing.assert_close(conv, ref_conv, rtol=0, atol=0)
    for _, p, slot in records:
        boundary = torch.tensor([0, p], device='cuda')
        one_state, one_conv = state.new_zeros(2, *state.shape[1:]), conv.new_zeros(2, *conv.shape[1:])
        one_state[1], one_conv[1] = initial_state, initial_conv
        baseline.prefill(x[:p], a[:p], b[:p], conv_states=one_conv, ssm_states=one_state,
                         query_start_loc=boundary, query_start_loc_i64=boundary,
                         **(kwargs | {'state_indices': slots[:1]}))
        torch.testing.assert_close(checkpoint_state[slot], one_state[1], rtol=.02, atol=3e-4)
        torch.testing.assert_close(checkpoint_conv[slot], one_conv[1], rtol=0, atol=0)


@pytest.mark.parametrize('reserved', [1, 3])
def test_invalid_release_preserves_live_ownership(reserved):
    slots = Slots(8, reserved=reserved)
    ids = slots.alloc(2)
    before = slots.refs.copy()
    with pytest.raises(RuntimeError):
        slots.release([ids[0], ids[0]])
    with pytest.raises(ValueError):
        slots.release([-1])
    with pytest.raises(ValueError):
        slots.release([reserved - 1])
    with pytest.raises(ValueError):
        slots.retain([reserved - 1])
    np.testing.assert_array_equal(slots.refs, before)
    slots.release(ids)
    assert len(slots) == 8 - reserved
    slots.sort_free()
    np.testing.assert_array_equal(slots.alloc(len(slots)), np.arange(reserved, 8))


def test_branch_split_and_ancestor_eviction():
    slots = Slots(64)
    a, b = slots.alloc(6), slots.alloc(6)
    released = []
    tree = PrefixCache([slots], released.append)
    tree.insert([1, 2, 3, 4], [list(a[:4])], 'parent')
    tree.insert([1, 2, 3, 4, 5, 6], [list(a)], 'child')
    tree.insert([1, 2, 7, 8, 9, 10], [list(b)], 'branch')
    slots.release(a)
    slots.release(b)
    assert tree.match([1, 2, 3, 9])[:2] == (3, 0)
    _, pos, node = tree.match([1, 2, 3, 4, 5, 6, 7])
    assert pos == 6
    borrowed = tree.full_prefix(node)[0]
    slots.retain(borrowed)
    assert tree.evict() and released == ['parent']
    assert tree.match([1, 2, 3, 4, 5, 6])[1] == 6
    tree.clear()
    assert len(slots) == 63 - 6
    slots.release(borrowed)
    assert len(slots) == 63


def test_match_requires_complete_ancestry_and_checkpoint():
    slots = Slots(64)
    ids = slots.alloc(12)
    released = []
    tree = PrefixCache([slots], released.append)
    tree.insert([1, 2, 3, 4], [list(ids[:4])], 'a')
    tree.insert([9, 2, 3, 4], [list(ids[4:8])], 'b')
    tree.insert([1, 2, 7, 8], [list(ids[8:])], 'c')
    assert tree.match([8, 2, 3, 4])[:2] == (0, 0)
    # Splitting an edge creates no recurrent-state snapshot at the split.
    assert tree.match([1, 2])[:2] == (2, 0)
    duplicate = tree.insert([1, 2, 3, 4], [list(ids[8:])], 'duplicate')
    assert duplicate.checkpoint == 'a' and released == ['duplicate']
    slots.release(ids)
    tree.clear()
    assert len(slots) == 63


class CacheBatch(list):
    def __init__(self, requests, records=(), paths=None):
        from types import SimpleNamespace
        super().__init__(requests)
        self.sequences = self
        from sfllm.engine.forward_params import ForwardBatch
        self.forward_batch = ForwardBatch(None)
        self.prefill_plan = None
        self.prefix_restores = None
        self.window_writes = None
        if records:
            if paths is None:
                offsets = np.cumsum([0, *(r.prefill_end - r.prefill_start for r in requests)])
                paths = [list(range(a, b)) for a, b in zip(offsets, offsets[1:])]
            self.prefill_plan = SimpleNamespace(records=records, paths=paths)


def test_swa_ring_wrap_keeps_checkpoint_and_other_request():
    from array import array
    from types import SimpleNamespace as NS
    import torch
    from sfllm.engine.cache.pool import KVPool
    args = NS(max_running_requests=2, cuda_graph_max_bs=2, max_context_length=32,
              speculative_algorithm='dflash2', speculative_num_steps=2,
              speculative_num_draft_tokens=3, enable_prefill_cuda_graph=False, enable_prefix_cache=True)
    meta = dict(num_kv_heads=2, head_dim=8, v_head_dim=8, dtype=torch.bfloat16,
                is_causal=True)
    runner = NS(server_args=args, is_draft=True, device_id='cpu', get_config=lambda: None,
                layer_metadata={0: meta | {'window_size': (-1, -1)},
                                1: meta | {'window_size': (3, 0)}})
    pool = KVPool(runner, 128, 2)
    manager = cache_manager([pool])
    a, b = [NS(sequence_id=i, request_index=i, out_cache_loc_spec=array('q'),
               prefill_start=0, prefill_end=8) for i in (0, 1)]
    manager.admit(a, 0, 8, None)
    swa = pool.groups[1]
    ring = swa.ring
    buffers, _, _ = prepare_window_writes(pool, CacheBatch([a], [(0, 8, 1)]), {})
    _, key, _, destinations, rows, secondary = buffers[1]
    saved_key, _, split = secondary
    projected = torch.arange(8, dtype=key.dtype)[rows]
    key[:, destinations[:split]] = projected[None, :split, None, None]
    saved_key[:, destinations[split:]] = projected[None, split:, None, None]
    saved = saved_window_slots((1, 8), ring.history)
    expected = saved_key[:, saved].clone()
    assert swa.storage.key.shape == (1, 2, 9, 2, 8)
    saved_storage = pool.saved_windows[id(swa)]
    assert saved_storage.key.shape == (1, 2, 3, 2, 8)
    assert key.untyped_storage().data_ptr() != saved_key.untyped_storage().data_ptr()
    no_saved = KVPool(runner, 128, 0).groups[1]
    assert no_saved.storage.key.shape == swa.storage.key.shape
    assert no_saved.storage.buffers[0].shape == key.shape
    assert no_saved.ring.temporary_base == ring.temporary_base
    pool.logical.retain(a.out_cache_loc_spec, unique=True)
    b.out_cache_loc_spec.extend(a.out_cache_loc_spec)
    manager.admit(b, 8, 8, None)
    sources, dest = ring.prepare_reuse(CacheBatch([b]), {b.sequence_id: (1, 8)}, pool.materialized_context)
    key[:, dest] = saved_key[:, sources]
    # Several wraps, with differently sized accepted prefixes, stay in A's ring.
    position = 8
    for accepted in (3, 1, 2, 3, 1, 3):
        proposed = np.arange(position, position + 3)
        key[:, ring.work_slots(a.request_index, proposed)] = -1
        committed = np.arange(position, position + accepted)
        key[:, ring.work_slots(a.request_index, committed)] = torch.tensor(committed, dtype=key.dtype)[None, :, None, None]
        manager.allocate_batch([a], [[accepted]])
        position += accepted
        tail = np.arange(position - ring.history, position)
        torch.testing.assert_close(key[0, ring.work_slots(a.request_index, tail), 0, 0], torch.tensor(tail, dtype=key.dtype))
        torch.testing.assert_close(saved_key[:, saved], expected)
        torch.testing.assert_close(key[:, dest], expected)
    assert swa.slots is None
    assert saved_window_slots((2, 32), ring.history)[-1] < saved_key.shape[1]
    for request in (a, b):
        manager.release_request(request)
    assert len(pool.logical) == pool.num_blocks - pool.logical.reserved


def test_swa_prefill_writes_shared_tails_and_checkpoint_windows_only():
    from array import array
    from types import SimpleNamespace as NS
    import torch
    from sfllm.engine.cache.pool import KVPool

    args = NS(max_running_requests=3, cuda_graph_max_bs=3, max_context_length=2048,
              speculative_algorithm='dspark', speculative_num_steps=2,
              speculative_num_draft_tokens=3, enable_prefill_cuda_graph=False, enable_prefix_cache=True)
    meta = dict(num_kv_heads=1, head_dim=8, v_head_dim=8, dtype=torch.bfloat16,
                is_causal=True)
    runner = NS(server_args=args, is_draft=True, device_id='cpu', get_config=lambda: None,
                layer_metadata={0: meta | {'window_size': (7, 0)},
                                1: meta | {'window_size': (-1, -1)}})
    pool = KVPool(runner, 2048, 2)
    manager = cache_manager([pool])
    a, b, c = [NS(sequence_id=i, request_index=i, out_cache_loc_spec=array('q'), prefill_start=start,
                  prefill_end=end) for i, start, end in ((0, 0, 100), (1, 50, 100), (2, 30, 30))]
    manager.admit(a, 0, 100, None)
    for request, prefix in ((b, 50), (c, 30)):
        ids = a.out_cache_loc_spec[:prefix]
        pool.logical.retain(ids, unique=True)
        request.out_cache_loc_spec.extend(ids)
        manager.admit(request, prefix, request.prefill_end, None)
    swa = pool.groups[0]
    assert swa.storage.buffers[0].shape[1] < 100  # Full projected prefill cannot fit in this pool.
    buffers, _, _ = prepare_window_writes(pool, CacheBatch([a, b, c], [(0, 25, 1), (1, 75, 2)],
        paths=[list(range(100)), list(range(50)) + list(range(100, 150)), list(range(30))]), {})
    aa, bb = np.asarray(a.out_cache_loc_spec), np.asarray(b.out_cache_loc_spec)
    expected = np.concatenate([aa[93:], bb[93:], aa[23:30], aa[18:25], bb[68:75]])
    writer_ids = np.concatenate([aa, bb[50:]])
    _, _, _, destinations, rows, secondary = buffers[0]
    np.testing.assert_array_equal(writer_ids[rows], expected)
    split = secondary[2]
    assert len(set(destinations[:split].tolist())) == split
    assert len(set(destinations[split:].tolist())) == len(expected) - split
    assert buffers[1].rows is None  # Full still writes every input token.
    for request in (a, b, c):
        manager.release_request(request)
    assert len(pool.logical) == pool.num_blocks - pool.logical.reserved


@pytest.mark.parametrize('decode', [False, True])
def test_window_indices_graph_replay(decode):
    import torch
    from types import SimpleNamespace
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    from sfllm.kernels.cache_indices import resolve_window, ring_write_locations
    from sfllm.engine.cache.manager import CacheManager

    lengths = [0, 3, 9, 6]
    query_lengths = [1, 1, 1, 1] if decode else [0, 1, 2, 5]
    indptr = torch.tensor([0, *np.cumsum(lengths)], device='cuda')
    qo = torch.tensor([0, *np.cumsum(query_lengths)], device='cuda')
    ids = torch.arange(1, sum(lengths) + 1, device='cuda')
    locations = torch.arange(24, 24 + sum(query_lengths), device='cuda')
    ring = SimpleNamespace(base=1, stride=10, history=4,
                           request_indices=torch.tensor([0, 1, 2, 3], device='cuda'),
                           prefill_mapping=None)
    cache = CacheManager.__new__(CacheManager)
    cache.window_pools, cache.prefix, cache.request_order = [ring], None, None
    resolved = torch.full_like(ids, -1)
    positions = torch.zeros_like(locations)

    def prepare():
        batch = SimpleNamespace(kv_indices=ids, kv_indptr=indptr,
                                out_cache_loc=locations, qo_indptr=qo,
                                forward_mode=SimpleNamespace(name='EXTEND'),
                                is_decode=lambda: decode)
        resolve_window(batch, ring, resolved)
        batch.writes = ring_write_locations(locations, positions, qo, ring.request_indices, ring.base, ring.stride)
        return batch

    prepare()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        batch = prepare()
    # The same graph consumes new request slots and lengths.
    requests = [3, 0, 1, 2]
    lengths = [2, 5, 5, 6]
    request_batch = CacheBatch([SimpleNamespace(request_index=i) for i in requests])
    request_batch.forward_batch = SimpleNamespace(forward_mode=ForwardMode.DECODE)
    cache.before_forward(request_batch)
    version = ring.request_indices._version
    cache.before_forward(request_batch)
    assert ring.request_indices._version == version  # Stable order does not upload again.
    positions.copy_(torch.arange(locations.numel(), device='cuda') + 17)
    indptr.copy_(torch.tensor([0, *np.cumsum(lengths)], device='cuda'))
    graph.replay()
    expected, writes = [], []
    for request, length, query in zip(requests, lengths, query_lengths):
        start = max(0, length - 4 - int(decode))
        expected.append(1 + request * 10 + torch.arange(start, length, device='cuda') % 10)
        start = length - int(decode)
        writes.append(1 + request * 10 + torch.arange(start, start + query, device='cuda') % 10)
    counts = [x.numel() for x in expected]
    torch.testing.assert_close(batch.kv_indptr,
                               torch.tensor([0, *np.cumsum(counts)], device='cuda'))
    torch.testing.assert_close(batch.kv_indices[:sum(counts)], torch.cat(expected))
    torch.testing.assert_close(batch.out_cache_loc, torch.cat(writes))
    owners = torch.repeat_interleave(ring.request_indices, torch.tensor(query_lengths, device='cuda'))
    torch.testing.assert_close(batch.writes, 1 + owners * 10 + positions % 10)


@pytest.mark.parametrize('backend', ['triton', 'flashinfer'])
@pytest.mark.parametrize('dtype_name', ['float32', 'bfloat16'])
def test_gdn_continuation_short_suffix(backend, dtype_name):
    import torch
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    from sfllm.kernels.gdn import GatedDeltaNetBackend

    torch.manual_seed(21)
    h, hv, k, v, dim, n = 2, 4, 128, 128, 1024, 96
    x = torch.randn(n, dim, device='cuda', dtype=torch.bfloat16) * .2
    a, b = [torch.randn(n, hv, device='cuda', dtype=torch.bfloat16) for _ in range(2)]
    weight = torch.randn(dim, 4, device='cuda', dtype=torch.bfloat16) * .2
    alog = torch.randn(hv, device='cuda') * .1
    bias = torch.randn(hv, device='cuda', dtype=torch.bfloat16) * .1
    initial = torch.randn(3, hv, v, k, device='cuda', dtype=getattr(torch, dtype_name)) * .1
    taps = torch.randn(3, dim, 3, device='cuda', dtype=torch.bfloat16) * .2
    slots = torch.tensor([2], device='cuda', dtype=torch.int32)
    op = GatedDeltaNetBackend(backend, 'triton', use_initial_state=True)

    def run(lengths):
        states, conv = initial.clone(), taps.clone()
        out, start = [], 0
        for length in lengths:
            cu = torch.tensor([0, length], device='cuda', dtype=torch.int64)
            out.append(op.prefill(
                x[start:start+length], a[start:start+length], b[start:start+length],
                conv_weight=weight, conv_states=conv, ssm_states=states,
                state_indices=slots, query_start_loc=cu, query_start_loc_i64=cu,
                a_log=alog, dt_bias=bias, num_k_heads=h, num_v_heads=hv,
                head_k_dim=k, head_v_dim=v,
            ))
            start += length
        return torch.cat(out), states, conv

    whole, state, conv = run([96])
    split, split_state, split_conv = run([64, 1, 2, 29])
    torch.testing.assert_close(split_conv, conv, rtol=0, atol=0)
    torch.testing.assert_close(split, whole, rtol=.02, atol=3e-4)
    torch.testing.assert_close(split_state, state, rtol=.03, atol=2e-4)


@pytest.mark.parametrize('selection', ['all', 'tail', 'windows'])
@pytest.mark.parametrize('separate', [False, True])
def test_context_writer_independent_pools_and_graph(selection, separate):
    import torch
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    import sf_kernel
    from sfllm.kernels.kv_cache import write_context_kv

    torch.manual_seed(127)
    n, layers, heads, dim, capacity = 17, 6, 2, 128, 37
    raw = torch.randn(n, layers, 2 * heads, dim, device='cuda', dtype=torch.bfloat16)
    original = raw.clone()
    norms = torch.randn(layers, dim, device='cuda', dtype=raw.dtype)
    rope = torch.randn(64, dim, device='cuda')
    positions = torch.arange(4, 4 + n, device='cuda')
    locations = torch.randperm(capacity, device='cuda')[:n]
    reference = tuple(torch.zeros(layers, capacity, heads, dim, device='cuda', dtype=raw.dtype)
                      for _ in range(2))
    selected = (None if selection == 'all' else slice(11, n) if selection == 'tail'
                else torch.tensor([1, 2, 3, 8, 9, 15, 16], device='cuda'))
    mapped = locations if selected is None else locations[selected]
    mapping = mapped + 3
    destinations = []
    for start, end, pitch, table in ((0, 5, capacity + 9, mapping), (5, 6, capacity, None)):
        k = torch.full((end - start, pitch, heads, dim), 7., device='cuda', dtype=raw.dtype)
        secondary = None
        if separate and start == 0:
            secondary = (*(torch.full((end-start, pitch+7, heads, dim), 7., device='cuda', dtype=raw.dtype)
                           for _ in range(2)), mapped.numel() // 2)
        destinations.append(KVWrite(slice(start, end), k, torch.full_like(k, 7.), table,
                             selected if table is not None else None, secondary))

    write_context_kv(raw, norms, rope, positions, False, destinations, locations, 1e-6)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        write_context_kv(raw, norms, rope, positions, False, destinations, locations, 1e-6)
    # The captured writer must consume the new map, not the capture-time indices.
    mapping.add_(2)
    raw.copy_(original)
    for _, k, v, _, _, secondary in destinations:
        k.fill_(7.)
        v.fill_(7.)
        if secondary is not None:
            secondary[0].fill_(7.)
            secondary[1].fill_(7.)
    graph.replay()
    torch.ops.sfkernels.kv_norm_rope_and_cache(
        original, norms, rope, positions, False, *reference, locations, 1e-6,
    )
    for group, key, value, table, rows, secondary in destinations:
        source = locations if rows is None else locations[rows]
        locs = source if table is None else table
        transfers = [(key, value, locs, source)]
        if secondary is not None:
            other_key, other_value, split = secondary
            transfers = [(key, value, locs[:split], source[:split]),
                         (other_key, other_value, locs[split:], source[split:])]
        for k, v, dst, src in transfers:
            for actual, expected in ((k, reference[0]), (v, reference[1])):
                torch.testing.assert_close(actual[:, dst], expected[group][:, src], rtol=0, atol=0)
                untouched = torch.ones(actual.shape[1], device='cuda', dtype=torch.bool)
                untouched[dst] = False
                assert bool((actual[:, untouched] == 7).all())



@pytest.mark.parametrize('families', [1, 2])
def test_cold_twenty_requests_share_rows_without_splitting(families):
    from array import array
    from types import SimpleNamespace as NS
    import torch
    from sfllm.engine.cache.pool import KVPool
    from sfllm.engine.cache.prefix import PrefixReuse

    args = NS(max_running_requests=20, cuda_graph_max_bs=20, max_context_length=16384,
              speculative_num_steps=1, speculative_num_draft_tokens=2,
              enable_prefill_cuda_graph=False, enable_prefix_cache=True)
    meta = dict(num_kv_heads=1, head_dim=8, v_head_dim=8, dtype=torch.bfloat16,
                is_causal=True)
    runner = NS(server_args=args, is_draft=False, device_id='cpu', get_config=lambda: None,
                layer_metadata={0: meta | {'window_size': (-1, -1)},
                                1: meta | {'window_size': (7, 0)}})
    pool = KVPool(runner, 16384, 20)
    cache = PrefixReuse([pool], NS(capacity=20, slots=Slots(21)))
    manager = cache_manager([pool], cache)
    requests = []
    family_size = 20 // families
    for family in range(families):
        trunk = list(range(family * 10000, family * 10000 + 768))
        for i in range(family_size):
            boundary = i * 32
            tokens = trunk if i == 0 else trunk[:boundary] + [1000+i] * (768-boundary)
            requests.append(NS(sequence_id=len(requests), tokens=tokens, out_cache_loc=array('q')))
    duplicate = 20 - family_size + 5
    requests[-1].tokens = requests[duplicate].tokens.copy()
    requests[6].tokens = requests[0].tokens[:192]
    for request in requests:
        start, end, node = cache.select(request)
        assert (start, end, node) == (0, len(request.tokens), None)
        manager.admit(request, start, end, node)
        assert request.prefill_end == len(request.tokens)
    batch = NS(prefill_plan=None)
    cache.prepare_batch(batch)
    plan = batch.prefill_plan
    stats = cache.get_metrics()
    assert stats['requests'] == stats['misses'] == 20
    assert stats['hits'] == stats['reused_tokens'] == 0
    assert stats['batch_shared_tokens'] == stats['prompt_tokens'] - len(plan.owners)
    assert len(plan.paths) == 20
    assert len(plan.owners) == len(set(i for path in plan.paths for i in path))
    assert len(plan.owners) < sum(len(r.tokens) for r in requests)
    assert plan.paths[-1] == plan.paths[duplicate]
    for i, request in enumerate(requests[:-1]):
        source = i // family_size * family_size
        boundary = (i % family_size) * 32
        assert request.out_cache_loc[:boundary] == requests[source].out_cache_loc[:boundary]
        assert list(plan.paths[i][:boundary]) == list(plan.paths[source][:boundary])
    # Several spaced boundaries are allowed; dense forks must not each replay
    # their prefix. A short request already supplies its endpoint's state.
    assert families < len(plan.internal) < 10
    assert {requests[row].tokens[0] for row, _, _ in plan.internal} == {i * 10000 for i in range(families)}
    for i, (row, position, _) in enumerate(plan.internal):
        assert position >= 128
        for other_row, other_position, _ in plan.internal[:i]:
            if requests[row].tokens[0] == requests[other_row].tokens[0]:
                assert abs(position - other_position) >= 128
    assert not any(requests[row].tokens[:p] == requests[6].tokens for row, p, _ in plan.internal)
    assert any(row == 6 and p == 192 for row, p, _ in plan.records)
    assert len(plan.records) <= cache.state_pool.capacity
    assert all(len(path) == len(request.tokens) for path, request in zip(plan.paths, requests))
    for request in requests:
        manager.release_request(request)
    assert len(pool.logical) == pool.num_blocks - 1
    for group in pool.groups:
        if group.slots is not None:
            assert len(group.slots) == group.slots.capacity - 1
    cache.state_pool.slots.release([slot for _, _, slot in plan.records])
    assert len(cache.state_pool.slots) == 20


@pytest.mark.parametrize('snapshot_slots', [False, True])
@pytest.mark.parametrize('exact_hit', [False, True])
def test_published_checkpoint_restore_survives_eviction(snapshot_slots, exact_hit, monkeypatch):
    from array import array
    from types import SimpleNamespace as NS
    import torch
    from sfllm.engine.cache.pool import KVPool
    from sfllm.engine.cache.manager import CacheManager
    from sfllm.engine.cache.prefix import PrefixReuse
    from sfllm.engine.cache.state import StatePool
    from sfllm.kernels.gdn import GatedDeltaNetBackend
    from sfllm.engine.cache.state import GatedDeltaNetState

    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    args = NS(max_running_requests=2, cuda_graph_max_bs=2, max_context_length=16,
              speculative_num_steps=1, speculative_num_draft_tokens=2,
              enable_prefill_cuda_graph=False, enable_prefix_cache=True)
    runner = NS(server_args=args, is_draft=False, device_id='cuda', get_config=lambda: None,
                layer_metadata={0: dict(num_kv_heads=1, head_dim=8, v_head_dim=8,
                                        dtype=torch.bfloat16, is_causal=True,
                                        window_size=(-1, -1))})
    pool = KVPool(runner, 64, 2)
    monkeypatch.setenv('SFLLM_GDN_JOURNAL', '0')
    backends = [GatedDeltaNetBackend('triton', 'triton') for _ in range(2)]
    state = GatedDeltaNetState(
        backends, (1, 8, 8), (8, 3), num_k_heads=1, max_requests=2,
        spec_steps=2 if snapshot_slots else 0,
        ssm_dtype=torch.float32, dtype=torch.float32, device='cuda',
    )
    ssm, conv = state.ssm.normal_(), state.conv.normal_()
    if snapshot_slots:
        state.current_slots.copy_(torch.tensor([0, 3, 6], dtype=torch.int32, device='cuda'))
    state_pool = StatePool(2, 2, 8, torch.float32, 'cuda', (ssm, conv))
    state.bind_checkpoints(state_pool.ssm, state_pool.conv)
    cache = PrefixReuse([pool], state_pool)
    manager = cache_manager([pool], cache, state)
    cache.state_pool.prefill_hidden.normal_()

    producer = NS(sequence_id=1, request_index=0, tokens=[1, 2, 3], out_cache_loc=array('q'))
    consumer = NS(sequence_id=2, request_index=1, tokens=[1, 2, 3] if exact_hit else [1, 2, 3, 4],
                  out_cache_loc=array('q'))
    manager.admit(producer, 0, 3, None)
    first = CacheBatch([producer])
    cache.prepare_batch(first)
    manager.before_forward(first)
    assert not any(b.use_initial_state for b in backends)
    source = 3 if snapshot_slots else 1
    expected_ssm, expected_conv = ssm[:, source].clone(), conv[:, 1].clone()
    manager.after_prefill(first)
    start, end, node = cache.select(consumer)
    assert (start, end) == (3, len(consumer.tokens))
    assert cache.tree.match(consumer.tokens)[1] == 3
    checkpoint_slot = node.checkpoint.state_slot
    manager.admit(consumer, start, end, node)
    second = CacheBatch([consumer])
    cache.prepare_batch(second)
    ssm[:, source].zero_()
    conv[:, 1].zero_()
    manager.before_forward(second)
    if exact_hit:
        assert second.prefill_plan is None
        # No model forward runs; completion must restore private working state.
        manager.after_prefill(second)
    assert all(b.use_initial_state for b in backends)
    torch.cuda.synchronize()
    cache.complete(first)
    assert cache.tree.evict()
    assert cache.state_pool.slots.refs[0, checkpoint_slot] == 1
    destination = 4 if snapshot_slots else 2
    # The pending prefill may read an immutable checkpoint view directly.
    # Eviction must preserve that input until the consuming batch completes.
    for layer, backend in enumerate(backends):
        initial = backend.prefill_initial_states
        restored_ssm, restored_conv = (
            (ssm[layer, destination], conv[layer, 2]) if initial is None
            else (initial[0][0], initial[1][0])
        )
        torch.testing.assert_close(restored_ssm, expected_ssm[layer])
        torch.testing.assert_close(restored_conv, expected_conv[layer])
    cache.complete(second)
    if second.prefill_plan is not None:
        cache.state_pool.slots.release([slot for _, _, slot in second.prefill_plan.records])
    for request in (producer, consumer):
        manager.release_request(request)
    assert len(cache.state_pool.slots) == 2
    assert len(pool.logical) == pool.num_blocks - 1


def test_full_only_matches_compressed_edges_and_keeps_live_references():
    slots = Slots(64)
    a, b = slots.alloc(6), slots.alloc(6)
    tree = PrefixCache([slots])
    tree.insert([1, 2, 3, 4], [a[:4]])
    tree.insert([1, 2, 3, 4, 5, 6], [a])
    tree.insert([1, 2, 7, 8, 9, 10], [b])
    slots.release(a)
    slots.release(b)
    assert len(tree.lru) == 3
    assert all(node.checkpoint is None for node in tree.lru)
    for tokens, length in (([1, 2], 2), ([1, 2, 3, 9], 3),
                           ([1, 2, 3, 4, 5, 6, 7], 6)):
        matched, resume, node = tree.match(tokens)
        assert matched == resume == length
        assert node in tree.lru
        np.testing.assert_array_equal(tree.full_prefix(node)[0][:length], a[:length])
    _, length, consumer = tree.match([1, 2, 3, 9])
    borrowed = tree.full_prefix(consumer)[0][:length]
    slots.retain(borrowed)
    tree.protected.add(consumer)
    assert tree.evict()
    assert consumer in tree.lru
    tree.protected.clear()
    tree.clear()
    assert len(slots) == 63 - length
    slots.release(borrowed)
    assert len(slots) == 63
    assert tree.match([1, 2, 3]) == (0, 0, None)


@pytest.mark.parametrize('batched', [False, True])
def test_checkpoint_reference_release_before_and_after_batching(batched):
    from types import SimpleNamespace as NS
    from sfllm.engine.cache.prefix import Checkpoint, PrefixReuse

    state_pool = NS(capacity=1, slots=Slots(2))
    prefix = PrefixReuse([], state_pool)
    slot = int(state_pool.slots.alloc(1)[0])
    state_pool.slots.retain([slot], cache=True)
    state_pool.slots.release([slot])
    prefix.tree.insert([1, 2, 3], [], Checkpoint(slot))
    requests = [NS(sequence_id=i, tokens=[1, 2, 3]) for i in range(2)]
    for request in requests:
        prefix.admit(request, *prefix.select(request))
    batch = CacheBatch(requests)
    if batched:
        prefix.prepare_batch(batch)
    # Eviction cannot recycle a checkpoint still borrowed by either consumer.
    prefix.tree.clear()
    assert len(state_pool.slots) == 0
    for request in requests:
        prefix.release_request(request)
    if batched:
        assert len(state_pool.slots) == 0  # The batch now owns both references.
        prefix.complete(batch)
        prefix.complete(batch)
    assert len(state_pool.slots) == 1


def test_full_only_cold_batch_and_complete_hit_need_no_checkpoints():
    from array import array
    from types import SimpleNamespace as NS
    from sfllm.engine.cache.manager import CacheManager
    from sfllm.engine.cache.prefix import PrefixReuse

    slots = Slots(64)
    pool = NS(logical=slots, has_full=True, has_window=False, is_draft=False,
              owners=set(), alloc_block=lambda count: array('q', slots.alloc(count)),
              free_block=lambda ids: slots.release(ids, unique=True))
    prefix = PrefixReuse([pool])
    manager = CacheManager.__new__(CacheManager)
    manager.pools, manager.prefix, manager.window_pools = [pool], prefix, []
    manager.recurrent_state = None

    def request(index, tokens):
        return NS(sequence_id=index, tokens=tokens, out_cache_loc=array('q'))

    batch = CacheBatch([request(0, [1, 2, 3, 4, 5]),
                        request(1, [1, 2, 3, 7, 8]), request(2, [1, 2, 3]),
                        request(3, [1, 2, 3, 4, 5])])
    for r in batch:
        manager.admit(r, *manager.select(r))
    manager.prepare_batch(batch)
    assert [r.prefill_start for r in batch] == [0, 3, 2, 4]
    assert all(r.new_tokens for r in batch)
    assert batch.prefill_plan is None
    assert batch.prefix_restores is None
    assert batch[0].out_cache_loc[:3] == batch[1].out_cache_loc[:3]
    prefix.after_prefill(batch)
    assert len(prefix.tree.lru) == 3
    assert prefix.get_metrics()['checkpoint_capacity'] == 0
    assert prefix.get_metrics()['checkpoints'] == 0
    for r in batch:
        manager.release_request(r)
    hit = request(4, [1, 2, 3, 4, 5])
    start, end, node = manager.select(hit)
    assert (start, end) == (4, 5)
    manager.admit(hit, start, end, node)
    assert hit.new_tokens == [5]
    prefix.tree.clear()
    assert len(slots) == 63 - 5
    manager.release_request(hit)
    assert len(slots) == 63
