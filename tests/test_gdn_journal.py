"""GPU regression tests for speculative GDN journals and model dispatch."""

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def assert_bits_equal(actual, expected):
    bits = torch.int32 if actual.dtype == torch.float32 else torch.int16
    assert torch.equal(actual.view(bits), expected.view(bits))


def assert_output_close(actual, expected):
    # The explicit thread layout changes FP32 reductions before BF16 rounding.
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-6)


def assert_state_close(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-7)


@pytest.mark.parametrize("steps", [1, 2, 3, 4, 5, 6, 7, 8, 16, 31])
@pytest.mark.parametrize("geometry", [(2, 4, 128, 128), (3, 6, 48, 80)])
def test_journal_prefixes_and_successive_rounds(steps, geometry):
    dtype = torch.float32
    from sfllm.kernels.gdn import packed_gdn_decode, update_recurrent_state_indices
    from sfllm.kernels.gdn_journal import packed_gdn_journal_verify, replay_gdn_journal

    h, hv, k, v = geometry
    layers, batch, capacity = 2, 5, 8
    torch.manual_seed(841)
    device = "cuda"
    # Both indices and gates can be strided views. Zero and -1 are padding.
    indices = torch.tensor([3, 99, -1, 99, 1, 99, 0, 99, 5, 99],
                           device=device, dtype=torch.int32)[::2]
    base = torch.arange(capacity, device=device, dtype=torch.int32) * (steps + 1) + 1
    current = torch.cat((base.new_zeros(1), base))
    read = torch.empty(batch, device=device, dtype=torch.int32)
    write = torch.empty(batch, steps, device=device, dtype=torch.int32)
    baseline = torch.zeros(layers, capacity * (steps + 1) + 1, hv, v, k,
                           device=device, dtype=dtype)
    # An envelope pitch exercises non-contiguous outer state strides.
    state_storage = torch.randn(layers, capacity + 1, 2, hv, v, k, device=device, dtype=dtype) * .1
    state = state_storage[:, :, 0]
    baseline[:, base.long()] = state[:, 1:]
    # Capacity exceeds active batch AND verified width; layer offsets must use
    # the allocated pitch, not the runtime batch or block length.
    journal = (
        torch.empty(layers, capacity, hv, steps + 3, v, device=device),
        torch.empty(layers, capacity, hv, steps + 3, k, device=device),
        torch.empty(layers, capacity, hv, steps + 3, device=device),
    )
    a_log = torch.randn(layers, hv, device=device) * .2
    bias = torch.randn(layers, hv, device=device, dtype=torch.bfloat16)
    accepted_storage = torch.empty(batch * 2, device=device, dtype=torch.int32)
    accepted = accepted_storage[::2]
    valid = indices > 0
    saved_unused = state_storage[:, :, 1].clone()

    for round_id in range(4):
        x = torch.randn(layers, batch * steps, 2 * h * k + hv * v,
                         device=device, dtype=torch.bfloat16)
        gates = torch.randn(layers, batch * steps, 2 * hv, device=device, dtype=torch.bfloat16)
        a, b = gates[..., :hv], gates[..., hv:]
        update_recurrent_state_indices(current, indices.contiguous(), read, write)
        before = state.clone()
        for layer in range(layers):
            expected = packed_gdn_decode(x[layer], a[layer], b[layer], a_log[layer],
                                         bias[layer], baseline[layer], read, h, write)
            actual = packed_gdn_journal_verify(
                x[layer], a[layer], b[layer], a_log[layer], bias[layer], state[layer],
                indices, h, tuple(t[layer] for t in journal),
            )
            assert_output_close(actual, expected)
            assert_bits_equal(actual.view(batch, steps, hv, v)[~valid],
                              expected.view(batch, steps, hv, v)[~valid])
        assert_bits_equal(state, before)

        # On the first round check EVERY possible prefix against stored snapshots.
        # Each row still has its own length, including no commit (-1).
        counts = range(-1, steps) if round_id == 0 else [round_id]
        for last in counts:
            accepted.copy_((torch.arange(batch, device=device) + last + 1) % (steps + 1) - 1)
            state.copy_(before)
            expected_state = before.clone()
            rows = torch.arange(batch, device=device)[valid & (accepted >= 0)]
            destinations = indices[rows].long()
            source_slots = write[rows, accepted[rows].long()].long()
            expected_state[:, destinations] = baseline[:, source_slots]
            replay_gdn_journal(state, journal, indices, accepted)
            assert_state_close(state, expected_state)
            untouched = torch.ones(capacity + 1, device=device, dtype=torch.bool)
            untouched[destinations] = False
            assert_bits_equal(state[:, untouched], before[:, untouched])
        update_recurrent_state_indices(current, indices.contiguous(), read, write,
                                        accepted_steps=accepted.contiguous())
        assert_state_close(state[:, indices[valid].long()],
                           baseline[:, current[indices[valid].long()].long()])
        # Reorder requests between rounds; the journal is indexed by batch row.
        indices.copy_(indices.roll(1))
        valid = indices > 0
    assert_bits_equal(state_storage[:, :, 1], saved_unused)


def test_journal_cuda_graph_uses_current_indices_and_acceptance():
    dtype = torch.float32
    from sfllm.kernels.gdn import packed_gdn_decode
    from sfllm.kernels.gdn_journal import packed_gdn_journal_verify, replay_gdn_journal

    batch, steps, h, hv, k, v = 5, 9, 2, 4, 64, 96
    torch.manual_seed(47)
    state = torch.randn(2, 12, hv, v, k, device="cuda", dtype=dtype) * .1
    initial = state.clone()
    journal = tuple(torch.empty(*shape, device="cuda") for shape in (
        (2, 8, hv, steps, v), (2, 8, hv, steps, k), (2, 8, hv, steps),
    ))
    x = torch.randn(batch * steps, 2 * h * k + hv * v, device="cuda", dtype=torch.bfloat16)
    a = torch.randn(batch * steps, hv, device="cuda", dtype=torch.bfloat16)
    b = torch.randn_like(a)
    a_log = torch.randn(hv, device="cuda") * .2
    bias = torch.randn(hv, device="cuda", dtype=torch.bfloat16)
    indices = torch.arange(1, batch + 1, device="cuda", dtype=torch.int32)
    accepted = torch.zeros_like(indices)

    def run():
        outputs = [packed_gdn_journal_verify(x, a, b, a_log, bias, state[layer], indices,
                                             h, tuple(t[layer] for t in journal))
                   for layer in range(2)]
        replay_gdn_journal(state, journal, indices, accepted)
        return outputs

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = run()
    for seed in range(4):
        torch.manual_seed(seed)
        indices.copy_(torch.randperm(10, device="cuda")[:batch] + 1)
        accepted.copy_(torch.randint(-1, steps, (batch,), device="cuda"))
        x.normal_()
        state.copy_(initial)
        reference = torch.cat((initial, initial.new_zeros((2, batch * steps, hv, v, k))), dim=1)
        write = torch.arange(12, 12 + batch * steps, device="cuda", dtype=torch.int32).view(batch, steps)
        expected_outputs = [packed_gdn_decode(x, a, b, a_log, bias, reference[layer], indices, h, write)
                            for layer in range(2)]
        graph.replay()
        for actual, expected in zip(outputs, expected_outputs):
            assert_output_close(actual, expected)
        expected_state = initial.clone()
        rows = torch.arange(batch, device="cuda")[accepted >= 0]
        expected_state[:, indices[rows].long()] = reference[:, write[rows, accepted[rows].long()].long()]
        assert_state_close(state, expected_state)
        untouched = torch.ones(state.shape[1], device="cuda", dtype=torch.bool)
        untouched[indices[rows].long()] = False
        assert_bits_equal(state[:, untouched], initial[:, untouched])


@pytest.mark.parametrize("steps", [2, 5, 16])
def test_journal_matches_flashinfer_across_rounds(steps):
    from flashinfer.gdn_decode import gated_delta_rule_mtp
    from sfllm.kernels.gdn import update_recurrent_state_indices
    from sfllm.kernels.gdn_journal import packed_gdn_journal_verify, replay_gdn_journal

    batch, h, hv, k, v = 20, 16, 32, 128, 128
    torch.manual_seed(630)
    indices = torch.arange(1, batch + 1, device="cuda", dtype=torch.int32)
    base = (indices - 1) * (steps + 1) + 1
    current = torch.cat((base.new_zeros(1), base))
    read = torch.empty_like(indices)
    write = torch.empty(batch, steps, device="cuda", dtype=torch.int32)
    state = torch.randn(1, batch + 1, hv, v, k, device="cuda") * .1
    reference = torch.zeros(batch * (steps + 1) + 1, hv, v, k, device="cuda")
    reference[base.long()] = state[0, 1:]
    journal = tuple(torch.empty(*shape, device="cuda") for shape in (
        (1, batch, hv, steps, v), (1, batch, hv, steps, k), (1, batch, hv, steps),
    ))
    a_log = torch.randn(hv, device="cuda") * .2
    bias = torch.randn(hv, device="cuda", dtype=torch.bfloat16)
    for _ in range(4):
        x = torch.randn(batch * steps, 2 * h * k + hv * v, device="cuda", dtype=torch.bfloat16)
        a = torch.randn(batch, steps, hv, device="cuda", dtype=torch.bfloat16)
        b = torch.randn_like(a)
        update_recurrent_state_indices(current, indices, read, write)
        q, key, val = x.split((h * k, h * k, hv * v), dim=-1)
        expected, _ = gated_delta_rule_mtp(
            q=q.view(batch, steps, h, k), k=key.view(batch, steps, h, k),
            v=val.view(batch, steps, hv, v), A_log=a_log, a=a, dt_bias=bias, b=b,
            use_qk_l2norm=True, initial_state=reference, initial_state_indices=read,
            ssm_state_indices=write, disable_state_update=False,
        )
        actual = packed_gdn_journal_verify(x, a.view(-1, hv), b.view(-1, hv), a_log,
                                           bias, state[0], indices, h, tuple(t[0] for t in journal))
        # FlashInfer uses a different normalization/FMA order. Check numerical
        # agreement with the current FP32 serving backend as well as Triton.
        torch.testing.assert_close(actual.view_as(expected), expected, rtol=1e-2, atol=1e-4)
        accepted = torch.randint(0, steps, (batch,), device="cuda", dtype=torch.int32)
        replay_gdn_journal(state, journal, indices, accepted)
        update_recurrent_state_indices(current, indices, read, write, accepted_steps=accepted)
        torch.testing.assert_close(state[0, 1:], reference[current[1:].long()], rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
@pytest.mark.parametrize("steps", [3, 8])
@pytest.mark.parametrize("varlen", [False, True])
def test_model_verification_and_commit(monkeypatch, dtype, steps, varlen):
    import sfllm.models.qwen3_5 as qwen
    from sfllm.engine.forward_params import ForwardMode

    # Construct real GDN layers; bypass unrelated full-attention/MLP weights.
    args = SimpleNamespace(
        max_running_requests=8, speculative_algorithm="dspark",
        speculative_num_draft_tokens=steps, mamba_ssm_dtype=dtype,
        spec_adaptive_verify=f"d{steps}t1" if varlen else None,
        linear_attn_backend="triton", linear_attn_prefill_backend=None,
        linear_attn_decode_backend=None, enable_prefill_cuda_graph=False,
    )
    config = SimpleNamespace(
        hidden_size=32, vocab_size=32, num_hidden_layers=2,
        layer_types=["linear_attention"] * 2, linear_num_key_heads=2,
        linear_num_value_heads=4, linear_key_head_dim=64, linear_value_head_dim=64,
        linear_conv_kernel_dim=4, rms_norm_eps=1e-6,
    )
    monkeypatch.setattr(qwen, "get_global_server_args", lambda: args)
    monkeypatch.setattr(qwen, "get_pool_index_layers", lambda _: [])

    class Layer(torch.nn.Module):
        def __init__(self, cfg, layer_id, cache_id, state_id, buffers, *rest):
            super().__init__()
            self.linear_attn = qwen.Qwen3_5GatedDeltaNet(
                cfg, buffers[0][state_id], buffers[1][state_id], buffers[2],
            )

        def forward(self, positions, hidden, batch, residual, *args):
            return self.linear_attn(hidden, batch, *args), residual

    monkeypatch.setattr(qwen, "Qwen3_5DecoderLayer", Layer)
    models = []
    for enabled in (False, True):
        monkeypatch.setenv("SFLLM_GDN_JOURNAL", "1" if enabled else "0")
        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            with torch.device("cuda"):
                model = qwen.Qwen3_5Model(config)
        finally:
            torch.set_default_dtype(previous_dtype)
        torch.manual_seed(427)
        for param in model.parameters():
            param.data.normal_(0, .1)
        active = enabled and dtype == "float32"
        assert (model.ssm_journal is not None) == active
        assert model.ssm_states.shape[1] == 8 * (1 if active else steps + 1) + 1
        assert model.ssm_states.dtype == getattr(torch, dtype)
        if active:
            assert all(t.dtype == torch.float32 for t in model.ssm_journal)
            assert model.ssm_output_indices is None
            assert not any("journal" in key for key in model.state_dict())
        wrapper = qwen.Qwen3_5ForConditionalGeneration.__new__(qwen.Qwen3_5ForConditionalGeneration)
        torch.nn.Module.__init__(wrapper)
        wrapper.model = model
        models.append(wrapper)
    baseline, candidate = models
    # Changing the environment after construction cannot change captured state layouts.
    monkeypatch.setenv("SFLLM_GDN_JOURNAL", "0")

    for mode, count in ((ForwardMode.EXTEND, 1), (ForwardMode.TARGET_VERIFY, steps),
                        (ForwardMode.DECODE, 1), (ForwardMode.TARGET_VERIFY, 1 if varlen else steps),
                        (ForwardMode.EXTEND, 1)):
        if mode == ForwardMode.TARGET_VERIFY:
            indices = torch.tensor([1, 4, -1], device="cuda", dtype=torch.int32)
        elif mode == ForwardMode.EXTEND:
            indices = torch.tensor([4, 1, 6], device="cuda", dtype=torch.int32)
        for wrapper in models:
            wrapper.model.state_indices[:3].copy_(indices)
        lengths = ([1, count, max(1, count // 2)]
                   if varlen and mode == ForwardMode.TARGET_VERIFY else [count] * 3)
        cu = torch.tensor([0, *lengths], device="cuda", dtype=torch.int32).cumsum(0)
        batch = SimpleNamespace(forward_mode=mode, max_extend_len=count, qo_indptr=cu)
        inputs = torch.randint(config.vocab_size, (sum(lengths),), device="cuda")
        before = candidate.model.ssm_states.clone()
        if varlen and mode == ForwardMode.TARGET_VERIFY:
            # This unit test calls the model directly, without ModelRunner.
            qwen.GatedDeltaNetBackend.prepare_verify(batch)
        with torch.no_grad():
            outputs = [wrapper.model(inputs, inputs, batch) for wrapper in models]
        if dtype == "float32":
            assert_output_close(outputs[0], outputs[1])
        else:
            assert_bits_equal(outputs[0], outputs[1])
        assert bool(torch.isfinite(outputs[1]).all())
        if mode == ForwardMode.TARGET_VERIFY:
            if dtype == "float32":
                assert_bits_equal(candidate.model.ssm_states, before)
            accepted = torch.tensor([0, count - 1, -1], device="cuda", dtype=torch.int32)
            for wrapper in models:
                wrapper.commit_speculative_state(accepted)
        valid = indices[indices > 0].long()
        baseline_slots = baseline.model.ssm_current_slots[valid].long()
        candidate_slots = (valid if dtype == "float32"
                           else candidate.model.ssm_current_slots[valid].long())
        compare_state = assert_state_close if dtype == "float32" else assert_bits_equal
        compare_state(candidate.model.ssm_states[:, candidate_slots],
                      baseline.model.ssm_states[:, baseline_slots])
        if dtype == "float32":
            # The first layer has identical inputs; later layers receive the
            # small output differences from the preceding GDN reductions.
            assert_bits_equal(candidate.model.conv_states[0], baseline.model.conv_states[0])
            assert_output_close(candidate.model.conv_states, baseline.model.conv_states)
        else:
            assert_bits_equal(candidate.model.conv_states, baseline.model.conv_states)
