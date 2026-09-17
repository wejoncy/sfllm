"""Packed target verification against independent single-token GDN forwards."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("kind", ["journal", "fp32", "bf16"])
def test_varlen_verify_outputs_and_states(kind):
    from sfllm.kernels.gdn import GatedDeltaNetBackend, scatter_recurrent_state
    from sfllm.kernels.gdn_journal import replay_gdn_journal

    torch.manual_seed(81)
    batch, capacity, h, hv, k, v = 20, 8, 2, 4, 64, 80
    slots, conv_width = batch + 5, 4
    lengths = torch.tensor([(i * 3) % 8 + 1 for i in range(batch)])
    tokens = int(lengths.sum())
    cu = torch.empty(batch + 1, device="cuda", dtype=torch.int64)
    indices = torch.empty(batch * 2, device="cuda", dtype=torch.int32)[::2]
    dtype = torch.bfloat16 if kind == "bf16" else torch.float32
    conv_dim = 2 * h * k + hv * v
    x = torch.randn(tokens, conv_dim + hv * v, device="cuda", dtype=torch.bfloat16)
    gates = torch.randn(tokens, 2 * hv, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(conv_dim, conv_width, device="cuda", dtype=torch.bfloat16) * .1
    a_log = torch.randn(hv, device="cuda") * .2
    bias = torch.randn(hv, device="cuda", dtype=torch.bfloat16)
    initial = torch.randn(slots, hv, v, k, device="cuda", dtype=dtype) * .1
    initial_conv = torch.randn(slots, conv_dim, conv_width - 1,
                               device="cuda", dtype=torch.bfloat16)
    state = (initial.clone() if kind == "journal" else
             torch.cat((initial, initial.new_empty(batch * capacity, hv, v, k))))
    conv = initial_conv.clone()
    intermediate = torch.empty(1, batch, capacity, conv_dim, conv_width - 1,
                              device="cuda", dtype=torch.bfloat16)
    writes = torch.arange(slots, slots + batch * capacity, device="cuda",
                          dtype=torch.int32).view(batch, capacity)
    journal = tuple(torch.empty(*shape, device="cuda") for shape in (
        (1, batch, hv, capacity, v), (1, batch, hv, capacity, k),
        (1, batch, hv, capacity),
    )) if kind == "journal" else None
    backend = GatedDeltaNetBackend("triton", "triton")

    def forward():
        return backend.decode(
            x, gates, conv_weight=weight, conv_states=conv, ssm_states=state,
            state_indices=indices, a_log=a_log, dt_bias=bias, num_k_heads=h,
            num_v_heads=hv, head_k_dim=k, head_v_dim=v,
            intermediate_conv=intermediate[0],
            ssm_output_indices=writes if journal is None else None,
            ssm_journal=tuple(t[0] for t in journal) if journal is not None else None,
            cu_seqlens=cu,
        )

    def metadata(round_id):
        # Exercise different request lengths and state slots, including 1 and 8.
        lens = lengths.roll(round_id * 3)
        if round_id == 2:
            # An empty graph row must not read the next request's first token.
            donor = next(i for i, n in enumerate(lens) if n == 1 and i not in (2, 7))
            recipient = next(i for i, n in enumerate(lens) if n == 7)
            lens[donor], lens[recipient] = 0, 8
        cu.copy_(torch.cat((lens.new_zeros(1), lens.cumsum(0))))
        indices.copy_(torch.randperm(slots - 1, device="cuda")[:batch] + 1)
        indices[2], indices[7] = 0, -1
        return lens.tolist(), indices.tolist()

    for round_id in range(3):
        lens, request_slots = metadata(round_id)
        x.normal_()
        gates.normal_()
        state[:slots].copy_(initial)
        if journal is None:
            state[slots:].fill_(17)
        conv.copy_(initial_conv)
        intermediate.fill_(17)
        if journal is not None:
            for tensor in journal:
                tensor.fill_(17)
        outputs = forward()
        # Verify must leave every request's persistent starting state intact.
        assert torch.equal(state[:slots], initial)
        assert torch.equal(conv, initial_conv)

        ref_state, ref_conv = initial.clone(), initial_conv.clone()
        expected_outputs, expected_conv, expected_state = [], [], []
        begin = 0
        for row, (length, slot) in enumerate(zip(lens, request_slots)):
            row_conv, row_state = [], []
            for step in range(length):
                token = begin + step
                core, _ = backend.decode(
                    x[token:token + 1], gates[token:token + 1], conv_weight=weight,
                    conv_states=ref_conv, ssm_states=ref_state,
                    state_indices=indices[row:row + 1], a_log=a_log, dt_bias=bias,
                    num_k_heads=h, num_v_heads=hv, head_k_dim=k, head_v_dim=v,
                )
                expected_outputs.append(core)
                if slot > 0:
                    row_conv.append(ref_conv[slot].clone())
                    row_state.append(ref_state[slot].clone())
            expected_conv.append(row_conv)
            expected_state.append(row_state)
            begin += length
            if slot > 0 and length:
                assert torch.equal(intermediate[0, row, :length], torch.stack(row_conv))
                if journal is None:
                    torch.testing.assert_close(
                        state[writes[row, :length]], torch.stack(row_state), rtol=1e-2, atol=1e-6,
                    )
            assert bool((intermediate[0, row, length:] == 17).all())
            if slot <= 0:
                assert bool((intermediate[0, row] == 17).all())
            if journal is not None:
                for tensor in journal:
                    assert bool((tensor[0, row, :, length:] == 17).all())
            else:
                unused = writes[row, length:] if slot > 0 else writes[row]
                assert bool((state[unused] == 17).all())

        torch.testing.assert_close(outputs[0], torch.cat(expected_outputs), rtol=1e-2, atol=1e-6)
        torch.testing.assert_close(outputs[1], x[:, conv_dim:].view(tokens, hv, v), rtol=0, atol=0)

        # Commit independently chosen prefixes. Padding rows never commit.
        last = torch.tensor([(row + round_id) % (length + 1) - 1 if request_slots[row] > 0 else -1
                             for row, length in enumerate(lens)], device="cuda", dtype=torch.int32)
        expected_final, expected_final_conv = initial.clone(), initial_conv.clone()
        for row, (slot, step) in enumerate(zip(request_slots, last.tolist())):
            if slot > 0 and step >= 0:
                expected_final[slot] = expected_state[row][step]
                expected_final_conv[slot] = expected_conv[row][step]
        scatter_recurrent_state(intermediate, conv.unsqueeze(0), indices.contiguous(), last)
        assert torch.equal(conv, expected_final_conv)
        if journal is not None:
            replay_gdn_journal(state.unsqueeze(0), journal, indices, last)
            actual_final = state
        else:
            actual_final = initial.clone()
            for row, (slot, step) in enumerate(zip(request_slots, last.tolist())):
                if slot > 0 and step >= 0:
                    actual_final[slot] = state[writes[row, step]]
        torch.testing.assert_close(actual_final, expected_final,
                                   rtol=1e-2 if kind == "bf16" else 1e-5,
                                   atol=1e-6 if kind == "bf16" else 2e-7)


@pytest.mark.parametrize("kind,attention_backend,varlen", [
    ("journal", "fa3", True),
    ("fp32", "fa3", True),
    ("bf16", "fa3", True),
    ("journal", "triton", True),
    ("journal", "fa3", False),
    ("bf16", "triton", False),
])
@torch.inference_mode()
def test_target_forward_and_commit_cuda_graph(monkeypatch, kind, attention_backend, varlen):
    from types import SimpleNamespace

    from transformers import Qwen3_5Config, Qwen3_5TextConfig

    import sfllm.server_args as server_args
    import sfllm.kernels.gdn as gdn
    from sfllm.engine.forward_params import ForwardMode
    from sfllm.engine.model_runner import ModelRunner
    from sfllm.engine.schedule_batch import ScheduleBatch
    from sfllm.layers.radix_attention import collect_attention_metadata
    from sfllm.layers.sampler import Sampler
    from sfllm.model_loader.model_loader import TorchDefaultReset
    from sfllm.models.qwen3_5 import Qwen3_5ForConditionalGeneration

    monkeypatch.setattr(server_args, "_global_server_args", None)
    monkeypatch.setenv("SFLLM_GDN_JOURNAL", "1" if kind == "journal" else "0")
    if varlen:
        monkeypatch.setenv("SFLLM_ENABLE_VARLEN_VERIFY", "1")
    else:
        monkeypatch.delenv("SFLLM_ENABLE_VARLEN_VERIFY", raising=False)
    batch_size, total, capacity = 4, 16, 8 if varlen else 4
    args = server_args.ServerArgs(
        model_path="unused", speculative_algorithm="dspark",
        speculative_num_draft_tokens=capacity, max_running_requests=batch_size,
        cuda_graph_max_bs=batch_size, max_context_length=128,
        mamba_ssm_dtype="bfloat16" if kind == "bf16" else "float32",
        attention_backend=attention_backend, linear_attn_backend="triton",
    )
    config = Qwen3_5TextConfig(
        vocab_size=128, hidden_size=128, intermediate_size=256, num_hidden_layers=3,
        layer_types=["linear_attention", "full_attention", "linear_attention"],
        num_attention_heads=2, num_key_value_heads=1, head_dim=64,
        linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=64, linear_value_head_dim=64,
        max_position_embeddings=256,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.,
                         "partial_rotary_factor": .5},
    )
    config.attn_output_gate = True
    torch.manual_seed(936)
    with TorchDefaultReset(torch.bfloat16), collect_attention_metadata() as metadata:
        model = Qwen3_5ForConditionalGeneration(Qwen3_5Config(text_config=config))
    for param in model.parameters():
        param.normal_(0, .1)
    model.set_layers_to_capture([0, 2])
    backbone = model.model
    # Construction fixes graph dispatch; later environment changes cannot alter it.
    monkeypatch.setenv("SFLLM_ENABLE_VARLEN_VERIFY", "0" if varlen else "1")
    backbone.ssm_states.normal_(0, .1)
    backbone.conv_states.normal_(0, .1)

    order_calls = 0
    original_order = gdn.gdn_verify_request_order

    def prepare_order(cu_seqlens):
        nonlocal order_calls
        order_calls += 1
        return original_order(cu_seqlens)

    monkeypatch.setattr(gdn, "gdn_verify_request_order", prepare_order)

    # Use the serving forward/attention preparation inside one outer graph,
    # as the speculative runner does. No target-specific graph is installed.
    runner = ModelRunner.__new__(ModelRunner)
    runner.server_args, runner.model = args, model
    runner.device_id, runner.dtype, runner.rank, runner.is_draft = 0, torch.bfloat16, 0, False
    runner.cuda_graphs, runner.cuda_graphs_extend, runner.cuda_graphs_target_verify = {}, {}, {}
    runner.prefill_graph_runner, runner.sampler = None, Sampler(config)
    runner.num_kv_splits_buffer = torch.full((batch_size,), 2, device="cuda", dtype=torch.int32)
    runner.init_attn_backend_buffers(metadata)
    kv = [torch.randn(batch_size * 64, 1, 64, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    batch = ScheduleBatch([None] * batch_size, SimpleNamespace(kv_buffers=[tuple(kv)]))
    fb = batch.forward_batch
    # The E2E runner captures with the configured capacity, not the first
    # batch's actual maximum query length.
    fb.forward_mode, fb.max_extend_len = ForwardMode.TARGET_VERIFY, capacity
    fb.qo_indptr = torch.arange(batch_size + 1, device="cuda", dtype=torch.int32) * 4
    fb.kv_indptr = torch.arange(batch_size + 1, device="cuda", dtype=torch.int32) * 16
    fb.kv_indices = torch.cat([torch.arange(r * 64, r * 64 + 16, device="cuda")
                              for r in range(batch_size)])
    fb.out_cache_loc = torch.empty(total, device="cuda", dtype=torch.int64)
    fb.seq_lens = torch.full((batch_size,), 16, device="cuda", dtype=torch.int32)
    batch.input_ids = torch.empty(total, device="cuda", dtype=torch.int64)
    batch.position_ids = torch.empty_like(batch.input_ids)
    runner.bind_cuda_graph_logits_buffer(fb, total)
    accepted = torch.zeros(batch_size, device="cuda", dtype=torch.int32)

    def set_inputs(lengths, round_id):
        lens = torch.tensor(lengths, device="cuda", dtype=torch.int32)
        fb.qo_indptr[1:].copy_(lens.cumsum(0))
        batch.input_ids.random_(config.vocab_size)
        batch.position_ids.copy_(torch.cat([torch.arange(16, 16 + n, device="cuda") for n in lengths]))
        fb.out_cache_loc.copy_(torch.cat([torch.arange(r * 64 + 16, r * 64 + 16 + n, device="cuda")
                                         for r, n in enumerate(lengths)]))
        backbone.state_indices.copy_(torch.randperm(batch_size, device="cuda", dtype=torch.int32) + 1)
        accepted.copy_((torch.arange(batch_size, device="cuda") + round_id) % lens)

    state_buffers = [backbone.ssm_states, backbone.conv_states]
    if backbone.ssm_journal is None:
        state_buffers.append(backbone.ssm_current_slots)

    def run():
        before_order_calls = order_calls
        result = runner.forward(batch)
        assert order_calls - before_order_calls == int(varlen)
        model.commit_speculative_state(accepted)
        return result

    set_inputs([4] * batch_size, 0)
    initial = [t.clone() for t in state_buffers]
    run()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream, pool=torch.cuda.graph_pool_handle()):
        output = run()
    torch.cuda.current_stream().wait_stream(stream)
    for dst, src in zip(state_buffers, initial):
        dst.copy_(src)

    # One capture, unchanged addresses/total, including lengths above the width
    # seen at capture. Continue each round from the previously committed state.
    length_batches = ([4, 4, 4, 4], [1, 8, 6, 1], [7, 2, 3, 4], [4, 4, 4, 4])
    if not varlen:
        length_batches = ([4, 4, 4, 4],) * 4
    for round_id, lengths in enumerate(length_batches):
        set_inputs(lengths, round_id)
        before = [t.clone() for t in state_buffers]
        eager = run()
        expected_logits = eager.next_token_logits.clone()
        expected_hidden = [t.clone() for t in eager.aux_hidden_states]
        expected_states = [t.clone() for t in state_buffers]
        for dst, src in zip(state_buffers, before):
            dst.copy_(src)
        graph.replay()
        torch.testing.assert_close(output.next_token_logits, expected_logits, rtol=0, atol=0)
        for actual, expected in zip(output.aux_hidden_states, expected_hidden):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for actual, expected in zip(state_buffers, expected_states):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
