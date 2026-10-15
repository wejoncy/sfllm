from itertools import accumulate
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="Short paged attention requires SM90 or newer.",
)
def test_short_query_paged_attention_varlen_and_graph_replay():
    from sfllm.kernels.fa3_attention import fa3_attention_fwd

    torch.manual_seed(42)
    query_lengths = [1, 8, 6, 3, 0, 4, 2, 7]
    kv_lengths = [0, 1037, 191, 3, 0, 135, 1200, 17]
    tokens = sum(query_lengths)
    # Different token/head strides and shuffled physical slots exercise addressing.
    q = torch.randn(tokens + 3, 32, 256, device="cuda", dtype=torch.bfloat16)[:, ::2]
    k = torch.randn(sum(kv_lengths), 8, 256, device="cuda", dtype=q.dtype)[:, ::2]
    v = torch.randn(sum(kv_lengths), 12, 256, device="cuda", dtype=q.dtype)[:, ::3]
    out = torch.empty(q.shape, device="cuda", dtype=q.dtype)
    table = torch.full(
        (len(query_lengths), max(kv_lengths) + 5),
        2**30, device="cuda", dtype=torch.int32,
    )
    slots = torch.randperm(k.shape[0], device="cuda").int()
    offset = 0
    for req, length in enumerate(kv_lengths):
        table[req, :length] = slots[offset:offset + length]
        offset += length
    metadata = SimpleNamespace(
        page_table=table,
        cache_seqlens=torch.tensor(kv_lengths, device="cuda", dtype=torch.int32),
        qo_indptr=torch.tensor(
            [0, *accumulate(query_lengths)], device="cuda", dtype=torch.int32,
        ),
        max_query_len=8,
    )

    def forward():
        return fa3_attention_fwd(q, out, k, v, metadata, 256**-0.5, True, layer_id=0)

    forward()
    reference = torch.zeros_like(out)
    query_begin = 0
    for req, (query_len, kv_len) in enumerate(zip(query_lengths, kv_lengths)):
        if query_len and kv_len:
            indices = table[req, :kv_len].long()
            keys = k[indices].repeat_interleave(4, dim=1).float().transpose(0, 1)
            values = v[indices].repeat_interleave(4, dim=1).float().transpose(0, 1)
            queries = q[query_begin:query_begin + query_len].float().transpose(0, 1)
            scores = queries @ keys.transpose(-2, -1) * 256**-0.5
            causal = (
                torch.arange(kv_len, device="cuda")[None, :]
                <= kv_len - query_len + torch.arange(query_len, device="cuda")[:, None]
            )
            probabilities = scores.masked_fill(~causal, -float("inf")).softmax(-1)
            reference[query_begin:query_begin + query_len] = (
                probabilities.nan_to_num() @ values
            ).transpose(0, 1)
        query_begin += query_len
    torch.testing.assert_close(out, reference, atol=0.008, rtol=0.02)
    assert torch.equal(out[tokens:], torch.zeros_like(out[tokens:]))

    expected = out.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        forward()
    graph.replay()
    torch.testing.assert_close(out, expected, atol=0, rtol=0)
    # The same captured launches must read updated lengths, including empty KV.
    metadata.cache_seqlens.zero_()
    graph.replay()
    assert torch.equal(out, torch.zeros_like(out))
