"""Budget packing and accepted-prefix indexing across CUDA Graph replays."""

from types import SimpleNamespace

import pytest
import torch

from sfllm.kernels.verify_budget import allocate_verify_budget
from sfllm.spec_decoding.spec_utils import SpecVerifyInput

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("batch,width,budget", [
    (20, 6, 100), (22, 6, 110), (3, 9, 14), (3, 8, 3), (3, 8, 24),
])
def test_budget_graph_matches_prefix_oracle(batch, width, budget):
    scores = torch.zeros(batch, width - 1, device="cuda")
    candidates = torch.arange(batch * width, device="cuda").view(batch, width)
    allocate_verify_budget(scores, candidates, budget)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = allocate_verify_budget(scores, candidates, budget)

    generator = torch.Generator().manual_seed(42)
    for kind in ("random", "ties", "zero_probability"):
        probabilities = torch.rand(batch, width - 1, generator=generator)
        if kind == "ties":
            probabilities.fill_(1)
        elif kind == "zero_probability":
            probabilities[:, 2:] = 0
        prefix = probabilities.log().cumsum(-1)
        scores.copy_(prefix)
        graph.replay()
        lengths, boundaries, source, packed = (t.cpu() for t in outputs)
        # Independently take the best next prefix, respecting request order on ties.
        expected = [1] * batch
        for _ in range(budget - batch):
            row = max((i for i in range(batch) if expected[i] < width),
                      key=lambda i: (float(prefix[i, expected[i] - 1]), -i))
            expected[row] += 1
        expected_source = torch.cat([
            torch.arange(row * width, row * width + length)
            for row, length in enumerate(expected)
        ])
        assert lengths.tolist() == expected
        assert boundaries.tolist() == [0] + torch.tensor(expected).cumsum(0).tolist()
        assert torch.equal(source, expected_source)
        assert torch.equal(packed, candidates.cpu().flatten()[expected_source])


def test_packed_accept_stops_at_each_request_budget():
    batch, width, vocab = 3, 6, 32
    candidates = torch.arange(batch * width, device="cuda").view(batch, width)
    indices = torch.arange(batch * width, device="cuda").view(batch, width)
    next_token = torch.arange(1, width + 1, device="cuda").repeat(batch, 1)
    lengths = torch.tensor([1, 4, 6], device="cuda")
    source = torch.cat([indices[0, :1], indices[1, :4], indices[2, :6]])
    proposal = SpecVerifyInput(
        draft_token=candidates.flatten(), custom_mask=None, positions=indices.flatten(),
        retrive_index=indices,
        retrive_next_token=torch.where(next_token < lengths[:, None], next_token, -1),
        retrive_next_sibling=torch.full_like(indices, -1), retrive_cum_len=None,
        spec_steps=width - 1, topk=1, draft_token_num=width, packed_source_indices=source,
    )
    predicted = (candidates + 1).flatten()[source]
    predicted[6] = 31  # Third request rejects its second draft; others match throughout.
    logits = torch.full((source.numel(), vocab), -100., device="cuda")
    logits.scatter_(1, predicted[:, None], 100.)
    output = SimpleNamespace(next_token_logits=logits)
    accepted_indices, accepted, tokens = proposal.verify([None] * batch, output, 1)
    assert accepted.tolist() == [0, 3, 1]
    assert accepted_indices.tolist() == [[0, -1, -1, -1, -1, -1],
                                       [6, 7, 8, 9, -1, -1],
                                       [12, 13, -1, -1, -1, -1]]
    assert tokens[accepted_indices[accepted_indices >= 0]].tolist() == [1, 7, 8, 9, 10, 13, 31]
