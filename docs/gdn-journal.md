# FP32 GDN verification journals

Set `SFLLM_GDN_JOURNAL=1` before starting a Qwen3.5/Qwen3.8 target with
DFlash2 or DSpark to enable the Triton journal path. It uses the configured
`--speculative-num-draft-tokens` block length, including the anchor. There
is no width-six or fixed head-count specialization.

The flag defaults to off and is read once during model construction. FP32
state uses the journal when enabled; BF16 always uses its existing path.
Prefill and ordinary decode continue to use the selected backends. Changing
the environment after model construction does not change state layouts or
captured graphs.

## What changes

Current verification writes a complete recurrent state for every candidate.
Commit selects the accepted snapshot by changing an index; it already avoids
a large state copy. The journal reduces the verification writes and state
allocation, at the cost of replaying the accepted prefix.

For a V-by-K state, verification already computes:

```text
update[t] = beta[t] * (v[t] - decay[t] * state[t-1] @ key[t])
state[t]  = decay[t] * state[t-1] + update[t] @ key[t].T
```

The new kernel records the computed FP32 update vector, normalized key and
decay factor. It does not modify persistent state during verification. Commit
reads each request's own GPU acceptance index and replays those recorded
updates in a single launch across all GDN layers. Index zero commits the
anchor; index -1 commits nothing. Both zero and negative state slots are
padding, and request slots can change order between rounds.

The persistent state and journal operands stay FP32. The replay kernel keeps
the rounded decay multiplication separate from the subsequent rank-one FMA.
It avoids recomputing gates, normalization, predictions or outputs.

For K=V=128, a candidate snapshot is 65,536 bytes per head, while a journal
entry is 1,028 bytes. This is about 63.75 times less candidate-cache data, not
an equivalent latency speedup. The implementation allocates journals from
model dimensions and configured block capacity, with separate layer strides
for replay and 64-bit offsets for large allocations.

## Validation

GPU tests cover blocks 1, 2, 3, 6, 8, 9, 16 and 31, every accepted prefix,
heterogeneous acceptance lengths, no commit, padding, request reordering,
strided indices, envelope-strided states and journals whose capacity exceeds
the active batch/block. Outputs and committed states are compared bitwise
with the existing FP32 Triton verification path over successive rounds.

Separate tests compare against FlashInfer with numerical tolerances, since
its normalization and FMA order differ. Model integration tests exercise
prefill, verification, commit, ordinary decode and request-slot reuse for
both DFlash2 and DSpark. They also check that BF16 ignores the environment
flag and retains its state layout and results.

```bash
PYTHONPATH=python python -m pytest tests/test_gdn_journal.py tests/test_qwen3_8.py
```

## Performance

Measurements use an H100 NVL, concurrency 20, 27 GDN layers, H=16, HV=32,
K=V=128, BF16 projected inputs and FP32 state. Each CUDA graph executes
32 consecutive rounds. Each round selects a new GPU vector of per-request
acceptance lengths after verification; the accepted states feed the next
round. All backends use the same trace, with shuffled timing order.

The benchmark includes SSM index preparation where needed, verification, a
common synthetic acceptance selector and commit/replay. It excludes
projection, convolution, sampling and other model operations. Input
projections are fixed synthetic tensors; the acceptance distributions are
synthetic, not measured serving acceptance rates. Results are kernel latency,
not end-to-end throughput.

Median milliseconds per round across all 27 layers, five timing trials.
This table uses uniform per-request acceptance; negative latency changes are faster.

| Block | Current Triton | Current FlashInfer | Journal | Change vs FlashInfer |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 1.345 | 1.173 | 1.583 | +34.9% |
| 4 | 2.250 | 2.036 | 2.206 | +8.3% |
| 6 | 3.169 | 2.927 | 2.804 | -4.2% |
| 8 | 4.047 | 3.792 | 3.391 | -10.6% |
| 16 | 7.464 | 9.193 | 5.585 | -39.2% |

Block 6 under the other acceptance distributions:

| Distribution | Mean replay count (statistic only) | FlashInfer | Journal | Change |
| --- | ---: | ---: | ---: | ---: |
| short | 2.138 | 2.923 | 2.751 | -5.9% |
| uniform | 3.553 | 2.927 | 2.804 | -4.2% |
| long | 4.922 | 2.926 | 2.839 | -3.0% |
| bimodal | 3.625 | 2.936 | 2.796 | -4.8% |

The means above describe the saved traces; the kernel uses each row's own count.
All distributions, trial samples, trace digests and source hashes are in the
[recorded results](../benchmark/results/gdn_journal_h100_nvl.json).

At block 6, this configuration reduces SSM state plus journal allocation
from 7.436 GiB to 1.207 GiB. Convolution buffers and model weights are excluded.

Short blocks can be slower: replay's additional full-state read and write
can outweigh the avoided snapshots. The opt-in flag does not impose a minimum
block length; measure the intended workload before enabling it.

A BF16 prototype was also measured before narrowing this change to FP32.
At block 6 with uniform per-request acceptance it took 1.715 ms versus
1.623 ms for the current BF16 path. Benefits were not consistent across block
sizes, so this PR leaves BF16 kernels and state allocation unchanged.

Reproduce the FP32 comparison from an installed checkout:

```bash
PYTHONPATH=python python benchmark/bench_gdn_journal.py \
  --batch 20 --layers 27 --steps 2 4 6 8 16 \
  --rounds 32 --trials 5 --output /tmp/gdn_journal.json
```

The output records the full acceptance traces, all timing samples, software
versions and source hashes. The short and long distributions use exponentially
decaying weights toward opposite ends of the block; uniform uses all prefixes
and bimodal uses the two endpoints. Every replay length includes the anchor.
