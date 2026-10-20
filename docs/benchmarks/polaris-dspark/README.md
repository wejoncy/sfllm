# Benchmark evidence

`results.csv` / `results.json` contain the 18 scored points. Re-render with
`python plot.py --output /tmp/comparison.png` (requires matplotlib). `summaries/` retains
the client summaries, including output lengths and latency distributions.
`commands.json` records the exact selected server commands, environment overrides,
and client commands. Local absolute paths identify the measured checkpoint and
sample; adapt these paths on another host.

`protocol.json` includes all 300 zero-based source indices, sampling seed, and
sample SHA256. Private model files and prompt text are excluded.
`sfllm-selection.json` and `sglang-selection.json` retain every eligible candidate's
screening throughput. The initial default-pool SGLang width 8 run was capacity 57;
it is eligible only through client concurrency 48. An additional native 80-slot candidate
was screened at concurrency 48 and 64. A native no_buffer prefix-cache width 8 candidate is screened at every concurrency,
with overlap disabled as required by upstream.
Width 16 uses no_buffer with 64 slots and static memory fraction 0.93 and prefill chunk 4096; other candidates use 0.9.
All formal points have enough server slots.

`native-adaptive-command.json` and `native-adaptive-screen.json` show the native
adaptive candidate at all nine concurrencies. `native-sps.json` is fitted with the
unmodified upstream profiler from the adjacent rounds. B1–B32 costs are reused
from the first calibration; B64 costs are supplemented after native pool sizing.
Acceptance simulation is used only for compute-cost profiling, never scored or
tuning generation. Native GDN compact ragged graphs are unsupported at this SHA;
the native adaptive candidate uses cap-accept.

`source-provenance.json`, `upstream-runtime-imports.json`, and
`dependency-integrity.json` document source and runtime identity.
`verification.json` records the completed workload/source/capacity/acceptance audit.
Source hash-map digests use SHA256 of compact sorted JSON mapping paths to hashes.

The SFLLM measured source revision is 1755593, available on qwen35_specfic, which
is ahead of the main branch used as the base of this documentation-only PR.
