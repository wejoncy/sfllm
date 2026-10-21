# Polaris comparison: configuration alignment audit

These diagnostic screens use the same sampled prompts as the original comparison, but run 32 requests at concurrency 12 and 128 at concurrency 64. Each point runs once after a 128-request / concurrency-64 warmup and a cache flush. Do not substitute these numbers for the original 300-request formal curve.

Both engines keep the original target and six-layer draft checkpoints, input token lengths, temperature 0, max output 1000, seed 42, stop token 248044, context 16384, BF16 model dtype, and FP32 recurrent state. All successful checks validate prompt token lengths against the original sample and require every request to succeed.

The same-draft ablation below still uses SGLang's default multimodal wrapper with raw text requests. It does not align the pure-text execution path; `lm-only` separately tests the native `--language-model-only` option. This option disables the multimodal mRoPE position path that rejected nonuniform compact verify lengths.

## Same draft precision, proposal policy and verify width

The SFLLM diagnostic disables selective draft FP8 and top-k16 proposals, using BF16 draft and the original full-vocabulary greedy proposal path instead. Both engines verify width 8. SFLLM retains its engine optimizations, including GDN journal and its CUDA graph coverage. This is a diagnostic of the configuration mismatch, not the best-configuration comparison.

| Concurrency | Requests | SF BF16/full-vocab tok/s | SG BF16/full-vocab tok/s | SF accept | SG accept |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 12 | 32 | 3310.60 | 1734.82 | 3.4819 | 3.4362 |
| 64 | 128 | 4355.57 | 2168.48 | 3.3084 | 3.3527 |

Accept length uses `(output tokens - requests) / verify calls`, consistently for both engines.

## Unmodified SGLang native option checks

| Native configuration | C12 tok/s | C64 tok/s | C12 accept | C64 accept |
| --- | ---: | ---: | ---: | ---: |
| baseline | 1734.82 | 2168.48 | 3.4362 | 3.3527 |
| draft-fp8 | 1729.11 | 2196.66 | 3.4338 | 3.3369 |
| tokenizer4 | 1736.86 | 2175.39 | 3.4362 | 3.3350 |
| verify-triton | 1816.75 | 2137.48 | 3.5030 | 3.3213 |

- `compact` did not produce a valid result: Server exited -9; see /mnt/data/jicwen/work/sglang-sfllm-migration-20261019/alignment-audit-20261021/compact/server.log
  - `RuntimeError: shape '[64, 7, 16, 128]' is invalid for input of size 983040`
  - `Exception: Capture cuda graph failed: shape '[64, 7, 16, 128]' is invalid for input of size 983040`

- `compact-triton` did not produce a valid result: Command '['/workspace/sglang/.venv/bin/python', '-B', '/mnt/data/work/dspark_repro_download/qwe35_dspark_qaa/dspark_repro/benchmark/bench_sglang_raw.py', '--url', 'http://127.0.0.1:30000/generate', '--input', '/mnt/data/jicwen/work/sglang-sfllm-migration-20261019/curve-upstream-polaris-20261020/polaris-random300-seed42.jsonl', '--num-prompts', '128', '--concurrency', '64', '--max-new-tokens', '1000', '--temperature', '0', '--seed', '42', '--stop-token-id', '248044', '--request-timeout-seconds', '1800', '--output-dir', '/mnt/data/jicwen/work/sglang-sfllm-migration-20261019/alignment-audit-20261021/compact-triton/warmup/bench', '--label', 'compact-triton-warmup']' returned non-zero exit status 1.
  - `RuntimeError: shape '[64, -1]' is invalid for input of size 480`

- `compile` did not produce a valid result: Server exited -9; see /mnt/data/jicwen/work/sglang-sfllm-migration-20261019/alignment-audit-20261021/compile/server.log
  - `Exception: Capture cuda graph failed: CUDA error: operation failed due to a previous error during capture`

The baseline uses the original official checkout and native Triton draft attention, FA3 target attention, FlashInfer GDN, greedy drafting, overlap scheduling, prefix caching, and CUDA graphs. The explicit Mamba pool provides at least 64 running-request slots. Each alternative changes only its named native option; commands and before/after source manifests are retained alongside this report. No serving source is edited.

The previous native cap-accept adaptive screen covered all nine original concurrencies with a native SPS cost table through B64. Cap-accept still physically verifies width 8. The compact check uses that same table and native compact mode; its graph fallback or compatibility must be judged from its actual logs, not inferred from the flag.

The compact-triton exploratory check, if present, reuses the existing FlashInfer-based SPS table rather than claiming a Triton-specific optimum. A change of verify backend requires refitting native SPS costs before a tuned adaptive-performance claim.

The earlier native ReplaySSM attempt was rejected with `--enable-linear-replayssm-spec with DSPARK/DFLASH requires a KDA model; got a non-KDA model.` It is not a usable native option for this GDN target in the tested revision.

## Limits on the comparison

- The original curve used SF selective draft FP8 and top-k16 versus SG BF16/full-vocabulary drafting. It is not a same-draft-precision ablation.
- The original SF server used four tokenizer workers; SG used one. The tokenizer4 candidate tests this difference independently.
- The original SG command did not enable `--language-model-only`, whereas SF loads its text model. Text-only static and compact checks are required before treating the native configuration review as complete.
- Dependencies differ (versions recorded in results.json). The isolated, working environments are preserved.
- The original C64 formal cache records show SF reused 141488 / 1351215 input tokens (10.47%); SG reported 132160 cached tokens (9.78%). Both started with flushed prefix caches.
- A normal mean acceptance length and a clean source checkout do not establish globally optimal settings or token-for-token output equality. These screens do not establish statistical significance or isolate the cause of the remaining throughput gap.

Full commands and machine-readable results: `results.json`. Full server logs, per-request client records, and before/after source manifests remain in this audit directory.
