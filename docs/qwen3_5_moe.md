# Qwen3.5-35B-A3B

SFLLM supports BF16 text generation from
[Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B).
The dedicated [`qwen3_5_moe.py`](../python/sfllm/models/qwen3_5_moe.py)
registers the checkpoint's `Qwen3_5MoeForConditionalGeneration` architecture.
It reuses Qwen3.5 attention, Gated DeltaNet, recurrent states and weight loading.
Model selection follows the checkpoint's `architectures` field.

All 40 decoder layers use MoE: each token selects 8 of 256 routed experts,
plus a sigmoid-gated shared expert. The attention stack has 30 Gated DeltaNet
and 10 full-attention layers.

`--moe-runner-backend auto` selects `flashinfer_cutlass` on Hopper when its
dependencies are installed and shared/routed expert widths match, and
`triton_kernel` otherwise. Both backends
can also be selected explicitly:

- `flashinfer_cutlass` packs the shared expert into the routed expert tensors
  and calls FlashInfer's fused CUTLASS MoE. The library handles dispatch, both
  grouped GEMMs, SiLU-and-multiply and weighted finalization. The packed router
  uses the local Triton top-k kernel; TorchInductor fuses appending the shared
  expert ID and sigmoid gate. Loading converts Qwen's `[gate, up]` weights to
  CUTLASS's `[up, gate]` layout, retaining only one weight copy. FlashInfer tunes
  decode and prefill buckets before graph capture and persists tactics under
  `~/.cache/flashinfer/sfllm-moe-*.json`. One backend per model retains library
  scratch separately for each CUDA stream and power-of-two token capacity.
- `triton_kernel` uses local MoE kernels for top-k, ragged metadata, both
  grouped GEMMs with gather/scatter, and weighted output reduction. The existing
  `sfkernels` SiLU-and-multiply supplies Qwen's activation: the upstream
  `swiglu_fn` includes an incompatible +1 term. Weights retain the checkpoint's
  `[gate, up]` layout; GEMMs consume transpose views. The shared expert reuses
  the existing Qwen MLP and its sigmoid gate, allowing a different width.

The source at [`sfllm/kernels/moe_triton`](../python/sfllm/kernels/moe_triton/)
contains MoE forward kernels and their coupled helpers, extracted from
**Triton official v3.7.1**, commit `f797708c0626e5f9840ca5b0a98790e2c7cb09ad`.
The upstream MIT license and source hashes are retained. Training, distributed
communication, generic conversion APIs, metadata remapping and reference
implementations are omitted. Grouped GEMM, top-k, metadata construction and
reduction GPU calculations and launch heuristics retain the measured upstream
implementation. Triton 3.7.1 or newer compiles these local sources; no separate
MoE Python package is required.

**Performance requirement:** the FlashInfer/CUTLASS execution path must remain
unaffected by the Triton backend. Its top-k GPU kernel and launch settings,
aligned packed weights, shared-expert fusion, workspace and tuning parameters
are retained. The routing wrapper uses the same kernel from local source.
Install `sfllm[flashinfer]` for that path: it includes `flashinfer-python>=0.6.18`
and `sglang-kernel>=0.4.6.post1` for FA3 attention. No SGLang server or external
MoE package is imported.

SFLLM does not use a Python loop over experts or MoE-specific token chunking.
Routing buffers use
power-of-two capacities to bound the library's bitmatrix stride specializations;
`n_rows` excludes padding from expert counts without changing model inputs.

The packed router stores 272 output rows for this checkpoint: 256 routed
experts, one shared gate and 15 unused alignment rows. The 16-column alignment
allows faster cuBLAS kernels than the unaligned 257-column GEMM. Only the 256
real routed logits enter top-k/softmax; padding cannot become an expert. The
router writes directly into bucketed output storage, avoiding a separate copy.
This changes neither the model's expert count nor the input tokens.

Routing computes softmax in FP32 and stores selected normalized weights in
activation dtype. Equal logits select the lowest expert index. Library top-k
requires a power of two up to 32; this checkpoint uses 8. Packing the shared
router changes the BF16 GEMM shape and can round near-tied logits differently,
changing expert selection and greedy continuations compared with separate
routing. Component tests check each backend against independent expert GEMMs
using its actual router logits.

## Start the server

```bash
pip install -e '.[flashinfer]'
hf download Qwen/Qwen3.5-35B-A3B --local-dir /mnt/data/work/Qwen3.5-35B-A3B
python python/sfllm/serving/app.py \
  --model /mnt/data/work/Qwen3.5-35B-A3B \
  --dtype bfloat16 --attention-backend fa3 \
  --moe-runner-backend flashinfer_cutlass \
  --max-running-requests 16 --cuda-graph-max-bs 16 \
  --max-context-length 16384 --port 8084
```

This command was tested on one H100 NVL reporting 95,830 MiB total memory.
The text checkpoint tensors occupy 64.56 GiB; SFLLM skips the 0.83 GiB vision
encoder and 1.57 GiB MTP tensors. The fused CUTLASS Polaris run peaked at 88.97 GiB, including
weights, FP32 recurrent states, KV cache, CUDA Graphs and runtime allocations. No CPU offload or
quantization was needed. Different workloads and higher concurrency need
separate memory checks.

### Without FlashInfer

SFLLM's existing GDN backend supports Triton for both prefill and decode. Dense
Qwen3.5/Qwen3.8 and Qwen3.5 MoE share this backend, which runs without FlashInfer.
Select the MoE and linear-attention backends with the flags below. Full attention
also supports Triton, so this configuration does not require `sglang-kernel`:

```bash
pip install -e .
python python/sfllm/serving/app.py \
  --model /mnt/data/work/Qwen3.5-35B-A3B \
  --dtype bfloat16 --attention-backend triton \
  --moe-runner-backend triton_kernel --linear-attn-backend triton \
  --max-running-requests 16 --cuda-graph-max-bs 16 \
  --max-context-length 16384 --port 8084
```

GDN prefill uses FLA's chunked Triton kernels (`fla-core`); decode uses the
existing SFLLM packed recurrent Triton kernel. BF16 weights/activations, FP32
recurrent states and decode CUDA Graphs are retained. Full **prefill** CUDA Graph
capture still requires FlashInfer GDN; it is not enabled by the command above.
The `sfkernels` extension remains an SFLLM dependency for existing model ops.
The FlashInfer default is unchanged when its dependencies are installed.

Migration tests require bit-identical MoE routing and outputs against the former
external package at 1, 16, 37 and 4103 tokens, in BF16 and FP16. An isolated
subprocess blocks `flashinfer`, `sglang` and all modules supplied by the optional
kernel distribution, and executes MoE, GDN prefill/decode, and changing-input GDN decode
CUDA Graph replay. Run it with:

```bash
python -m pytest tests/test_optional_triton_backends.py
```

Before source extraction, on 2026-09-14, the full official model passed all nine smoke questions with
those optional packages blocked in the server and its spawned workers. Both
GDN phases and the local MoE executed, and decode graphs were captured and used.
The unchanged FlashInfer configuration then completed all 500 original Polaris
requests at **781.90 output tokens/s**, versus the prior **783.49** (-0.20%);
mean TPOT was 19.517 versus 19.504 ms. This single regression run shows no material
performance change; generated output lengths differed (112,481 versus 113,451).
The FlashInfer routing, shared-expert packing and CUTLASS execution routines
were unchanged in that earlier run. Source hashes, commands,
numerical checks and regression results are recorded in
[`qwen35_triton_backend_validation.json`](../benchmark/qwen35_triton_backend_validation.json).

## Chat

```bash
curl http://localhost:8084/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3.5-35B-A3B",
    "messages": [{"role": "user", "content": "计算37乘41，只输出整数答案。"}],
    "temperature": 0,
    "max_tokens": 64,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

Set `enable_thinking` to `true` and allow a larger output budget for thinking
mode. The checkpoint's own chat template controls the prompt. Generated thinking
remains in the text response.

## Scope and validation

- BF16 text prefill, decode and decode CUDA Graph replay are validated on the
  official checkpoint. Both MoE backends also have FP16 numerical tests.
- Vision inputs and native MTP are outside this text path. Quantized checkpoints
  and configs with dense `mlp_only_layers` are rejected. The Triton backend allows shared and routed
  experts to have different intermediate widths; fused CUTLASS requires equal widths. Contexts beyond 16K and
  speculative decoding have not been evaluated for this model.
- Nine short greedy smoke questions passed, covering arithmetic, Chinese factual
  recall, Python, logic, and one thinking response. Nonthinking requests ran at
  concurrency four. This is a functionality check, not a model-quality benchmark.
- Component tests compare the routed kernels with independent per-expert GEMMs,
  and the Triton router/shared-expert block with Transformers. The CUTLASS tests
  compare its packed router, routed experts and shared expert against independent
  PyTorch selection and GEMMs. Tests also cover
  empty batches, partial tiles, packed weight loading, shared-weight completeness,
  routing ties, exclusion of router alignment padding, concurrent streams and
  changing expert assignments during
  CUDA Graph replay. Checkpoint-sized tests cover decode graphs and a
  4103-token prefill, with relative L2 error below
  1% against independent BF16 expert GEMMs.

Run the focused tests with SFLLM and its development dependencies installed:

```bash
python -m pytest tests/test_qwen3_5_moe.py tests/test_qwen3_8.py
```

Validation used checkpoint revision
`59d61f3ce65a6d9863b86d2e96597125219dc754` and Transformers 5.12.1.

### Answer accuracy

#### AIME 2025: comparison with SGLang

All 30 AIME 2025 problems were evaluated with thinking enabled, one greedy
completion per problem and a 32,768-token output budget. Both engines received
identical saved input token IDs, including the same boxed-answer instruction.
BF16 weights/activations and FP32 recurrent states were retained. The previously
measured CUTLASS serving configurations kept overlap and decode CUDA Graphs
enabled; context length was increased to 65,536 for this accuracy run. Requests
were submitted in groups of 16 and 14, with a cache flush before each group.

| Metric | SFLLM aligned CUTLASS | Tuned SGLang |
| --- | ---: | ---: |
| Correct final answers | 27 / 30 (90.0%) | 23 / 30 (76.7%) |
| Output-limit hits, counted as incorrect | 2 | 6 |
| Completed but incorrect final answers | 1 | 1 |

All 23 problems answered correctly by SGLang were also correct in SFLLM.
SFLLM alone answered dataset IDs 11, 12, 14 and 24 correctly; both failed IDs
13, 27 and 29. The three SFLLM failures were two unfinished thinking outputs
at the token cap and one incorrect boxed answer. Raw failure outputs were
checked against the answer extractor.

This sample showed no accuracy degradation relative to SGLang. It is a small,
capped diagnostic comparison, not evidence of a general accuracy improvement
or equivalence: the exact paired McNemar two-sided p-value is 0.125. The engines
produced no identical full token sequences, which does not affect final-answer
scoring. This run does not replace the separate Polaris serving benchmark.

Dataset: `math-ai/aime25`, revision
`563bb8404243c5f09de6ec262f2db674fe5bce9b`, all 30 test records. Commands,
input/source hashes, per-problem outcomes, final answers and raw-artifact hashes
are recorded in
[`benchmark/qwen35_moe_aime25_accuracy.json`](../benchmark/qwen35_moe_aime25_accuracy.json).
Raw outputs and evaluation scripts use the flat `/tmp/qwen35-moe-aime25-*` prefix.

#### GSM8K: optimization regression audit

The precision audit uses 100 GSM8K official-test records sampled with seed 42;
the first 20 are also evaluated with thinking enabled. Both runs use greedy
decoding, fixed batches of 16, decode CUDA Graphs, overlap disabled and identical
chat templates. Output limits are 1024 tokens without thinking and 4096 with
thinking. Truncated and unparsed outputs count as incorrect; official reference
answers are not edited.

| Implementation | No thinking | Thinking | Output-limit hits (off / on) |
| --- | ---: | ---: | ---: |
| Before router alignment (762.44 tokens/s version) | 92 / 100 | 15 / 20 | 6 / 4 |
| Aligned router (783.49 tokens/s version) | 94 / 100 | 17 / 20 | 3 / 2 |
| Experimental Triton tile tuning, not retained | 95 / 100 | 16 / 20 | 1 / 3 |
| Experimental CUDA top-k, rejected | 91 / 100 | 18 / 20 | 6 / 1 |

The aligned version did not turn any previously correct answer into an
incorrect answer in this sample. This small, capped evaluation does not establish
equivalent quality on all tasks. Polaris contains prompts only, so its generated
text agreement is not an answer-accuracy score.

Experimental Triton tile tuning gained one correct non-thinking answer and lost one
thinking answer on net. In the thinking subset, one previously truncated answer
became correct and two previously correct answers hit the output cap. Generated
token sequences are not identical, and these small capped samples cannot
establish a broad accuracy regression or improvement. The routing regression
tests required identical expert IDs and rounded weights against the original
library wrapper; that check is separate from end-to-end answer accuracy. The
experiment was not retained, as its full serving gain was negligible.

The CUDA top-k experiment retained BF16 weights and FP32 states, but changed
the softmax implementation. Three previously correct non-thinking answers then
hit the output limit. It was rejected before the serving benchmark. Kernel-level
agreement on a few random inputs was not treated as sufficient evidence.

Using real checkpoint weights and prefill activations from layers 0, 10, 20 and
39, independent BF16 expert GEMMs gave relative L2 errors of 0.36–0.57% with
identical routing. This checks expert computation separately from changes in
router selection. No FP8, INT8 or INT4 quantization was introduced. Full sample
indices, scores, answer transitions and artifact hashes are recorded in
[`benchmark/qwen35_moe_accuracy.json`](../benchmark/qwen35_moe_accuracy.json).

## Serving benchmark

Measured on 2026-09-14 on one H100 NVL: BF16 weights, FP32 recurrent states,
FA3, FlashInfer GDN and decode CUDA Graphs, without speculative decoding.
SFLLM used the server command above and the tuned fused CUTLASS implementation.
SGLang used the fastest measured configuration described below.

Polaris used all 500 records from `dashv4_polaris_test_data.jsonl`, preserving
the raw preformatted prompts, chat delimiters and literal backslash-n sequences.
Input length averaged 4486.1 tokens, with a maximum of 14,187. Both engines used
concurrency 16, unlimited request rate, greedy streaming generation, a 1000-token
output cap, EOS and stop token 248044. Four warmup requests preceded the timed
run and cache reset. Dataset SHA-256:
`bf1f5dde36b28e3fd337fb44630201b50906eedef0d98c06dd97be81eef95cc2`.

| Metric | SFLLM aligned CUTLASS (default) | Before router alignment | SFLLM `triton_kernel` | Tuned SGLang |
| --- | ---: | ---: | ---: | ---: |
| Successful requests | 500 / 500 | 500 / 500 | 500 / 500 | 500 / 500 |
| Total duration (s) | 144.80 | 151.99 | 157.76 | 161.41 |
| Output tokens/s | 783.49 | 762.44 | 717.13 | 713.56 |
| Mean TTFT (ms) | 180.77 | 203.92 | 206.37 | 190.41 |
| P99 TTFT (ms) | 1034.37 | 1597.74 | 1712.25 | 1001.43 |
| Mean TPOT (ms) | 19.50 | 19.91 | 21.23 | 21.42 |
| Mean request latency (s) | 4.58 | 4.80 | 4.98 | 5.09 |
| Output tokens | 113,451 | 115,881 | 113,135 | 115,173 |
| Requests reaching output cap | 1 | 4 | 1 | 2 |
| Peak GPU memory (GiB, 1 Hz) | 88.97 | 88.47 | 86.91 | 87.43 |

Aligned CUTLASS output throughput was **9.80% higher than SGLang** and
**2.76% higher than the preceding CUTLASS run**. This does not establish the
requested 20–30% advantage. All 500 input lengths and the dataset checksum
matched exactly. Output lengths differ across backends, so these are end-to-end
greedy-serving measurements, not fixed-work decode tests or answer-accuracy
measurements. Each configuration has one full run.

With fixed input tensors, the aligned router reduced CUDA Graph time from
0.0204 to 0.0157 ms at 16 tokens and from 0.1161 to 0.0227 ms at 4096 tokens.
Both removing the unaligned GEMM and avoiding the routing-buffer copy are
included in that comparison. The serving run also reduced mean TPOT from
19.91 to 19.50 ms and mean TTFT from 203.92 to 180.77 ms.

Before the aligned run's validation, an untimed 16-request probe captured one
full-model decode step. Its temporary profiling wrapper restored the original
forward method before smoke tests, serving warmups and the benchmark cache
reset. The recorded peak memory includes allocations retained from that probe.
The normal launch command above runs without this profiling wrapper; the exact
profiling launch is also preserved in the result record.

The durable [result record](../benchmark/qwen35_moe_h100_nvl_results.json) includes
commands, source hashes, input verification, all four results and historical
source hashes for both earlier SFLLM runs. Validation passed 54 focused tests and all 9
smoke questions. The earlier removed CUTLASS implementation measured 747.32
tokens/s; its result is retained separately in the record. The first unbucketed
Triton integration trial was aborted due to repeated routing-kernel compilation
and is excluded from performance comparisons.

SFLLM's 16 selected CUTLASS GEMM tactics are saved in
[`benchmark/qwen35_sfllm_cutlass_h100_nvl.json`](../benchmark/qwen35_sfllm_cutlass_h100_nvl.json).
For the recorded H100 NVL and software versions, restore them before startup:

```bash
mkdir -p /root/.cache/flashinfer
cp benchmark/qwen35_sfllm_cutlass_h100_nvl.json \
  /root/.cache/flashinfer/sfllm-moe-0.6.18-sm90-torch.bfloat16-257-2048-512-9.json
```

Startup automatically tunes and saves this cache when it is absent. Tuning is
outside the timed benchmark. Decode and prefill buckets are 1, 8, 16, 64, 256,
1024, 4096 and 8192 tokens; larger prefills use the last bucket's tactics.

### Rejected routing experiments

Replacing Triton top-k with `sgl_kernel.topk_softmax` reduced small-batch routing
time, but the capped GSM8K non-thinking score fell from 94/100 to 91/100. This
candidate was removed before a full serving benchmark.

A second experiment kept the external Triton primitive and tuned its tiles:
one row by 256 experts up to 32 tokens, and four rows by 256 experts up to 1024
tokens, with four warps. Larger prefills kept the original tile. BF16 and FP16
kernel comparisons produced identical expert IDs and weights on all tested
inputs, and 67 candidate regression tests passed. Routing CUDA Graph time at
16 tokens fell from 15.7 to 10.6 microseconds.

The complete 500-prompt Polaris run nevertheless measured only **784.70 vs
783.49 output tokens/s (+0.16%)**. Mean TPOT slightly increased from 19.504 to
19.518 ms; mean TTFT decreased from 180.77 to 175.46 ms. It generated 111,047
tokens in 141.51 seconds, with no output-limit hits. Its command, input hashes,
source snapshots and full measurements are preserved in the result record.
This single run did not establish a useful throughput improvement, so the tile
adapter and its experimental tests were removed. All retained implementation
source hashes match the 783.49-token/s version. Its 54 focused tests were rerun
after restoration.

Validation used PyTorch 2.13.0+cu130, Triton 3.7.1, FlashInfer 0.6.18,
sglang-kernel 0.4.6.post1 and Transformers 5.12.1.

## SGLang configuration and tuning

The same benchmark was run against the local SGLang checkout at
`95f5ecd3d26665423d3e6577a2a00c04f5cde733` (installed distribution reports
`0.5.19.dev898+g02d9b3060`). It used the same H100 NVL, checkpoint, BF16 weights,
FP32 recurrent states, FA3, FlashInfer GDN, concurrency 16 and generation
settings. Speculative decoding was disabled. The checkout contained local
changes in speculative-decoding files; these were retained.

Configuration selection used 96 unchanged Polaris prompts at evenly spaced
dataset indices, totaling 409,895 input tokens. Each trial used the same four
warmup requests, cache reset and stopping conditions as the full benchmark.
The following numbers are configuration trials, not the 500-request comparison:

| SGLang configuration | Output tokens/s |
| --- | ---: |
| Auto MoE (Triton), 4096-token prefill, no overlap | 598.95 |
| CUTLASS, decode-only tuning, 8192-token prefill, overlap | 525.70 |
| CUTLASS, decode and prefill tuning, 8192-token prefill, overlap | 686.97 |
| CUTLASS, decode and prefill tuning, 4096-token prefill, overlap | 684.36 |
| `triton_kernel`, 8192-token prefill, overlap | 662.87 |
| Tuned CUTLASS, 8192-token prefill, no_buffer, no overlap | 632.37 |

SGLang's automatic MoE selection lacked a tuned Triton configuration for this
GPU and expert geometry. Its CUTLASS startup tuning covered decode shapes only:
the optional extend tuning pass skipped this multimodal architecture even with
`--language-only`. Before measuring the tuned variants, FlashInfer's autotuner
extended SGLang's tactic cache through 8192 tokens using its exact expert geometry
(256 experts, top-k 8, hidden size 2048, intermediate size 512, BF16). This added
32 GEMM configurations while preserving the existing decode tactics. The cache
was loaded before graph capture; tuning time is excluded from serving results.

The fastest measured configuration and all six trial results are saved in
[`benchmark/qwen35_sglang_h100_nvl.json`](../benchmark/qwen35_sglang_h100_nvl.json).
The 42 CUTLASS tactics, including 32 added prefill entries, are preserved in
[`benchmark/qwen35_sglang_cutlass_h100_nvl.json`](../benchmark/qwen35_sglang_cutlass_h100_nvl.json).
Reuse these records for this H100 NVL, model, software stack and workload;
repeat tuning only when those conditions change. This is the fastest measured
configuration among the six trials, not a claim of a global optimum.

Run from the SFLLM checkout. Restore the saved cache to reproduce the prefill tactics; the exact
cache path below is specific to the recorded model path and software versions.
SGLang logs the cache path at startup. If that path changes, check the model and
software settings before reusing tactics. No benchmark rerun is needed to restore
this configuration.

```bash
mkdir -p /root/.cache/sglang/flashinfer/autotune/0.6.18/sm90/8b3756ddadd38e79
cp benchmark/qwen35_sglang_cutlass_h100_nvl.json \
  /root/.cache/sglang/flashinfer/autotune/0.6.18/sm90/8b3756ddadd38e79/rank_tp0_pp0_dp0.json
FLASHINFER_WORKSPACE_BASE=/root \
SGLANG_FLASHINFER_AUTOTUNE_EXTEND=1 \
/workspace/sglang/.venv/bin/python -m sglang.launch_server \
  --model-path /mnt/data/work/Qwen3.5-35B-A3B \
  --dtype bfloat16 --language-only --attention-backend fa3 \
  --linear-attn-backend triton \
  --linear-attn-decode-backend flashinfer \
  --linear-attn-prefill-backend flashinfer \
  --mamba-radix-cache-strategy extra_buffer --mamba-ssm-dtype float32 \
  --moe-runner-backend flashinfer_cutlass \
  --max-running-requests 16 --cuda-graph-max-bs-decode 16 \
  --context-length 16384 --chunked-prefill-size 8192 \
  --mem-fraction-static 0.9 --random-seed 1 --host 127.0.0.1 --port 8084
```

The base linear-attention setting permits the extra-buffer cache strategy;
both actual GDN modes use the explicit FlashInfer overrides. Radix caching,
scheduler overlap, prefill CUDA Graphs and decode CUDA Graphs were enabled.
SGLang's memory fraction includes model weights, whereas SFLLM's fraction
applies after model loading; the fractions therefore have different meanings.

The durable JSON record includes the complete command, environment, model/source
revisions, dataset hash, generation settings and measured results. SGLang
`--language-only` skips vision weights in this checkout but still constructs a
vision tower; approximately 0.83 GiB remains allocated.

The local detailed run artifacts are `/tmp/qwen35-moe-sglang-comparison.json`,
`/tmp/qwen35-moe-sglang-bench-metadata.json`,
`/tmp/qwen35-moe-sglang-bench.jsonl` and
`/tmp/qwen35-moe-sglang-cutlass-autotune.json`. The temporary
`/tmp/qwen35-moe-sglang-cutlass-autotune.py` generated the extra prefill tactics
using FlashInfer’s public API; it is not needed when restoring the saved cache.

## TensorRT-LLM MegaMoE source review

Reviewed `/mnt/data/jicwen/work/TensorRT-LLM` at commit
`75ca0821c08fee3212a5455615d4557df2e18f88` on 2026-09-14. This is a source
review; no TensorRT-LLM kernel was copied, ported or benchmarked in this step.

| Implementation | Source capability gate | Relevance to H100 BF16 |
| --- | --- | --- |
| `MegaMoECuteDsl` | SM100/SM103, NVFP4 only | Requires a different GPU and quantized computation; no direct BF16 path |
| `MegaMoEDeepGemm` | SM100/SM103, W4A8 MXFP4/MXFP8 only | No direct BF16 path |
| `DenseGEMMFusedMoE` | SM100/SM103, NVFP4 | Computes all routed experts densely; its documented TP8 sweet spot is 64–208 tokens, outside the current decode-16 workload |
| `CutlassFusedMoE` | Unquantized FP16/BF16 on SM80+ | Applicable family; SFLLM already uses FlashInfer's TensorRT-LLM-derived CUTLASS implementation |

The MegaMoE implementation separates the local two-GEMM pipeline from optional
cross-GPU token communication:

- `tensorrt_llm/_torch/cute_dsl_kernels/mega_moe_nvfp4/kernel_fc12.py`
  defines `Sm100SwapABSwigluFp4Fc12Kernel`. FC1, gated activation and FC2 share
  a persistent launch, with separate warp roles for tensor loads, MMA,
  scheduling and epilogues. It uses Blackwell `tcgen05` block-scaled MMA and
  tensor memory; changing a dtype or the architecture gate cannot port it to
  Hopper's BF16 WGMMA pipeline.
- `fc1_fc2_fuse_sched.py`, `MoEFusedFc12PersistentTileScheduler`, schedules
  `(group, phase, expert, token_block, output_block)`. Within an expert group,
  FC1 tiles precede FC2 tiles. Completion counters let FC2 wait for its own
  FC1 token block inside the GPU launch. FC1 intermediates still pass through
  global workspace; fusion does not remove every intermediate memory access.
- `megamoe_kernel.py` adds dispatch warps and token-return logic around that
  local pipeline. Single-rank operation uses local buffers; cross-rank
  communication overlaps peer loads/stores with compute. The multi-GPU
  communication savings do not apply to this single-H100 deployment.
- The default backend sets `in_kernel_fc2_reduce=False` in
  `tensorrt_llm/_torch/moe/fused_moe/mega_moe/mega_moe_cute_dsl.py`.
  `megamoe_kernel.py` then launches a separate deterministic `TopkReduce`
  after the main kernel. Router projection, top-k and activation quantization
  are also outside the persistent compute kernel. One Python op does not
  imply that the entire MoE layer is one GPU launch.

For this checkpoint, a 16-token step has 128 routed token/expert assignments
across 256 experts, plus 16 shared-expert assignments. The useful next kernel
question is whether a Hopper BF16 persistent FC1/activation/FC2 pipeline can
reduce launch, metadata and tile-scheduling costs on these sparse real routes.
That would require Hopper-specific implementation work, dependency-ordering
and occupancy checks, and preserving the existing BF16 rounding and routing
semantics. No speedup has been established for that proposal.

The lower-cost prerequisite remains measuring grouped-GEMM tactics and tile
waste on real decode routing distributions across layers. The previous routing
experiments did not change GEMM tactics and do not settle this question.
The existing BF16 CUTLASS path remains the baseline while this is investigated.
