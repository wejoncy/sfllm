# Serving benchmarks

Measured 2026-09-08 on H100 NVL, SFLLM, BF16 activations outside FP8 linears, FA3 and overlap enabled. Qwen3.5 uses FlashInfer GDN prefill/decode. These are paired measurements; lower accuracy results remain visible and have not passed the no-regression check.

## ShareGPT

1000/1000 successful requests per run, server 32/client 24, seed 1, chat template enabled, EOS respected, output cap 1024. One warmup request is excluded from timing. The baseline is `53a01d7`; each candidate run records its source snapshot and hashes.

| Model | Output tok/s, before → after | QPS, before → after | Avg output, after | Range, after | Accept len, after | GSM8K correct/1319, before → after |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-4B | 4519.01 → 4508.79 | 4.7276 → 4.7200 | 955.255 | 78–1024 | — | 1148 → 1148 |
| Qwen3-4B + EAGLE3 | 5053.80 → 5098.28 | 5.3131 → 5.3585 | 951.444 | 78–1024 | 2.5002 | 1146 → 1147 |
| Qwen3-4B + DFlash2 | 7115.85 → 7114.18 | 7.4561 → 7.4551 | 954.269 | 78–1024 | 3.0914 | 1149 → 1147 |
| Qwen3.5-4B BF16 | 4369.77 → 4387.89 | 4.4441 → 4.4627 | 983.247 | 145–1024 | — | 1155 → 1157 |
| Qwen3.5-4B ModelOpt FP8 | 5039.70 → 5157.87 | 5.1282 → 5.2518 | 982.112 | 145–1024 | — | 1107 → 1095 |

The reversed-order Qwen3 ordinary pair measured 4474.18 → 4470.96 tok/s, with 213.595 → 213.574 seconds for 1000 requests. Both first and repeat runs are retained. The DFlash2 repeat measured 7118.85 → 7243.11 tok/s and 1149 → 1150 correct on GSM8K. All 35 initially changed DFlash2 outputs match token-for-token when replayed sequentially at concurrency 1.

The initial BF16 candidate scored 1149 vs 1155, and its control scored 1153 vs 1159. The fused prefill Q/K reduction changed floating-point summation order. Preserving that order restores exact preprocessing outputs; the repaired run above scores 1157 and reaches 4387.89 tok/s. Original results remain in the measurement records.

The earlier ModelOpt FP8 compatibility candidate preserved the original norm reduction, BF16 rounding, static-scale reciprocal and static GEMM accumulation. All 400 captured producer calls were byte-exact. The full sequential GSM8K comparison was token-exact on all 1319 questions, with 1106 correct on both sides. Replaying all 50 concurrent correctness-flipped questions in identical batches (up to 24, overlap enabled) also gave identical tokens and schedules. The unchanged baseline changed correctness on 18 of these questions between sequential and batched execution. Original concurrent scores (1107 → 1095) remain above.

The current FP8 norm uses FP32 `gl.sum`, with manual reductions and layout conversions removed. Its paired ShareGPT 1000 measurement is **5163.08 → 5286.89 tok/s (+2.40%)**. Full GSM8K at concurrency 24 scored **1123 → 1119 / 1319**; a separate full evaluation with fixed admission batches of 24 scored **1092 → 1119 / 1319**. Outputs are not token-exact, and the fixed-batch result does not replace the concurrent score. Throughput remains 1.73% below the historical 5379.74 tok/s. Records: `/tmp/sfllm-review20-20260908/static-order-cleanup` and `/tmp/sfllm-static-ablation-20260909`.

From the repository root, set paths in both terminals:

```bash
export PYTHONPATH="$PWD/python:$PWD/sfkernels/python"
export BENCH_MODEL=/path/to/Qwen3.5-4B
export BENCH_DATASET=/path/to/ShareGPT_V3_unfiltered_cleaned_split.json
```

Start the server:

```bash
python python/sfllm/serving/app.py \
  --model-path "$BENCH_MODEL" --port 8092 --dtype bfloat16 \
  --max-running-requests 32 --cuda-graph-max-bs 32 \
  --attention-backend fa3 \
  --linear-attn-prefill-backend flashinfer --linear-attn-decode-backend flashinfer \
  --mem-fraction 0.7 --max-context-length 8192
```

For ModelOpt FP8, use the exported checkpoint as `BENCH_MODEL`; quantization is detected automatically. For Qwen3, use Qwen3-4B and optionally append:

```bash
# EAGLE3: keep FA3 and 4/4/8.
--speculative-algorithm eagle3 \
  --speculative-draft-model-path /path/to/Qwen3-4B_eagle3 --speculative-num-steps 4 --speculative-eagle-topk 4 --speculative-num-draft-tokens 8

# DFlash2: mgoin/Qwen3-4B-speculator.dflash2, checkpoint block size 8.
--speculative-algorithm dflash2 \
  --speculative-draft-model-path /path/to/Qwen3-4B-speculator.dflash2
```

Wait for `Application startup complete` and a successful health check, then run:

```bash
curl --fail http://127.0.0.1:8092/health
python benchmark/bench_serving.py \
  --backend sglang-native --host 127.0.0.1 --port 8092 \
  --model "$BENCH_MODEL" --tokenizer "$BENCH_MODEL" \
  --dataset-name sharegpt --dataset-path "$BENCH_DATASET" \
  --num-prompts 1000 --max-concurrency 24 \
  --sharegpt-context-len 8192 --sharegpt-output-len 1024 \
  --seed 1 --warmup-requests 1 --disable-ignore-eos --apply-chat-template \
  --output-details --output-file /tmp/sharegpt-1000.jsonl
```

`sglang-native` names the HTTP protocol; these measurements run SFLLM. The saved measurement client also records output IDs and stream events. Accept len = successful output tokens / total decode rounds, excluding one prefill event per request. The standard client reports N/A when it does not collect these rounds.

## Polaris

500/500 successful requests, server 32/client 22, seed 42, temperature 0, stop token 248044, output cap 1000. DFlash2 block size is 5. The baseline is the saved pre-refactor snapshot, which already supports this compressed-tensors checkpoint; HEAD does not support it.

| Model | Output tok/s, before → after | QPS, before → after | Avg output, after | Range, after | Accept len, after | GSM8K correct/1319, before → after |
|---|---:|---:|---:|---:|---:|---:|
| Polaris target | 2729.83 → 2710.40 | 5.0184 → 5.0148 | 540.480 | 211–1000 | — | 1075 → 1067 |
| Polaris target + DFlash2 | 3406.50 → 3400.66 | 6.3299 → 6.3099 | 538.944 | 185–1000 | 3.0427 | 1074 → 1066 |

The pre-fix Polaris ordinary accuracy control scored 1071 → 1069; all 23 initially correctness-flipped outputs matched token-for-token at concurrency 1. The final table retains the later measurements after the shared Q/K summation fix. The first runs (2722.22/3404.44 tok/s, GSM8K 1064/1067 for ordinary/DFlash2) remain in the saved records.

Polaris has a reproducible Q/K summation-order difference: 53 of 74 correctness-flipped ordinary requests still differ in identical batches, including first-token differences. Applying only the same Q/K summation fix to the old snapshot makes all 74 ordinary and 84 DFlash2 diagnostic outputs token-exact, with identical schedules. This identifies the numerical change; it does not restore equivalence to the original Polaris baseline or replace its scores. Against the supplied SGLang reference (3056.47 tok/s, 5.6520 QPS), the final DFlash2 run is 11.26% higher in output throughput and 11.64% higher in QPS.

The earlier SFLLM result was 6.4773 QPS and 3487.06 tok/s under the same recorded server/client flags, dataset and client-script hashes. The final run is lower by 2.58% in QPS and 2.48% in output throughput; the intervening baseline in the table does not replace that earlier performance target.

After removing the Q/K layout conversions, a separate DFlash2 run measured **6.4101 QPS, 3452.71 tok/s**, 500/0 success/failure, 78.002 s, average output 538.636 tokens and accept len 3.0457. GSM8K returned to **1074/1319 (81.43%)**. All 74 ordinary and 84 DFlash2 fixed-batch diagnostic outputs match the original fused baseline token-for-token. QPS remains 1.04% below 6.4773; performance is only partially restored. These newer results are saved in `qk-warp-restored/`; the tables above retain their earlier measurements.

Set `DSPARK_ROOT` to `/mnt/data/work/dspark_repro_download/qwe35_dspark_qaa/dspark_repro`. Use the server command above with model `$DSPARK_ROOT/target`, `--mem-fraction 0.5`, and `--max-context-length 16384`. For DFlash2 append:

```bash
--speculative-algorithm dflash2 \
  --speculative-draft-model-path "$DSPARK_ROOT/sglang/dflash2_loss3_checkpoint2" --speculative-num-draft-tokens 5
```

After the health check, prewarm with the client below using `--num-prompts 22 --max-new-tokens 32 --output-dir /tmp/polaris-prewarm`, then run the timed dataset:

```bash
python "$DSPARK_ROOT/benchmark/bench_sglang_raw.py" \
  --url http://127.0.0.1:8092/generate \
  --input "$DSPARK_ROOT/data/dashv4_polaris_test_data.jsonl" \
  --output-dir /tmp/polaris-benchmark --concurrency 22 \
  --max-new-tokens 1000 --num-prompts 0 \
  --temperature 0 --seed 42 --stop-token-id 248044
```

## Accuracy and records

GSM8K uses all 1319 test questions, the first 5 training examples as demonstrations, greedy decoding, max 512 tokens and client concurrency 24. Prompts use `Question: … Answer:`; scoring extracts the last number. Run against the same server:

```bash
python /tmp/sfllm-pr1-serving-fix.2dldw1tz/gsm8k_client.py \
  --port 8092 --max-concurrency 24 --output-dir /tmp/gsm8k-results
```

Exact launch/client commands, source/data hashes, token IDs, per-question answers and comparison reports are saved in `/tmp/sfllm-review20-20260908`. All 28466 saved ShareGPT/GSM8K texts match complete tokenizer decoding; this checks detokenization, independently of answer accuracy. Original runs are retained alongside diagnostic repeats.
