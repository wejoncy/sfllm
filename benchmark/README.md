# ShareGPT benchmark

Hardware: **NVIDIA H100 NVL, BF16**. ShareGPT, 1000/1000 successful requests per run, server concurrency 32, client concurrency 24.

| Model | Input tokens | Output tokens | Duration | Output tok/s | Acclen |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-4B | 318819 | 910614 | 209.628 s | 4343.96 | — |
| Qwen3-4B | 308695 | 1023588 | 225.953 s | 4530.08 | — |
| Qwen3-4B + EAGLE3 | 308695 | 1024000 | 341.638 s | 2997.32 | 2.93 |
| Qwen3-4B + DFlash2 | 308695 | 1024000 | 114.030 s | **8980.06** | 4.04 |

Measured 2026-09-07. Acclen is the median of logged ShareGPT windows, including the target token. Both speculative runs returned exactly 1024 tokens per request.

GSM8K (1319 questions, 5-shot, greedy, max 512 tokens): Qwen3.5-4B **86.81%**, Qwen3-4B **87.04%**, EAGLE3 **86.88%**, DFlash2 **87.26%**.

Sources: non-speculative `3a1f5d2`; EAGLE3 `3927e5f`; DFlash2 `3927e5f` plus the local checkpoint-config and FA3 context-window compatibility changes.

Set the model and dataset paths in both terminals, from the repository root. Use a Python environment with sfllm and its CUDA kernels installed.

```bash
export BENCH_MODEL=/path/to/Qwen3.5-4B
export BENCH_DATASET=/path/to/ShareGPT_V3_unfiltered_cleaned_split.json
export PYTHONPATH="$PWD/python:$PWD/sfkernels/python"
```

Start the server:

```bash
python python/sfllm/serving/app.py \
  --model-path "$BENCH_MODEL" --port 8081 --dtype bfloat16 \
  --max-running-requests 32 --cuda-graph-max-bs 32 \
  --attention-backend fa3 --linear-attn-backend flashinfer --mem-fraction 0.7
```

For DFlash2, set `BENCH_MODEL` to Qwen3-4B and append `--speculative-algorithm dflash2 --speculative-draft-model-path /path/to/Qwen3-4B-speculator.dflash2` (`mgoin/Qwen3-4B-speculator.dflash2`, revision `e3e7a18`). Its config selects block size 8 and a non-causal draft block with 2048 context tokens. Ensure `ninja` is on `PATH`.

For EAGLE3, use Qwen3-4B, replace `--attention-backend fa3` with `--attention-backend triton`, and append `--speculative-algorithm eagle3 --speculative-draft-model-path /path/to/Qwen3-4B_eagle3 --speculative-num-steps 4 --speculative-eagle-topk 4 --speculative-num-draft-tokens 8`.

Wait for `Application startup complete` and a successful health check, then run the client in the second terminal:

```bash
curl --fail http://127.0.0.1:8081/health
python benchmark/bench_serving.py \
  --backend sglang-native --host 127.0.0.1 --port 8081 \
  --model "$BENCH_MODEL" --tokenizer "$BENCH_MODEL" \
  --dataset-name sharegpt --dataset-path "$BENCH_DATASET" \
  --num-prompts 1000 --max-concurrency 24 \
  --sharegpt-context-len 8192 --sharegpt-output-len 1024 \
  --seed 1 --warmup-requests 1 --disable-ignore-eos \
  --output-details --output-file /tmp/sharegpt-1000.jsonl
```

The client uses streaming and temperature 0, respects EOS, and excludes the warmup request from timing. Wait for the final summary; detailed results are saved in the JSONL file.
