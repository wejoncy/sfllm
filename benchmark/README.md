# ShareGPT benchmark

Hardware: **NVIDIA H100 NVL, BF16**. ShareGPT, server concurrency 32, client concurrency 24.

| Successful requests | Input tokens | Output tokens | Duration | Output throughput |
| ---: | ---: | ---: | ---: | ---: |
| 1000/1000 | 318819 | 914531 | 218.924 s | **4177.39 tok/s** |

Measured on `54300bc`, 2026-09-07. Mean TTFT: 108.224 ms; mean TPOT: 5.542 ms.

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

Wait for `Application startup complete` and a successful health check, then run the client in the second terminal:

```bash
curl --fail http://127.0.0.1:8081/health
python -m sfllm.serving.sgl_bench_seving \
  --backend sglang-native --host 127.0.0.1 --port 8081 \
  --model "$BENCH_MODEL" --tokenizer "$BENCH_MODEL" \
  --dataset-name sharegpt --dataset-path "$BENCH_DATASET" \
  --num-prompts 1000 --max-concurrency 24 \
  --sharegpt-context-len 8192 --sharegpt-output-len 1024 \
  --seed 1 --warmup-requests 1 --disable-ignore-eos \
  --output-details --output-file /tmp/sharegpt-1000.jsonl
```

The client uses streaming and temperature 0, respects EOS, and excludes the warmup request from timing. Wait for the final summary; detailed results are saved in the JSONL file.
