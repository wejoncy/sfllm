# Serving benchmark

Run from the repository root. Set these paths in both terminals:

```bash
export PYTHONPATH="$PWD/python:$PWD/sfkernels/python"
export BENCH_MODEL=/path/to/Qwen3.5-4B
export BENCH_DATASET=/path/to/ShareGPT_V3_unfiltered_cleaned_split.json
```

Start SFLLM (server concurrency 32, FA3, FlashInfer GDN):

```bash
python python/sfllm/serving/app.py \
  --model-path "$BENCH_MODEL" --port 8092 --dtype bfloat16 \
  --max-running-requests 32 --cuda-graph-max-bs 32 \
  --attention-backend fa3 \
  --linear-attn-prefill-backend flashinfer --linear-attn-decode-backend flashinfer \
  --mem-fraction 0.7 --max-context-length 8192
```

After `Application startup complete`, run ShareGPT with 1000 requests and client concurrency 24:

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

`sglang-native` selects the HTTP protocol used by SFLLM. EOS is respected; CUDA Graph and overlap are enabled by default. FP8 is detected from the model checkpoint.

For speculative decoding, accept len is `(successful output tokens - one prefill token per successful request) / total decode rounds`. After warmup, the SFLLM benchmark calls `POST /flush_cache` to clear request caches and metrics before timing.
