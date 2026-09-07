# ShareGPT serving benchmark

Reproduce the Qwen3.5-4B benchmark with 1000 prompts, server concurrency 32, and client concurrency 24.

## Recorded environment and workload

Run Qwen3.5-4B with the following environment and workload.

| Setting | Value |
| --- | --- |
| GPU | 1 × NVIDIA H100 NVL, 93.09 GiB visible memory |
| Python / PyTorch / CUDA runtime | 3.12.3 / 2.13.0 / 13.0 |
| Triton / Transformers / FlashInfer | 3.7.1 / 5.12.1 / 0.6.18 |
| Model and tokenizer | Qwen3.5-4B, local path `/mnt/data/jicwen/work/Qwen3.5-4B` |
| Precision / memory fraction | BF16 / 0.7 |
| Full attention / GDN prefill / GDN decode | FA3 / FlashInfer / FlashInfer |
| Server / client maximum concurrency | 32 / 24 |
| CUDA graphs / CPU–GPU overlap | Enabled; graph maximum batch size 32 / enabled |
| Server context limit | 8192 tokens |
| Dataset | `ShareGPT_V3_unfiltered_cleaned_split.json` |
| Measured requests / warmup requests | 1000 / 1 (warmup excluded from timing) |
| Seed / request rate | 1 / unlimited (`inf`, client default) |
| Output token cap / dataset context filter | 1024 / prompt tokens + 1024 ≤ 8192 |
| Sampling / EOS | Temperature 0; EOS respected (`--disable-ignore-eos`) |
| API / streaming / chat template | Native `/generate` / enabled / disabled |

ShareGPT sampling uses the first prompt/response pair from conversations with at least two turns, shuffles with seed 1, and selects 1000 entries after length filtering. Inputs are plain prompts. The 1024-token setting is a cap; EOS can stop generation earlier.

Dataset SHA256: `35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4` (672837942 bytes).
Model `config.json` SHA256: `ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670`.
Tokenizer `tokenizer_config.json` SHA256: `316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8`.

## Prepare the model and dataset

Install sfllm and its CUDA kernels first; see the [repository installation instructions](../README.md#installation). The commands below use the benchmark machine's paths and Python environment. Substitute these paths for another machine. If the model or dataset is absent, download it before starting the server:

```bash
hf download Qwen/Qwen3.5-4B --local-dir /mnt/data/jicwen/work/Qwen3.5-4B
curl -fL https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json \
  -o /mnt/data/jicwen/work/ShareGPT_V3_unfiltered_cleaned_split.json
sha256sum /mnt/data/jicwen/work/ShareGPT_V3_unfiltered_cleaned_split.json
```

## Start the server

Run in the first terminal. These are the recorded server arguments:

```bash
cd /workspace/sfllm
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/workspace/sfllm/python:/workspace/sfllm/sfkernels/python
/workspace/sglang/.venv/bin/python python/sfllm/serving/app.py \
  --model-path /mnt/data/jicwen/work/Qwen3.5-4B \
  --port 8081 \
  --dtype bfloat16 \
  --max-running-requests 32 \
  --cuda-graph-max-bs 32 \
  --attention-backend fa3 \
  --linear-attn-backend flashinfer \
  --mem-fraction 0.7
```

Wait for weight loading, CUDA graph capture, and `Application startup complete` before sending requests.

## Run the client

In a second terminal, wait for the health endpoint and then run the recorded client command. Create the result directory, or replace `--output-file` with a new result path. The client appends one record per run.

```bash
cd /workspace/sfllm
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/workspace/sfllm/python:/workspace/sfllm/sfkernels/python
until curl --fail --silent --max-time 2 http://127.0.0.1:8081/health >/dev/null; do
  sleep 1
done
mkdir -p /tmp/sfllm-pr1-sharegpt.0tfmxqsv
/workspace/sglang/.venv/bin/python -m sfllm.serving.sgl_bench_seving \
  --backend sglang-native \
  --host 127.0.0.1 \
  --port 8081 \
  --model /mnt/data/jicwen/work/Qwen3.5-4B \
  --tokenizer /mnt/data/jicwen/work/Qwen3.5-4B \
  --dataset-name sharegpt \
  --dataset-path /mnt/data/jicwen/work/ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 1000 \
  --max-concurrency 24 \
  --sharegpt-context-len 8192 \
  --seed 1 \
  --warmup-requests 1 \
  --output-details \
  --output-file /tmp/sfllm-pr1-sharegpt.0tfmxqsv/results.jsonl \
  --sharegpt-output-len 1024 \
  --disable-ignore-eos
```

The client completes one warmup request before timing the 1000 measured requests. Wait for its final benchmark summary and successful exit. `--output-details` keeps per-request results in the JSONL output.

