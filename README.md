# SFLLM: High-Performance LLM Serving Framework

[🇨🇳 中文文档](./README_CN.md) | [🇺🇸 English](./README.md)

A production-ready, high-performance serving framework for large language models with OpenAI-compatible APIs.

## Project Background

SFLLM (Serving Framework for Large Language Models) is designed to provide efficient and scalable inference services for large language models. It focuses on maximizing GPU utilization and reducing inference latency through intelligent batching, CUDA optimizations, and memory-efficient implementations.

## Features

- **OpenAI-Compatible API**: Drop-in replacement for OpenAI API endpoints
- **High Performance**: Optimized inference with intelligent request batching
- **Streaming Support**: Real-time streaming responses for better user experience
- **CUDA Optimizations**: CUDA graphs and custom kernels for maximum performance
- **Memory Efficient**: Optimized KV-cache management and memory allocation
- **Production Ready**: Built-in health checks and error handling
- **Eagle3 Speculative Decoding**: Advanced speculative decoding with Eagle3 algorithm for faster generation
- **full graph mode for spec**: capture draft-verify-accept into one cuda graph, saved alot cpu overhead
- **Eagle3 with CUDA Graph**: Optimized Eagle3 implementation with CUDA graph acceleration
- **spec decoding with overlap** make the GPU are busy all the time
        ---from <img width="1529" height="332" alt="image" src="https://github.com/user-attachments/assets/13d18911-94b6-433c-8803-2c17c61db446" />
        to <img width="1509" height="481" alt="image" src="https://github.com/user-attachments/assets/a917fa16-cdd8-4ed5-a426-64aa7460edcc" />


## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- PyTorch 2.0+

### Install from Source

```bash
# Clone the repository
git clone https://github.com/wejoncy/sfllm.git
cd sfllm

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Quick Start

### 1. Start the Server

**Basic Usage:**
```bash
python python/sfllm/serving/app.py \
  --model /path/to/your/model \
  --port 8081 \
  --dtype float16
```

**With Eagle3 Speculative Decoding:**
```bash
python python/sfllm/serving/app.py \
  --model /path/to/your/model \
  --speculative-draft-model-path /path/to/eagle3/draft/model \
  --speculative-algorithm eagle3 \
  --speculative-num-steps 4 \
  --port 8081 \
  --dtype float16
```

**AngelSlim EAGLE3 config note:** Our tests favor `rope_theta=10000` in the
draft `config.json` for `AngelSlim/Qwen3-4B_eagle3` and
`AngelSlim/Qwen3-1.7B_eagle3`. Their configs declare `1000000`, but
[AngelSlim's published training code](https://github.com/Tencent/AngelSlim/blob/917de3ae8c3147ff158b1b7a9c40082808486ae9/angelslim/compressor/speculative/train/models/draft/llama_eagle3.py#L186-L190)
uses RoPE's default base of `10000`. Keep the target at `1000000`;
this correction is specific to these draft checkpoints.

ShareGPT acceptance length (`1000000` → `10000`): **4B: 2.3871 → 2.4971**
(1000 requests); **1.7B: 2.3170 → 2.4012** (32 requests). Both used BF16,
FA3, EAGLE3 4/4/8, server concurrency 32, client concurrency 22, and EOS enabled;
acceptance length excludes the prefill token. The tested 1.7B draft also needs
`tie_word_embeddings=false`: its 32000-row output head cannot share the
151936-row target embedding.

**Qwen3.8-27B (text):**

Qwen3.8-27B shares the dense Qwen3.5 implementation.
[`qwen3_8.py`](python/sfllm/models/qwen3_8.py) provides a separate registered
entry class; the official checkpoint loads via its declared
`Qwen3_5ForConditionalGeneration` architecture. Its GDN `swish`
output gate is SiLU; full attention continues to use sigmoid gating. SFLLM
loads the language model and skips the vision encoder and MTP weights.

```bash
hf download Qwen/Qwen3.8-27B --local-dir /mnt/data/work/Qwen3.8-27B
python python/sfllm/serving/app.py \
  --model /mnt/data/work/Qwen3.8-27B --dtype bfloat16 \
  --attention-backend fa3 --max-running-requests 8 --cuda-graph-max-bs 8 \
  --port 8081
```

This BF16 configuration fits an H100 96 GB. It uses FP32 recurrent states
and FlashInfer GDN by default. See [Qwen3.8 usage](docs/qwen3_8.md) for thinking
controls and the current support scope.

**Qwen3.5 FP8:**

Export a calibrated Hugging Face checkpoint with NVIDIA ModelOpt's
`FP8_DEFAULT_CFG` and `export_hf_checkpoint`, then serve it with:

```bash
python python/sfllm/serving/app.py \
  --model /path/to/Qwen3.5-4B-FP8 --dtype bfloat16 --quantization fp8 \
  --max-running-requests 32 --cuda-graph-max-bs 32 --port 8081
```

The exported quantization config is detected automatically, so `--quantization fp8`
is optional. Both `config.json` and legacy `hf_quant_config.json` metadata are
supported. ModelOpt's excluded layers retain BF16, including the GDN a/b projections;
FP8 KV-cache quantization is not supported.

Qwen3.5 `compressed-tensors` exports using `FP8_DYNAMIC` (per-channel weights,
per-token dynamic activations) are also detected automatically. Use the same
command with `--dtype bfloat16`, including for exports whose config declares float32.

For a compatible Qwen3.5 DFlash2 draft, add:

```bash
  --speculative-algorithm dflash2 \
  --speculative-draft-model-path /path/to/dflash2-checkpoint \
  --speculative-num-draft-tokens 5
```

The draft token count is configurable (2 to the checkpoint block size); omit it
to use the checkpoint default. The selector top-k stays as configured in the checkpoint.

For a Qwen3 DSpark draft (`DSparkDraftModel` or `Qwen3DSparkModel`), use:

```bash
  --attention-backend fa3 \
  --speculative-algorithm dspark \
  --speculative-draft-model-path /path/to/dspark-checkpoint \
  --speculative-num-draft-tokens 6
```

This example uses 6 draft queries (anchor + 5 masks) to propose 5 tokens from the mask
positions; the target verifies 6 tokens including the anchor. Omit the token count to use the checkpoint's gamma
plus one (`dspark_block_size` takes precedence over `block_size`). Qwen3 and Qwen3.5
targets, vanilla/gated/RNN Markov heads, and mixed sliding/full draft attention are
supported. Decoding is fixed-width by default and greedy; confidence-head weights are unused.
For vanilla Markov heads, `--speculative-dspark-topk 16` restricts draft proposals
to the top 16 unary candidates and computes their transitions together. This can
change draft acceptance; every emitted token is still verified by the target.
The default, `-1`, runs original DSpark and scores the full vocabulary.
For Qwen3.5, `--mamba-ssm-dtype bfloat16` explicitly selects BF16 recurrent
states. The default uses the model config's SSM dtype, or FP32 if unspecified.
GDN prefill and ordinary decode use their selected backends. BF16 speculative
verification always uses Triton and needs no backend setting.

Set `SFLLM_GDN_JOURNAL=1` to enable FP32 GDN speculative verification journals (off by default).
Use `--spec-adaptive-verify d8t5` for draft width 8 and a shared verify budget of `round(batch_size * 5)` tokens, both including the anchor (`d6t5` and `d8t4.6` also supported); this overrides `--speculative-num-draft-tokens`. Omit it for unchanged fixed-width verification. DSpark requires positive `--speculative-dspark-topk`.

### 2. Test the API

**Chat Completions (Streaming)**
```bash
curl -X POST "http://localhost:8081/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "stream": true,
    "max_new_tokens": 256,
    "temperature": 0.7
  }'
```

**Text Completions**
```bash
curl -X POST "http://localhost:8081/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model",
    "prompt": "The future of AI is",
    "max_new_tokens": 128,
    "temperature": 0.8
  }'
```

### 3. Health Check

```bash
curl http://localhost:8081/health
```

## Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | Path to model directory | Required |
| `--port` | Server port | 8081 |
| `--dtype` | Model precision (float16/float32) | float16 |
| `--max-context-length` | Maximum context length | 4096 |
| `--cuda-graph-max-bs` | Max CUDA graph batch size | 32 |
| `--disable-cuda-graph` | Disable CUDA graphs | False |
| `--enable-prefill-cuda-graph` | Capture full prefill graphs (Triton or FA3; GDN requires FlashInfer) | False |
| `--prefill-cuda-graph-sizes` | Total-token capacities for prefill graphs | Steps of 32 through 256, 64 through 512, 128 through 2048 |
| `--speculative-algorithm` | Speculative decoding algorithm (`eagle3`, `dflash2`, or `dspark`) | None |
| `--speculative-draft-model-path` | Path to the speculative draft model | None |
| `--speculative-num-steps` | Number of speculative steps | 4 |
| `--disable-overlap` | Disable overlap scheduling | False |

Prefill graphs select the smallest captured capacity that fits the batch's total
new tokens. Above 1024 tokens, replay allows at most 16 extra padding tokens;
larger gaps use eager execution. Up to 1024 tokens, the existing bucket spacing
controls padding. Each capacity supports different request counts and unequal
sequence lengths, up to `--max-running-requests`.
The 24 default capacities use finer spacing for small prefills: 129 tokens replay
the 160-token graph (+31), 1034 tokens use eager, and 1136 tokens replay the
1152-token graph (+16).
Use `--prefill-cuda-graph-sizes` to override capacities.
The default 2048-token ceiling limits graph use, not request or context length;
larger prefills use eager execution. For Qwen3, enable with
`--attention-backend triton --enable-prefill-cuda-graph`; FA3 is also supported.
Models with GDN, such as Qwen3.5, additionally require
`--linear-attn-prefill-backend flashinfer`. Backend and SSM precision settings
are preserved. Token padding can change GEMM/attention
rounding and generated tokens, so results are not guaranteed to be bit-identical
to eager execution.

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## Contributing

We welcome contributions! Please feel free to submit issues and pull requests.

---

**Made with ❤️ by [wejoncy](https://github.com/wejoncy)**
