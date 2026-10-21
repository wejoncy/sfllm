# SFLLM: High-Performance LLM Serving Framework

[🇨🇳 中文文档](./README_CN.md) | [🇺🇸 English](./README.md)

A production-ready, high-performance serving framework for large language models with OpenAI-compatible APIs.

## Performance

![SFLLM and SGLang: throughput and generation latency](docs/assets/internal-dataset-dspark-comparison.png)

DSpark on one H100 NVL, using the same 300 prompts randomly sampled from an internal dataset
(seed 42), an FP8 target, BF16 activations, and FP32 recurrent states.
Client concurrency ranges from 1 to 64. Each point uses the fastest configuration
among the candidates tested for that framework and concurrency; each formal
measurement runs once on all 300 prompts. Measured revisions: SFLLM
[`1755593`](https://github.com/wejoncy/sfllm/commit/17555935c3237ad1cf4afdd756943ae3f503dfa1)
(`qwen35_specfic`) and SGLang
[`e262b44`](https://github.com/sgl-project/sglang/commit/e262b44eef1f2bf83e34345276e2bb91a086587e).

SFLLM uses selective draft FP8/top-k16; SGLang uses native FlashInfer draft attention
with its selected draft configuration.

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


## Supported Models

| Model | Support |
| --- | --- |
| Llama / Phi-3 / InternLM3 | Text generation |
| Qwen2 / Qwen2.5 | Text generation |
| Qwen3 (dense) | Text generation |
| Qwen3.5 (dense) | Text generation; BF16 and FP8 (ModelOpt / compressed-tensors) |
| Qwen3.5-35B-A3B | BF16 text generation (MoE) |
| Qwen3.8-27B | BF16 text generation |

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

Optional GPU backends: `pip install -e '.[flashinfer]'`.

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
