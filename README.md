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
| `--speculative-algorithm` | Speculative decoding algorithm (`eagle3` or `dflash2`) | None |
| `--speculative-draft-model-path` | Path to the speculative draft model | None |
| `--speculative-num-steps` | Number of speculative steps | 4 |
| `--disable-overlap` | Disable overlap scheduling | False |

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## Contributing

We welcome contributions! Please feel free to submit issues and pull requests.

---

**Made with ❤️ by [wejoncy](https://github.com/wejoncy)**
