# SFLLM: High-Performance LLM Serving Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-green.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)

A production-ready, high-performance serving framework for large language models with OpenAI-compatible APIs. SFLLM provides enterprise-grade features including streaming responses, batched inference, CUDA graph optimization, and intelligent request scheduling.

## 🚀 Key Features

- **🔥 High Performance**: 3.5x throughput improvement through intelligent batching and CUDA optimizations
- **🌊 Streaming Support**: Real-time streaming responses compatible with OpenAI API
- **📊 Smart Batching**: Automatic request batching based on sequence length similarity
- **⚡ CUDA Graphs**: Optimized GPU kernel execution with automatic graph capture
- **🎯 OpenAI Compatible**: Drop-in replacement for OpenAI API endpoints
- **🔧 Production Ready**: Built-in health checks, monitoring, and error handling
- **🧠 Memory Efficient**: Optimized memory management and KV-cache handling

## 📈 Performance Benchmarks

| Configuration | Throughput (req/s) | Latency (ms) | GPU Utilization |
|---------------|-------------------|--------------|-----------------|
| Baseline      | 0.40              | 2,786        | 45%            |
| SFLLM         | 1.41              | 1,195        | 85%            |
| **Improvement** | **+252%**       | **-57%**     | **+89%**       |

*Benchmark: 40 concurrent requests, Qwen3-0.6B model, NVIDIA RTX 4090*

## 🛠️ Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/wejoncy/sfllm.git
cd sfllm

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Basic Usage

```bash
# Start the server
python python/sfllm/serving/app.py \
  --model /path/to/your/model \
  --port 8081 \
  --dtype float16
```

### API Examples

#### Chat Completions (Streaming)
```bash
curl -X POST "http://localhost:8081/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-0.6b",
    "messages": [
      {"role": "user", "content": "Explain quantum computing in simple terms"}
    ],
    "stream": true,
    "max_new_tokens": 512,
    "temperature": 0.7
  }'
```

#### Text Completions (Non-streaming)
```bash
curl -X POST "http://localhost:8081/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-0.6b",
    "prompt": "The future of artificial intelligence is",
    "stream": false,
    "max_new_tokens": 256,
    "temperature": 0.8
  }'
```

## 🏗️ Architecture

SFLLM follows a modular architecture designed for scalability and performance:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI       │    │   EngineServer   │    │ InferenceEngine │
│   Web Server    ├────┤   Process        ├────┤   Worker        │
│                 │    │   Manager        │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
    ┌─────────┐           ┌─────────────┐       ┌─────────────────┐
    │ Request │           │  Scheduler  │       │   ModelRunner   │
    │ Handler │           │  & Queue    │       │   + CUDA Graph  │
    └─────────┘           └─────────────┘       └─────────────────┘
```

### Core Components

- **FastAPI Server**: Handles HTTP requests, validates inputs, formats responses
- **EngineServer**: Manages multiprocessing communication and request lifecycle  
- **InferenceEngine**: Core inference logic with intelligent scheduling
- **ModelRunner**: CUDA-optimized model execution with graph capture
- **Scheduler**: Smart batching based on sequence length similarity

## 📚 Documentation

Detailed documentation is available in the [`docs/`](./docs/) directory:

- [🚀 Getting Started Guide](./docs/getting-started.md)
- [⚙️ Configuration Reference](./docs/configuration.md)
- [🏗️ Architecture Overview](./docs/architecture.md)
- [📊 Performance Tuning](./docs/performance-tuning.md)
- [🔌 API Reference](./docs/api-reference.md)
- [🐛 Troubleshooting](./docs/troubleshooting.md)

## ⚡ Performance Optimizations

### 1. Intelligent Request Batching
- Groups requests by similar sequence lengths to maximize GPU utilization
- Reduces padding overhead and improves memory efficiency
- Automatic batch size tuning based on available GPU memory

### 2. CUDA Graph Optimization  
- Pre-captures computation graphs for common batch sizes
- Eliminates kernel launch overhead for decode phases
- Up to 40% performance improvement for small batch sizes

### 3. Memory Management
- Efficient KV-cache allocation and reuse
- Dynamic memory pool management
- Optimized attention kernels with Triton

### 4. Asynchronous Processing
- Non-blocking request handling with FastAPI
- Multiprocess inference pipeline
- Streaming response generation

## 🔧 Configuration

### Command Line Arguments

```bash
python python/sfllm/serving/app.py \
  --model MODEL_PATH \              # Path to model directory
  --port 8081 \                     # Server port
  --dtype float16 \                 # Model precision
  --cuda-graph-max-bs 32 \          # Max CUDA graph batch size
  --max-context-length 4096 \       # Maximum context length
  --disable-cuda-graph             # Disable CUDA graphs
```

### Environment Variables

```bash
export SFLLM_MODEL_PATH=/path/to/model
export SFLLM_PORT=8081
export SFLLM_DTYPE=float16
export CUDA_VISIBLE_DEVICES=0
```

## 🧪 Testing

### Run Performance Benchmarks
```bash
# Single request baseline
python python/sfllm/serving/benchmark.py \
  --url http://localhost:8081 \
  --concurrency 1 \
  --requests 40

# High concurrency test  
python python/sfllm/serving/benchmark.py \
  --url http://localhost:8081 \
  --concurrency 20 \
  --requests 100
```

### Test Streaming API
```bash
python test_stream.py
```

## 🛡️ Production Deployment

### Docker Deployment
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt

EXPOSE 8081
CMD ["python", "python/sfllm/serving/app.py", "--port", "8081"]
```

### Health Monitoring
```bash
# Health check endpoint
curl http://localhost:8081/health

# Response
{
  "status": "healthy",
  "timestamp": 1698765432
}
```

### Load Balancing with nginx
```nginx
upstream sfllm_backend {
    server 127.0.0.1:8081;
    server 127.0.0.1:8082;
    server 127.0.0.1:8083;
}

server {
    listen 80;
    location / {
        proxy_pass http://sfllm_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](./CONTRIBUTING.md) for details.

### Development Setup
```bash
# Install development dependencies
pip install -e ".[dev]"

# Run code formatting
black python/
isort python/

# Run type checking  
mypy python/sfllm/

# Run tests
pytest
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/) and [PyTorch](https://pytorch.org/)
- Inspired by [vLLM](https://github.com/vllm-project/vllm) and [SGLang](https://github.com/sgl-project/sglang)
- Model architecture based on [Qwen3](https://huggingface.co/Qwen/Qwen3-0.6B)

## 🔗 Links

- [Documentation](./docs/)
- [Issue Tracker](https://github.com/wejoncy/sfllm/issues)
- [Changelog](./CHANGELOG.md)
- [Roadmap](./ROADMAP.md)

---

**Made with ❤️ by [wejoncy](https://github.com/wejoncy)**