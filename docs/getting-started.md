# Getting Started with SFLLM

This guide will help you get started with SFLLM, from installation to running your first inference server.

## Prerequisites

- **Python**: 3.8 or higher
- **PyTorch**: 2.0.1 or higher with CUDA support
- **CUDA**: 11.7 or higher (for GPU acceleration)
- **GPU Memory**: At least 4GB for small models (Qwen3-0.6B)

## Installation

### Option 1: From Source (Recommended for Development)

```bash
# Clone the repository
git clone https://github.com/wejoncy/sfllm.git
cd sfllm

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install SFLLM in development mode
pip install -e .
```

### Option 2: Using Docker

```bash
# Build the Docker image
docker build -t sfllm:latest .

# Run with GPU support
docker run --gpus all -p 8081:8081 -v /path/to/models:/models sfllm:latest \
  --model /models/your-model --port 8081
```

## Quick Start

### 1. Download a Model

For this example, we'll use the Qwen3-0.6B model:

```bash
# Using Hugging Face Hub
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-0.6B', local_dir='/path/to/models/Qwen3-0.6B')
"
```

### 2. Start the Server

```bash
python python/sfllm/serving/app.py \
  --model /path/to/models/Qwen3-0.6B \
  --port 8081 \
  --dtype float16
```

### 3. Test the Server

```bash
# Health check
curl http://localhost:8081/health

# Simple completion
curl -X POST "http://localhost:8081/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-0.6b",
    "prompt": "Hello, world!",
    "max_new_tokens": 50,
    "temperature": 0.7
  }'
```

## Configuration Options

### Basic Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model` | Path to model directory | Required |
| `--port` | Server port | 8081 |
| `--dtype` | Model precision (float16/float32) | auto |

### Performance Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--cuda-graph-max-bs` | Maximum batch size for CUDA graphs | 32 |
| `--max-context-length` | Maximum context length | 4096 |
| `--disable-cuda-graph` | Disable CUDA graph optimization | False |

### Example: High-Performance Configuration

```bash
python python/sfllm/serving/app.py \
  --model /path/to/models/Qwen3-0.6B \
  --port 8081 \
  --dtype float16 \
  --cuda-graph-max-bs 64 \
  --max-context-length 8192
```

## API Usage Examples

### Chat Completions

```python
import requests
import json

response = requests.post("http://localhost:8081/v1/chat/completions", 
    json={
        "model": "qwen3-0.6b",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is machine learning?"}
        ],
        "stream": False,
        "max_new_tokens": 256,
        "temperature": 0.7
    }
)

result = response.json()
print(result["choices"][0]["message"]["content"])
```

### Streaming Chat Completions

```python
import requests
import json

response = requests.post("http://localhost:8081/v1/chat/completions", 
    json={
        "model": "qwen3-0.6b",
        "messages": [
            {"role": "user", "content": "Tell me a story about robots"}
        ],
        "stream": True,
        "max_new_tokens": 512
    },
    stream=True
)

for line in response.iter_lines(decode_unicode=True):
    if line.strip():
        chunk = json.loads(line)
        delta = chunk["choices"][0].get("delta", {})
        content = delta.get("content", "")
        if content:
            print(content, end="", flush=True)
```

### Text Completions

```python
import requests

response = requests.post("http://localhost:8081/v1/completions", 
    json={
        "model": "qwen3-0.6b",
        "prompt": "The benefits of renewable energy include",
        "max_new_tokens": 128,
        "temperature": 0.8,
        "top_p": 0.9
    }
)

result = response.json()
print(result["choices"][0]["text"])
```

## Performance Tuning Tips

### 1. Batch Size Optimization
- Start with default CUDA graph batch sizes
- Monitor GPU utilization with `nvidia-smi`
- Increase `--cuda-graph-max-bs` if you have large GPU memory

### 2. Memory Management
- Use `float16` precision to save memory
- Adjust `--max-context-length` based on your use case
- Monitor memory usage and adjust batch sizes accordingly

### 3. Concurrency Testing
```bash
# Test different concurrency levels
for concurrency in 1 5 10 20; do
  echo "Testing concurrency: $concurrency"
  python python/sfllm/serving/benchmark.py \
    --url http://localhost:8081 \
    --concurrency $concurrency \
    --requests 50
done
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce precision or batch size
   --dtype float16 --cuda-graph-max-bs 16
   ```

2. **Port Already in Use**
   ```bash
   # Use a different port
   --port 8082
   ```

3. **Model Loading Issues**
   ```bash
   # Check model path and permissions
   ls -la /path/to/your/model/
   ```

### Debugging Mode

```bash
# Enable verbose logging
export SFLLM_LOG_LEVEL=DEBUG
python python/sfllm/serving/app.py --model /path/to/model
```

## Next Steps

- Explore the [API Reference](./api-reference.md) for detailed endpoint documentation
- Learn about [Architecture](./architecture.md) to understand the system design
- Check [Performance Tuning](./performance-tuning.md) for optimization techniques
- Review [Configuration](./configuration.md) for advanced settings

## Support

If you encounter any issues:

1. Check the [Troubleshooting Guide](./troubleshooting.md)
2. Search existing [GitHub Issues](https://github.com/wejoncy/sfllm/issues)
3. Create a new issue with detailed logs and system information