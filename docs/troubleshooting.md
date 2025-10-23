# Troubleshooting Guide

This guide covers common issues and their solutions when using SFLLM.

## Common Issues

### 1. CUDA Runtime Errors

#### Error: "RuntimeError: Cannot re-initialize CUDA in forked subprocess"

**Cause:** Using `fork` multiprocessing method with CUDA operations.

**Solution:**
```python
# In your application startup
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
```

**Complete fix in engine_server.py:**
```python
import multiprocessing as mp

class EngineServer:
    def __init__(self, server_args):
        # Force spawn method for CUDA compatibility
        if mp.get_start_method(allow_none=True) != 'spawn':
            mp.set_start_method('spawn', force=True)
        
        self.server_args = server_args
        self.process = None
```

#### Error: "CUDA out of memory"

**Diagnosis:**
```bash
# Check GPU memory usage
nvidia-smi

# Monitor memory during operation
watch -n 1 nvidia-smi
```

**Solutions:**

1. **Reduce batch size:**
```bash
python python/sfllm/serving/app.py \
  --model Qwen/Qwen3-0.6B \
  --cuda-graph-max-bs 8  # Reduce from default 32
```

2. **Lower precision:**
```bash
python python/sfllm/serving/app.py \
  --model Qwen/Qwen3-0.6B \
  --dtype float16  # or bfloat16
```

3. **Reduce context length:**
```bash
python python/sfllm/serving/app.py \
  --model Qwen/Qwen3-0.6B \
  --max-context-length 4096  # Reduce from default 8192
```

4. **Disable CUDA graphs temporarily:**
```bash
python python/sfllm/serving/app.py \
  --model Qwen/Qwen3-0.6B \
  --disable-cuda-graph
```

#### Error: "CUDA driver version is insufficient"

**Diagnosis:**
```bash
# Check CUDA driver version
nvidia-smi

# Check CUDA runtime version
nvcc --version

# Check PyTorch CUDA version
python -c "import torch; print(torch.version.cuda)"
```

**Solution:**
Update NVIDIA drivers to match PyTorch requirements:
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install nvidia-driver-535  # or latest

# Check compatibility
nvidia-smi
```

### 2. Model Loading Issues

#### Error: "Model not found" or "Repository not found"

**Cause:** Model path or Hugging Face repository issues.

**Diagnosis:**
```python
# Test model access
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    print("Tokenizer loaded successfully")
except Exception as e:
    print(f"Tokenizer error: {e}")

try:
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
    print("Model loaded successfully")
except Exception as e:
    print(f"Model error: {e}")
```

**Solutions:**

1. **Use local model path:**
```bash
# Download model first
git clone https://huggingface.co/Qwen/Qwen3-0.6B /path/to/models/Qwen3-0.6B

# Use local path
python python/sfllm/serving/app.py \
  --model /path/to/models/Qwen3-0.6B
```

2. **Set Hugging Face cache:**
```bash
export HF_HOME=/path/to/huggingface_cache
export TRANSFORMERS_CACHE=/path/to/transformers_cache
```

3. **Use authentication token:**
```bash
# Login to Hugging Face
huggingface-cli login

# Or set token
export HUGGINGFACE_HUB_TOKEN=your_token_here
```

#### Error: "Unsupported model type"

**Cause:** Model architecture not supported by SFLLM.

**Supported Models:**
- Qwen3 series (0.6B, 1.8B, 4B, 7B)
- Gemma3 series (2B, 7B)

**Solution:**
Check model architecture:
```python
from transformers import AutoConfig

config = AutoConfig.from_pretrained("model_name")
print(f"Model type: {config.model_type}")
print(f"Architecture: {config.architectures}")
```

### 3. Server Startup Issues

#### Error: "Port already in use"

**Diagnosis:**
```bash
# Check what's using the port
lsof -i :8081
netstat -tulpn | grep 8081
```

**Solutions:**

1. **Use different port:**
```bash
python python/sfllm/serving/app.py \
  --model Qwen/Qwen3-0.6B \
  --port 8082
```

2. **Kill existing process:**
```bash
# Find and kill process
sudo kill -9 $(lsof -t -i:8081)

# Or kill by name
pkill -f "sfllm"
```

#### Error: "Failed to bind socket"

**Cause:** Permission issues or address already in use.

**Solution:**
```bash
# Use different host/port combination
python python/sfllm/serving/app.py \
  --model Qwen/Qwen3-0.6B \
  --host 127.0.0.1 \
  --port 8081

# Or run with elevated permissions (not recommended for production)
sudo python python/sfllm/serving/app.py \
  --model Qwen/Qwen3-0.6B \
  --host 0.0.0.0 \
  --port 80
```

### 4. Performance Issues

#### Slow inference speed

**Diagnosis:**
```bash
# Run benchmark
python python/sfllm/serving/benchmark.py \
  --url http://localhost:8081 \
  --concurrency 1 \
  --requests 10

# Monitor GPU utilization
nvidia-smi dmon -s u -d 1
```

**Solutions:**

1. **Enable CUDA graphs:**
```bash
python python/sfllm/serving/app.py \
  --model Qwen/Qwen3-0.6B \
  --cuda-graph-max-bs 32  # Ensure this is enabled
```

2. **Optimize batch size:**
```bash
# Test different batch sizes
for bs in 8 16 32 64; do
  echo "Testing batch size: $bs"
  python python/sfllm/serving/app.py \
    --model Qwen/Qwen3-0.6B \
    --cuda-graph-max-bs $bs &
  
  sleep 10
  python python/sfllm/serving/benchmark.py --url http://localhost:8081
  pkill -f "sfllm"
  sleep 5
done
```

3. **Check CPU bottlenecks:**
```bash
# Monitor CPU usage
htop

# Optimize tokenizer
export TOKENIZERS_PARALLELISM=false  # Avoid warnings
export OMP_NUM_THREADS=1  # Prevent CPU oversubscription
```

#### High memory usage

**Diagnosis:**
```python
# Memory profiling script
import torch
import psutil
import time

def print_memory_usage():
    # GPU memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**3
        gpu_cached = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory - Allocated: {gpu_memory:.2f}GB, Reserved: {gpu_cached:.2f}GB")
    
    # System memory
    memory = psutil.virtual_memory()
    print(f"System Memory - Used: {memory.used/1024**3:.2f}GB ({memory.percent}%)")

# Monitor continuously
while True:
    print_memory_usage()
    time.sleep(5)
```

**Solutions:**

1. **Reduce KV cache size:**
```bash
python python/sfllm/serving/app.py \
  --model Qwen/Qwen3-0.6B \
  --max-context-length 4096 \
  --max-running-tokens 8192
```

2. **Optimize memory allocator:**
```bash
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
```

### 5. API and Client Issues

#### Error: "Connection refused" or "Connection timeout"

**Diagnosis:**
```bash
# Test server connectivity
curl -X GET http://localhost:8081/health

# Test with verbose output
curl -v -X POST http://localhost:8081/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3-0.6b", "prompt": "Hello", "max_new_tokens": 10}'
```

**Solutions:**

1. **Check server logs:**
```bash
# Run server with debug logging
python python/sfllm/serving/app.py \
  --model Qwen/Qwen3-0.6B \
  --log-level DEBUG
```

2. **Verify network configuration:**
```bash
# Check if server is listening
netstat -tlnp | grep 8081

# Test from different locations
curl http://127.0.0.1:8081/health  # Local
curl http://0.0.0.0:8081/health     # All interfaces
```

#### Error: "Invalid request format"

**Common request format issues:**

1. **Missing required fields:**
```json
// Incorrect
{
  "prompt": "Hello world"
}

// Correct
{
  "model": "qwen3-0.6b",
  "prompt": "Hello world",
  "max_new_tokens": 50
}
```

2. **Invalid parameter values:**
```json
// Incorrect
{
  "model": "qwen3-0.6b",
  "prompt": "Hello",
  "max_new_tokens": -1  // Invalid negative value
}

// Correct
{
  "model": "qwen3-0.6b", 
  "prompt": "Hello",
  "max_new_tokens": 50
}
```

#### Streaming issues

**Problem:** Streaming responses not working properly.

**Debug streaming:**
```python
import requests
import json

def test_streaming():
    url = "http://localhost:8081/v1/completions"
    data = {
        "model": "qwen3-0.6b",
        "prompt": "The future of AI is",
        "max_new_tokens": 50,
        "stream": True
    }
    
    try:
        response = requests.post(url, json=data, stream=True)
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith('data: '):
                    json_str = decoded_line[6:]  # Remove 'data: ' prefix
                    if json_str.strip() != '[DONE]':
                        try:
                            chunk = json.loads(json_str)
                            print(chunk)
                        except json.JSONDecodeError as e:
                            print(f"JSON decode error: {e}")
                            print(f"Raw line: {decoded_line}")
                            
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")

test_streaming()
```

### 6. Installation and Environment Issues

#### Python version incompatibility

**Requirements:**
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU acceleration)

**Check versions:**
```bash
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
nvcc --version
```

#### Missing dependencies

**Install all dependencies:**
```bash
# Basic installation
pip install -e .

# Development installation
pip install -r requirements.txt
pip install -e ".[dev]"

# Manual dependency installation
pip install torch>=2.0.0
pip install transformers>=4.35.0
pip install fastapi>=0.104.0
pip install uvicorn>=0.24.0
```

#### Virtual environment issues

**Create clean environment:**
```bash
# Using conda
conda create -n sfllm python=3.10
conda activate sfllm
pip install -e .

# Using venv
python -m venv sfllm_env
source sfllm_env/bin/activate  # Linux/Mac
# sfllm_env\Scripts\activate  # Windows
pip install -e .
```

### 7. Docker and Container Issues

#### Container fails to start

**Common Docker issues:**

1. **GPU access in container:**
```bash
# Ensure NVIDIA Docker runtime is installed
docker run --gpus all nvidia/cuda:11.8-base nvidia-smi

# Run SFLLM container with GPU
docker run --gpus all -p 8081:8081 sfllm:latest
```

2. **Memory limits:**
```bash
# Increase container memory
docker run --gpus all --memory=16g -p 8081:8081 sfllm:latest
```

3. **Volume mounting:**
```bash
# Mount model directory
docker run --gpus all -v /host/models:/models \
  -p 8081:8081 sfllm:latest \
  --model /models/Qwen3-0.6B
```

### 8. Development and Debugging

#### Enable debug logging

**Application-level debugging:**
```bash
python python/sfllm/serving/app.py \
  --model Qwen/Qwen3-0.6B \
  --log-level DEBUG
```

**Python logging configuration:**
```python
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

#### Profiling and monitoring

**PyTorch profiler:**
```python
from torch.profiler import profile, record_function, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    # Your inference code here
    pass

prof.export_chrome_trace("trace.json")
```

**Memory profiling:**
```python
import tracemalloc
tracemalloc.start()

# Your code here

current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current / 1024 / 1024:.1f} MB")
print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")
tracemalloc.stop()
```

## Getting Help

### Collecting diagnostic information

**System information script:**
```bash
#!/bin/bash
echo "=== SFLLM Diagnostic Information ==="
echo "Date: $(date)"
echo ""

echo "=== System Information ==="
uname -a
echo "Python: $(python --version)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
echo ""

echo "=== CUDA Information ==="
nvcc --version
nvidia-smi --query-gpu=driver_version,cuda_version --format=csv,noheader,nounits
echo ""

echo "=== Python Packages ==="
pip list | grep -E "(torch|transformers|fastapi|sfllm)"
echo ""

echo "=== SFLLM Server Status ==="
curl -s http://localhost:8081/health || echo "Server not responding"
echo ""

echo "=== Recent Logs ==="
# Add your log file path here
# tail -50 /path/to/sfllm.log
```

### Community and Support

**GitHub Repository:**
- Issues: [https://github.com/your-org/sfllm/issues](https://github.com/your-org/sfllm/issues)
- Discussions: [https://github.com/your-org/sfllm/discussions](https://github.com/your-org/sfllm/discussions)

**When reporting issues, include:**
1. Full error message and stack trace
2. SFLLM version and commit hash
3. System information (OS, Python, CUDA versions)
4. Hardware specifications (GPU model, memory)
5. Complete command used to start the server
6. Minimal reproduction steps

**Useful debugging commands:**
```bash
# Check SFLLM installation
python -c "import sfllm; print(sfllm.__version__)"

# Verify model compatibility
python -c "
from sfllm.models import model_registry
print('Supported models:', list(model_registry.keys()))
"

# Test basic functionality
python -c "
from sfllm.engine.inference_engine import InferenceEngine
print('InferenceEngine import successful')
"
```

This troubleshooting guide should help you resolve most common issues when working with SFLLM. For issues not covered here, please check the GitHub repository or create a new issue with detailed information.