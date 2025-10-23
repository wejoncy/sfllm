# Configuration Reference

This document provides a comprehensive reference for all configuration options available in SFLLM.

## Command Line Arguments

### Basic Configuration

#### `--model` (Required)
- **Type**: `str`
- **Description**: Path to the model directory or Hugging Face model identifier
- **Example**: `--model /path/to/Qwen3-0.6B` or `--model Qwen/Qwen3-0.6B`

#### `--port`
- **Type**: `int`
- **Default**: `8081`
- **Description**: Port number for the HTTP server
- **Example**: `--port 8080`

#### `--dtype`
- **Type**: `str`
- **Default**: `"auto"`
- **Options**: `"auto"`, `"float16"`, `"float32"`, `"bfloat16"`
- **Description**: Model precision/data type
- **Example**: `--dtype float16`

### Performance Configuration

#### `--cuda-graph-max-bs`
- **Type**: `int`
- **Default**: `32`
- **Description**: Maximum batch size for CUDA graph optimization
- **Range**: `1-512`
- **Example**: `--cuda-graph-max-bs 64`

#### `--max-context-length`
- **Type**: `int`
- **Default**: `4096`
- **Description**: Maximum context length supported
- **Example**: `--max-context-length 8192`

#### `--disable-cuda-graph`
- **Type**: `bool` (flag)
- **Default**: `False`
- **Description**: Disable CUDA graph optimization
- **Example**: `--disable-cuda-graph`

### Advanced Configuration

#### `--max-running-tokens`
- **Type**: `int`
- **Default**: `8192`
- **Description**: Maximum number of tokens in running sequences
- **Example**: `--max-running-tokens 16384`

#### `--log-level`
- **Type**: `str`
- **Default**: `"INFO"`
- **Options**: `"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"`
- **Description**: Logging verbosity level
- **Example**: `--log-level DEBUG`

## Environment Variables

### Core Settings

```bash
# Model configuration
export SFLLM_MODEL_PATH="/path/to/model"
export SFLLM_DTYPE="float16"

# Server configuration
export SFLLM_PORT="8081"
export SFLLM_HOST="0.0.0.0"

# Performance settings
export SFLLM_CUDA_GRAPH_MAX_BS="32"
export SFLLM_MAX_CONTEXT_LENGTH="4096"

# GPU configuration
export CUDA_VISIBLE_DEVICES="0"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"

# Logging
export SFLLM_LOG_LEVEL="INFO"
export PYTHONPATH="/path/to/sfllm/python:$PYTHONPATH"
```

### Memory Management

```bash
# PyTorch memory management
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="1"

# CUDA settings
export CUDA_LAUNCH_BLOCKING="0"
export CUDA_CACHE_DISABLE="0"
```

## Configuration Files

### Server Configuration (`config.py`)

```python
# serving/config.py
REQUEST_TIMEOUT = 300  # seconds
MAX_QUEUE_SIZE = 1000
HEALTH_CHECK_INTERVAL = 30  # seconds

# Streaming configuration  
STREAM_CHUNK_SIZE = 1024
STREAM_TIMEOUT = 60

# CORS settings
CORS_ORIGINS = ["*"]
CORS_METHODS = ["GET", "POST", "OPTIONS"]
CORS_HEADERS = ["*"]
```

### Model Configuration

Models are configured via the standard Hugging Face `config.json`:

```json
{
  "architectures": ["Qwen3ForCausalLM"],
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "hidden_act": "silu", 
  "hidden_size": 896,
  "intermediate_size": 4864,
  "max_position_embeddings": 32768,
  "model_type": "qwen3",
  "num_attention_heads": 14,
  "num_hidden_layers": 24,
  "num_key_value_heads": 2,
  "rms_norm_eps": 1e-06,
  "rope_theta": 1000000.0,
  "tie_word_embeddings": true,
  "torch_dtype": "float16",
  "transformers_version": "4.39.0",
  "use_cache": true,
  "vocab_size": 151936
}
```

## Runtime Configuration

### ServerArgs Class

All configuration is centralized in the `ServerArgs` dataclass:

```python
@dataclass
class ServerArgs:
    # Model settings
    model_path: str
    dtype: str = "auto"
    
    # Server settings  
    port: int = 8081
    host: str = "0.0.0.0"
    
    # Performance settings
    cuda_graph_max_bs: int = 32
    max_context_length: int = 4096
    max_running_tokens: int = 8192
    disable_cuda_graph: bool = False
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    @classmethod
    def from_cli_args(cls, args) -> 'ServerArgs':
        """Create ServerArgs from command line arguments"""
        
    @classmethod  
    def from_env(cls) -> 'ServerArgs':
        """Create ServerArgs from environment variables"""
```

### Configuration Priority

Configuration values are resolved in this order (highest to lowest priority):

1. Command line arguments
2. Environment variables  
3. Configuration files
4. Default values

```python
# Example resolution
model_path = (
    args.model_path or 
    os.getenv('SFLLM_MODEL_PATH') or 
    config.get('model_path') or
    None  # Required, no default
)
```

## Configuration Examples

### Development Setup

```bash
python python/sfllm/serving/app.py \
  --model /local/models/Qwen3-0.6B \
  --port 8081 \
  --dtype float32 \
  --log-level DEBUG \
  --disable-cuda-graph
```

### Production Setup

```bash
export SFLLM_MODEL_PATH="/opt/models/Qwen3-0.6B"
export SFLLM_DTYPE="float16"
export SFLLM_PORT="8081"
export SFLLM_CUDA_GRAPH_MAX_BS="64"
export SFLLM_LOG_LEVEL="WARNING"

python python/sfllm/serving/app.py
```

### High-Performance Setup

```bash
python python/sfllm/serving/app.py \
  --model /fast-ssd/models/Qwen3-0.6B \
  --dtype float16 \
  --cuda-graph-max-bs 128 \
  --max-context-length 8192 \
  --max-running-tokens 32768
```

### Memory-Constrained Setup

```bash
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:64"

python python/sfllm/serving/app.py \
  --model Qwen/Qwen3-0.6B \
  --dtype float16 \
  --cuda-graph-max-bs 16 \
  --max-context-length 2048
```

## Model-Specific Configuration

### Qwen3 Models

```bash
# Qwen3-0.6B (Recommended for development)
--model Qwen/Qwen3-0.6B --dtype float16 --cuda-graph-max-bs 32

# Qwen3-1.8B  
--model Qwen/Qwen3-1.8B --dtype float16 --cuda-graph-max-bs 24

# Qwen3-4B (High memory requirement)
--model Qwen/Qwen3-4B --dtype float16 --cuda-graph-max-bs 16
```

### Custom Models

For custom models, ensure your `config.json` includes:

```json
{
  "model_type": "qwen3",
  "architectures": ["Qwen3ForCausalLM"],
  "torch_dtype": "float16"
}
```

## Performance Tuning

### GPU Memory Optimization

```bash
# For 8GB GPU (RTX 3070/4060 Ti)
--dtype float16 --cuda-graph-max-bs 16 --max-context-length 4096

# For 12GB GPU (RTX 3080 Ti/4070 Ti) 
--dtype float16 --cuda-graph-max-bs 32 --max-context-length 8192

# For 24GB GPU (RTX 3090/4090)
--dtype float16 --cuda-graph-max-bs 64 --max-context-length 16384
```

### Throughput vs Latency

**Optimize for Throughput:**
```bash
--cuda-graph-max-bs 64 --max-running-tokens 16384
```

**Optimize for Latency:**  
```bash
--cuda-graph-max-bs 8 --max-context-length 2048
```

## Configuration Validation

SFLLM validates configuration at startup:

```python
def validate_config(args: ServerArgs):
    # Model path validation
    if not os.path.exists(args.model_path):
        raise ValueError(f"Model path not found: {args.model_path}")
    
    # GPU memory validation
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        if args.cuda_graph_max_bs > estimate_max_batch_size(gpu_memory):
            logger.warning("CUDA graph batch size may exceed GPU memory")
    
    # Port validation
    if not (1024 <= args.port <= 65535):
        raise ValueError(f"Port must be between 1024-65535, got {args.port}")
```

## Debugging Configuration

### Enable Debug Logging

```bash
export SFLLM_LOG_LEVEL="DEBUG"
python python/sfllm/serving/app.py --log-level DEBUG
```

### Configuration Dump

```python
# Add to startup to dump effective configuration
import json
print("Effective Configuration:")
print(json.dumps(asdict(server_args), indent=2))
```

### Memory Profiling

```bash
# Enable PyTorch profiling
export TORCH_PROFILER_ENABLED="1"

# Monitor GPU memory
watch -n 1 nvidia-smi
```

## Common Configuration Issues

### 1. CUDA Out of Memory

**Symptoms**: `RuntimeError: CUDA out of memory`

**Solutions**:
```bash
# Reduce batch size
--cuda-graph-max-bs 8

# Reduce precision
--dtype float16

# Reduce context length
--max-context-length 2048
```

### 2. Model Loading Errors

**Symptoms**: `FileNotFoundError` or `OSError` during model loading

**Solutions**:
```bash
# Check model path
ls -la /path/to/model/

# Verify model format
python -c "from transformers import AutoConfig; print(AutoConfig.from_pretrained('/path/to/model'))"

# Use Hugging Face model ID
--model Qwen/Qwen3-0.6B  # Downloads automatically
```

### 3. Performance Issues  

**Symptoms**: Low throughput or high latency

**Solutions**:
```bash
# Enable CUDA graphs (if disabled)
# Remove --disable-cuda-graph flag

# Increase batch size
--cuda-graph-max-bs 64

# Check GPU utilization
nvidia-smi dmon -s pucvmet -d 1
```

### 4. Port Binding Issues

**Symptoms**: `Address already in use`

**Solutions**:
```bash
# Use different port
--port 8082

# Kill existing process
sudo lsof -ti:8081 | xargs sudo kill -9

# Bind to specific interface
--host 127.0.0.1
```

This configuration reference covers all aspects of SFLLM configuration for various deployment scenarios and performance requirements.