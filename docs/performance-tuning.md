# Performance Tuning Guide

This guide covers advanced techniques for optimizing SFLLM performance across different hardware configurations and use cases.

## Performance Metrics

### Key Performance Indicators

| Metric | Description | Target |
|--------|-------------|--------|
| **Throughput** | Requests per second | 1.5+ req/s (Qwen3-0.6B) |
| **TTFT** | Time to first token | <500ms |
| **TPOT** | Tokens per output token | <50ms |
| **GPU Utilization** | GPU compute usage | >80% |
| **Memory Utilization** | GPU memory usage | 70-90% |

### Measurement Tools

```bash
# Built-in benchmark
python python/sfllm/serving/benchmark.py \
  --url http://localhost:8081 \
  --concurrency 20 \
  --requests 100

# GPU monitoring
nvidia-smi dmon -s pucvmet -d 1

# System monitoring  
htop
iostat -x 1
```

## Hardware-Specific Optimizations

### Consumer GPUs (8-12GB VRAM)

**RTX 3070/4060 Ti (8GB):**
```bash
python python/sfllm/serving/app.py \
  --model Qwen/Qwen3-0.6B \
  --dtype float16 \
  --cuda-graph-max-bs 16 \
  --max-context-length 4096 \
  --max-running-tokens 8192
```

**RTX 3080 Ti/4070 Ti (12GB):**
```bash
python python/sfllm/serving/app.py \
  --model Qwen/Qwen3-0.6B \
  --dtype float16 \
  --cuda-graph-max-bs 32 \
  --max-context-length 8192 \
  --max-running-tokens 16384
```

### High-End GPUs (16-24GB VRAM)

**RTX 3090/4090 (24GB):**
```bash
python python/sfllm/serving/app.py \
  --model Qwen/Qwen3-1.8B \
  --dtype float16 \
  --cuda-graph-max-bs 64 \
  --max-context-length 16384 \
  --max-running-tokens 32768
```

**A100/H100 (40-80GB):**
```bash
python python/sfllm/serving/app.py \
  --model Qwen/Qwen3-4B \
  --dtype bfloat16 \
  --cuda-graph-max-bs 128 \
  --max-context-length 32768 \
  --max-running-tokens 65536
```

## Model-Specific Optimizations

### Qwen3-0.6B (Recommended for Development)

**Memory Requirements:**
- Model weights: ~1.2GB (float16)
- KV cache (4K context): ~2GB
- CUDA graphs: ~1GB
- Total: ~4.5GB minimum

**Optimal Configuration:**
```bash
--model Qwen/Qwen3-0.6B \
--dtype float16 \
--cuda-graph-max-bs 32 \
--max-context-length 8192
```

### Qwen3-1.8B (Production Ready)

**Memory Requirements:**
- Model weights: ~3.6GB (float16)  
- KV cache (8K context): ~4GB
- CUDA graphs: ~2GB
- Total: ~10GB minimum

**Optimal Configuration:**
```bash
--model Qwen/Qwen3-1.8B \
--dtype float16 \
--cuda-graph-max-bs 24 \
--max-context-length 8192
```

### Qwen3-4B (High Performance)

**Memory Requirements:**
- Model weights: ~8GB (float16)
- KV cache (16K context): ~8GB  
- CUDA graphs: ~4GB
- Total: ~20GB minimum

**Optimal Configuration:**
```bash
--model Qwen/Qwen3-4B \
--dtype float16 \
--cuda-graph-max-bs 16 \
--max-context-length 16384
```

## Batching Strategies

### Intelligent Length-Based Batching

SFLLM automatically groups requests by similar sequence lengths:

```python
# Batching algorithm
def create_optimal_batch(pending_requests):
    # Group by sequence length (±10% tolerance)
    length_groups = {}
    for req in pending_requests:
        key = req.input_length // 64  # 64-token buckets
        length_groups.setdefault(key, []).append(req)
    
    # Select largest feasible group
    for group in sorted(length_groups.values(), key=len, reverse=True):
        if estimate_memory_usage(group) < available_memory:
            return group[:max_batch_size]
```

### Batch Size Tuning

**Small Batch Sizes (1-8):**
- Better latency for individual requests
- Lower GPU utilization
- Suitable for interactive applications

**Medium Batch Sizes (16-32):**  
- Balanced latency and throughput
- Good GPU utilization
- Recommended for most use cases

**Large Batch Sizes (64-128):**
- Maximum throughput
- Higher latency for individual requests  
- Best for batch processing workloads

### Dynamic Batch Adjustment

```bash
# Monitor batch effectiveness
python -c "
import time
import requests
import threading

def measure_throughput(batch_size):
    # Implementation details...
    pass

# Test different batch sizes
for bs in [8, 16, 32, 64]:
    throughput = measure_throughput(bs)
    print(f'Batch size {bs}: {throughput:.2f} req/s')
"
```

## CUDA Graph Optimization

### Understanding CUDA Graphs

CUDA graphs eliminate kernel launch overhead by pre-recording computation sequences:

```python
# Benefits of CUDA graphs
- 20-40% performance improvement for decode phase
- Consistent latency for common batch sizes
- Reduced CPU-GPU synchronization overhead
```

### Optimal Graph Configuration

**Graph Batch Sizes:**
SFLLM captures graphs for these batch sizes by default:
```python
DEFAULT_CUDA_GRAPH_BATCH_SIZES = [1, 2, 3, 4, 6, 8, 10, 12, 14, 16] + 
                                  list(range(20, 2048+1, 4))
```

**Memory vs Performance Trade-off:**
```bash
# Conservative (lower memory usage)
--cuda-graph-max-bs 16

# Aggressive (higher performance)  
--cuda-graph-max-bs 128

# Disable for debugging
--disable-cuda-graph
```

### Graph Warmup

```python
# Automatic warmup on server start
for batch_size in capture_batch_sizes:
    # Warmup with dummy inputs
    dummy_input = create_dummy_batch(batch_size)
    model.forward(dummy_input)  # Pre-compile
    
    # Capture graph
    with torch.cuda.graph(graph):
        output = model.forward(dummy_input)
```

## Memory Optimization

### KV-Cache Management

**Memory Layout:**
```
┌─────────────────────────────────────┐
│          GPU Memory (24GB)          │
├─────────────────────────────────────┤
│ Model Weights (8GB)                 │
├─────────────────────────────────────┤  
│ KV Cache Pool (12GB)                │
│  ├─ Active Sequences                │
│  ├─ Cached Sequences                │
│  └─ Free Blocks                     │
├─────────────────────────────────────┤
│ CUDA Graphs (2GB)                   │
├─────────────────────────────────────┤
│ Computation Buffers (2GB)           │
└─────────────────────────────────────┘
```

**Cache Allocation Strategy:**
```python
# Allocate cache based on expected usage
max_sequences = 100
cache_block_size = 16  # tokens per block
kv_cache_size = (
    max_sequences * 
    max_context_length // cache_block_size *
    num_layers * 
    hidden_size * 
    2 * dtype_bytes  # key + value
)
```

### Memory Pool Management

**PyTorch Memory Allocator:**
```bash
# Reduce memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

# Enable memory debugging (development only)
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
```

**Memory Monitoring:**
```python
# Monitor memory usage
def print_memory_stats():
    allocated = torch.cuda.memory_allocated() / 1024**3
    cached = torch.cuda.memory_reserved() / 1024**3
    print(f"Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")

# Call periodically during inference
print_memory_stats()
```

## Attention Optimization

### Flash Attention Integration

```python
# Enable Flash Attention for supported operations
use_flash_attention = (
    torch.cuda.is_available() and 
    torch.cuda.get_device_capability()[0] >= 8  # A100, RTX 30/40 series
)

if use_flash_attention:
    from flash_attn import flash_attn_func
    attention_output = flash_attn_func(q, k, v, causal=True)
```

### Triton Kernels

Custom attention kernels for specific patterns:

```python
# Optimized decode attention
@triton.jit
def decode_attention_kernel(
    q_ptr, k_cache_ptr, v_cache_ptr, output_ptr,
    seq_len, head_dim, BLOCK_SIZE: tl.constexpr
):
    # Efficient decode-phase attention
    # Optimized for single token generation
```

### Attention Backend Selection

```bash
# Automatic backend selection based on input characteristics
- Flash Attention: Prefill phase (long sequences)
- Triton Kernels: Decode phase (single token)  
- PyTorch SDPA: Fallback for unsupported configurations
```

## Scheduling Optimizations

### Request Prioritization

```python
class PriorityScheduler:
    def __init__(self):
        self.high_priority = deque()    # Interactive requests
        self.normal_priority = deque()  # Batch requests
        self.background = deque()       # Background tasks
    
    def schedule_next_batch(self):
        # Prioritize interactive requests
        if self.high_priority:
            return self.create_batch(self.high_priority)
        
        # Mix normal and background requests
        mixed_batch = []
        mixed_batch.extend(list(self.normal_priority)[:batch_size//2])
        mixed_batch.extend(list(self.background)[:batch_size//2])
        
        return mixed_batch
```

### Preemptive Scheduling

```python
# Interrupt long-running generations for high-priority requests
def preempt_if_needed(self, new_request):
    if (new_request.priority == "high" and 
        self.current_batch_runtime > threshold):
        
        # Save current state
        self.checkpoint_current_batch()
        
        # Process high-priority request
        self.process_immediately(new_request)
        
        # Resume previous batch
        self.restore_batch_checkpoint()
```

## Network and I/O Optimization

### Connection Pooling

```python
# Optimize FastAPI for high concurrency
app = FastAPI(
    title="SFLLM Server",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure uvicorn for performance
uvicorn.run(
    app,
    host="0.0.0.0", 
    port=8081,
    workers=1,  # Single worker for GPU sharing
    loop="uvloop",  # Faster event loop
    access_log=False  # Disable for performance
)
```

### Streaming Optimization

```python
# Minimize serialization overhead
async def generate_stream(request_id, generator):
    # Pre-serialize common response parts
    base_response = {
        "id": request_id,
        "object": "text_completion", 
        "created": int(time.time()),
        "model": model_name
    }
    base_json = json.dumps(base_response)[:-1]  # Remove closing brace
    
    async for chunk in generator:
        # Minimal serialization per chunk
        choices_json = json.dumps({"choices": [chunk]})
        yield f'{base_json},"choices":[{choices_json}]}}\n'
```

## Benchmarking and Profiling

### Performance Profiling

**PyTorch Profiler:**
```python
from torch.profiler import profile, record_function, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    with record_function("model_inference"):
        output = model(input_batch)

# Analyze results
prof.export_chrome_trace("trace.json")
```

**NVIDIA Nsight Systems:**
```bash
# Profile entire server
nsys profile --force-overwrite=true -o sfllm_profile \
  --trace=cuda,nvtx,osrt,cudnn --cuda-graph-trace=node \
  python python/sfllm/serving/app.py --model /path/to/model

# Analyze with Nsight Systems GUI
nsight-sys sfllm_profile.nsys-rep
```

### Load Testing

**Benchmark Script:**
```bash
# Comprehensive benchmark
python python/sfllm/serving/benchmark.py \
  --url http://localhost:8081 \
  --concurrency 1,5,10,20,50 \
  --requests 100 \
  --output-file results.json \
  --prompt-file prompts.txt
```

**Custom Load Testing:**
```python
import asyncio
import aiohttp
import time
from concurrent.futures import ThreadPoolExecutor

async def load_test(concurrency=20, requests=100):
    start_time = time.time()
    semaphore = asyncio.Semaphore(concurrency)
    
    async def single_request(session):
        async with semaphore:
            async with session.post(
                "http://localhost:8081/v1/completions",
                json={
                    "model": "qwen3-0.6b",
                    "prompt": "The future of AI is",
                    "max_new_tokens": 50
                }
            ) as response:
                return await response.json()
    
    async with aiohttp.ClientSession() as session:
        tasks = [single_request(session) for _ in range(requests)]
        results = await asyncio.gather(*tasks)
    
    duration = time.time() - start_time
    throughput = requests / duration
    print(f"Throughput: {throughput:.2f} req/s")
    return results

# Run benchmark
asyncio.run(load_test(concurrency=20, requests=100))
```

## Production Deployment Optimizations

### Container Configuration

**Dockerfile:**
```dockerfile
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Optimize Python interpreter
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV OMP_NUM_THREADS=1

# Install optimized PyTorch
RUN pip install torch==2.1.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# Multi-stage build for smaller images
COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Resource limits
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
  CMD curl -f http://localhost:8081/health || exit 1
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sfllm-deployment
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: sfllm
        image: sfllm:latest
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: 32Gi
            cpu: 4
          requests:
            nvidia.com/gpu: 1
            memory: 16Gi
            cpu: 2
        env:
        - name: SFLLM_CUDA_GRAPH_MAX_BS
          value: "32"
        - name: PYTORCH_CUDA_ALLOC_CONF
          value: "max_split_size_mb:128"
        livenessProbe:
          httpGet:
            path: /health
            port: 8081
          initialDelaySeconds: 60
          periodSeconds: 30
```

### Load Balancer Configuration

**NGINX:**
```nginx
upstream sfllm_backend {
    least_conn;  # Route to least loaded server
    server 10.0.1.10:8081 max_fails=3 fail_timeout=30s;
    server 10.0.1.11:8081 max_fails=3 fail_timeout=30s;
    server 10.0.1.12:8081 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    
    location / {
        proxy_pass http://sfllm_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        
        # Optimize for streaming  
        proxy_buffering off;
        proxy_cache off;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }
    
    # Health check endpoint
    location /health {
        proxy_pass http://sfllm_backend/health;
        access_log off;
    }
}
```

This performance tuning guide provides comprehensive optimization strategies for maximizing SFLLM performance across different deployment scenarios and hardware configurations.