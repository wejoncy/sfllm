# Engine Module

The Engine module is the core of SFLLM, responsible for orchestrating model inference, managing resources, and optimizing performance through intelligent batching and scheduling.

## Overview

The Engine module provides a high-performance inference runtime that maximizes GPU utilization through:

- **Intelligent Batching**: Groups requests by sequence length for optimal memory usage
- **CUDA Graph Optimization**: Eliminates kernel launch overhead for common batch sizes
- **Memory Pool Management**: Efficient KV-cache allocation and reuse
- **Request Scheduling**: Prioritizes requests based on latency requirements

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    InferenceEngine                          │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  Scheduler  │  │ MemoryPool  │  │ ModelRunner │        │
│  │             │  │             │  │             │        │
│  │ • Batching  │  │ • KV Cache  │  │ • Forward   │        │
│  │ • Priority  │  │ • Alloc/Free│  │ • CUDA Graphs│        │
│  │ • Preempt   │  │ • Defrag    │  │ • Sampling  │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
│            │              │              │                 │
│  ┌─────────────────────────────────────────────────────────┤
│  │                 RequestSequence Management                     │
│  │  • Request tracking  • State persistence  • Cleanup    │
│  └─────────────────────────────────────────────────────────┘
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### InferenceEngine

The main orchestrator that coordinates all inference operations.

**Class Definition:**
```python
class InferenceEngine:
    def __init__(
        self,
        model_path: str,
        dtype: str = "auto",
        max_context_length: int = 8192,
        max_running_tokens: int = 16384,
        cuda_graph_max_bs: int = 32,
        **kwargs
    ):
        """Initialize the inference engine.
        
        Args:
            model_path: Path to model files or HuggingFace model name
            dtype: Model precision (auto, float16, bfloat16, float32)
            max_context_length: Maximum input sequence length
            max_running_tokens: Maximum total tokens across all sequences
            cuda_graph_max_bs: Maximum batch size for CUDA graph capture
        """
```

**Key Methods:**

```python
async def add_request(self, request: SequenceRequest) -> str:
    """Add a new generation request.
    
    Args:
        request: Request containing prompt, generation params
        
    Returns:
        request_id: Unique identifier for tracking
    """

async def generate(self, request_id: str) -> AsyncIterator[GenerationChunk]:
    """Generate tokens for a request.
    
    Args:
        request_id: Request identifier
        
    Yields:
        GenerationChunk: Token and metadata for each generated step
    """

def get_stats(self) -> Dict[str, Any]:
    """Get engine performance statistics.
    
    Returns:
        Dictionary containing:
        - active_requests: Number of active requests
        - completed_requests: Total completed requests
        - avg_batch_size: Average batch size
        - gpu_memory_usage: GPU memory utilization
    """
```

**Usage Example:**
```python
from sfllm.engine import InferenceEngine
from sfllm.engine.sampling_params import SamplingParams

# Initialize engine
engine = InferenceEngine(
    model_path="Qwen/Qwen3-0.6B",
    dtype="float16",
    max_context_length=8192
)

# Create request
sampling_params = SamplingParams(
    max_new_tokens=50,
    temperature=0.7,
    top_p=0.9
)

request_id = await engine.add_request(
    prompt="The future of AI is",
    sampling_params=sampling_params
)

# Generate response
async for chunk in engine.generate(request_id):
    print(chunk.token, end="", flush=True)
```

### ModelRunner

Executes the actual model forward pass with optimizations.

**Class Definition:**
```python
class ModelRunner:
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        dtype: torch.dtype,
        device: str,
        cuda_graph_config: CudaGraphConfig
    ):
        """Initialize model runner with CUDA graph support."""
```

**CUDA Graph Integration:**
```python
def capture_cuda_graphs(self) -> None:
    """Capture CUDA graphs for common batch sizes.
    
    This eliminates kernel launch overhead by pre-recording
    computation sequences for frequently used batch sizes.
    """
    
    for batch_size in self.cuda_graph_batch_sizes:
        # Create dummy input for this batch size
        dummy_input = self.create_dummy_input(batch_size)
        
        # Warmup
        for _ in range(3):
            self.model(dummy_input)
        torch.cuda.synchronize()
        
        # Capture graph
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = self.model(dummy_input)
        
        self.cuda_graphs[batch_size] = (graph, dummy_input, output)
```

**Forward Pass Execution:**
```python
def forward(
    self, 
    input_batch: InputBatch,
    use_cuda_graph: bool = True
) -> OutputBatch:
    """Execute forward pass with optional CUDA graph acceleration.
    
    Args:
        input_batch: Batched input tokens and attention masks
        use_cuda_graph: Whether to use CUDA graphs if available
        
    Returns:
        OutputBatch: Logits and hidden states
    """
    
    batch_size = input_batch.batch_size
    
    # Use CUDA graph if available for this batch size
    if (use_cuda_graph and 
        batch_size in self.cuda_graphs and 
        input_batch.is_decode_phase):
        
        graph, graph_input, graph_output = self.cuda_graphs[batch_size]
        
        # Copy input to graph input tensors
        graph_input.copy_(input_batch.input_ids)
        
        # Replay captured graph
        graph.replay()
        
        # Copy output from graph output tensors
        return OutputBatch(logits=graph_output.clone())
    
    # Fallback to regular forward pass
    return self.model(input_batch)
```

### Scheduler

Manages request batching and execution order for optimal performance.

**Batching Strategy:**
```python
class BatchScheduler:
    def __init__(self, max_batch_size: int = 32):
        self.max_batch_size = max_batch_size
        self.pending_requests = deque()
        self.running_requests = {}
    
    def create_batch(self) -> Optional[List[RequestSequence]]:
        """Create an optimal batch from pending requests.
        
        Strategy:
        1. Group requests by sequence length (±10% tolerance)
        2. Prioritize high-priority requests
        3. Maximize batch size within memory constraints
        """
        
        if not self.pending_requests:
            return None
        
        # Group by sequence length buckets
        length_groups = {}
        for seq in self.pending_requests:
            bucket = seq.get_length() // 64  # 64-token buckets
            length_groups.setdefault(bucket, []).append(seq)
        
        # Find largest feasible group
        for group in sorted(length_groups.values(), key=len, reverse=True):
            if self.estimate_memory_usage(group) < self.available_memory:
                batch_size = min(len(group), self.max_batch_size)
                return group[:batch_size]
        
        # Fallback: single sequence
        return [self.pending_requests.popleft()]
```

**Priority Scheduling:**
```python
def schedule_next_batch(self) -> List[RequestSequence]:
    """Schedule the next batch considering priorities.
    
    Priority levels:
    - HIGH: Interactive requests (streaming chat)
    - NORMAL: Batch requests  
    - LOW: Background/bulk processing
    """
    
    # Separate by priority
    high_priority = [seq for seq in self.pending_requests 
                    if seq.priority == Priority.HIGH]
    normal_priority = [seq for seq in self.pending_requests 
                      if seq.priority == Priority.NORMAL]
    
    # Always prioritize high-priority requests
    if high_priority:
        return self.create_batch_from_sequences(high_priority)
    
    # Mix normal and low priority
    return self.create_batch_from_sequences(normal_priority)
```

### MemoryPool

Efficient KV-cache management with block-based allocation.

**Block Allocation:**
```python
class MemoryPool:
    def __init__(
        self, 
        total_blocks: int,
        block_size: int = 16,  # tokens per block
        device: str = "cuda"
    ):
        """Initialize memory pool with block-based allocation.
        
        Args:
            total_blocks: Total number of cache blocks
            block_size: Tokens per block (affects memory granularity)
            device: Device to allocate memory on
        """
        
        self.block_size = block_size
        self.total_blocks = total_blocks
        
        # Pre-allocate all blocks
        self.key_cache = torch.empty(
            (total_blocks, num_layers, block_size, num_heads, head_dim),
            dtype=torch.float16,
            device=device
        )
        self.value_cache = torch.empty_like(self.key_cache)
        
        # Track block allocation
        self.free_blocks = set(range(total_blocks))
        self.allocated_blocks = {}  # sequence_id -> block_ids
```

**Allocation Strategy:**
```python
def allocate_blocks(self, sequence_id: str, num_tokens: int) -> List[int]:
    """Allocate cache blocks for a sequence.
    
    Args:
        sequence_id: Unique sequence identifier
        num_tokens: Number of tokens to cache
        
    Returns:
        List of allocated block IDs
    """
    
    num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
    
    if len(self.free_blocks) < num_blocks_needed:
        # Trigger garbage collection
        self.garbage_collect()
        
        if len(self.free_blocks) < num_blocks_needed:
            raise RuntimeError("Out of memory blocks")
    
    # Allocate contiguous blocks when possible
    allocated = []
    for _ in range(num_blocks_needed):
        block_id = self.free_blocks.pop()
        allocated.append(block_id)
    
    self.allocated_blocks[sequence_id] = allocated
    return allocated

def deallocate_blocks(self, sequence_id: str) -> None:
    """Free blocks allocated to a sequence."""
    
    if sequence_id in self.allocated_blocks:
        blocks = self.allocated_blocks.pop(sequence_id)
        self.free_blocks.update(blocks)
```

### RequestSequence

Tracks individual request state throughout generation.

**State Management:**
```python
class RequestSequence:
    def __init__(
        self,
        request_id: str,
        prompt: str,
        sampling_params: SamplingParams,
        arrival_time: float
    ):
        self.request_id = request_id
        self.prompt = prompt
        self.sampling_params = sampling_params
        self.arrival_time = arrival_time
        
        # Generation state
        self.input_ids = []
        self.generated_ids = []
        self.finished = False
        self.finish_reason = None
        
        # Performance tracking
        self.prefill_time = None
        self.decode_start_time = None
        self.total_tokens = 0
```

**Token Generation:**
```python
def add_generated_token(self, token_id: int) -> None:
    """Add a generated token and update state."""
    
    self.generated_ids.append(token_id)
    self.total_tokens += 1
    
    # Check stopping criteria
    if (self.total_tokens >= self.sampling_params.max_new_tokens or
        token_id == self.tokenizer.eos_token_id):
        self.finish_generation("stop")

def finish_generation(self, reason: str) -> None:
    """Mark sequence as finished with given reason."""
    
    self.finished = True
    self.finish_reason = reason
    self.completion_time = time.time()
    
    # Calculate metrics
    self.total_latency = self.completion_time - self.arrival_time
    if self.decode_start_time:
        self.decode_latency = self.completion_time - self.decode_start_time
```

## Performance Optimizations

### CUDA Graph Benefits

CUDA graphs provide significant performance improvements:

```python
# Without CUDA graphs (kernel launch overhead)
for batch in batches:
    logits = model(batch)  # ~100μs kernel launch overhead

# With CUDA graphs (pre-recorded execution)
graph.replay()  # ~10μs replay overhead
```

**Performance Gains:**
- 20-40% faster inference for decode phase
- Reduced CPU-GPU synchronization
- Consistent latency for common batch sizes

### Memory Efficiency

**Block-based KV Cache:**
```python
# Traditional approach: allocate per sequence
cache = torch.zeros(seq_len, hidden_size)  # Fragmented memory

# Block-based approach: shared pool
blocks = memory_pool.allocate(sequence_id, num_tokens)  # Efficient reuse
```

**Benefits:**
- Reduced memory fragmentation
- Better cache locality
- Support for dynamic sequence lengths

### Intelligent Batching

**Length-based Grouping:**
```python
# Group sequences by similar lengths
short_sequences = [seq for seq in pending if seq.length <= 512]
medium_sequences = [seq for seq in pending if 512 < seq.length <= 2048]
long_sequences = [seq for seq in pending if seq.length > 2048]

# Process similar lengths together
batch = create_batch(medium_sequences)  # Minimal padding waste
```

## Configuration

### Engine Parameters

```python
# Memory configuration
engine = InferenceEngine(
    model_path="Qwen/Qwen3-0.6B",
    max_context_length=8192,      # Maximum input length
    max_running_tokens=16384,     # Total tokens across all sequences
    cuda_graph_max_bs=32          # Maximum batch size for graphs
)

# Performance tuning
engine = InferenceEngine(
    model_path="Qwen/Qwen3-0.6B",
    dtype="float16",              # Reduce memory usage
    block_size=16,                # KV cache block size
    max_batch_size=32,            # Limit batch size
    enable_cuda_graph=True        # Enable CUDA graphs
)
```

### Environment Variables

```bash
# PyTorch optimizations
export TORCH_CUDNN_V8_API_ENABLED=1
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

# Parallelism settings
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
```

This comprehensive documentation covers the Engine module's architecture, components, and optimization strategies for high-performance LLM inference.