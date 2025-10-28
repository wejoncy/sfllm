# Module Documentation

This directory contains detailed documentation for each SFLLM module.

## Core Modules

### [Engine](modules/engine.md)
Core inference engine with intelligent batching, scheduling, and CUDA graph optimization.

**Key Components:**
- `InferenceEngine`: Main orchestrator for request processing
- `ModelRunner`: CUDA-optimized model execution  
- `Scheduler`: Request batching and prioritization
- `MemoryPool`: KV-cache management
- `RequestSequence`: Request state tracking

### [Serving](modules/serving.md)
FastAPI-based web server with OpenAI-compatible REST APIs and streaming support.

**Key Components:**
- `app.py`: FastAPI application with endpoints
- `engine_server.py`: Multiprocess communication layer
- `client.py`: Python client library
- `benchmark.py`: Performance testing tools

### [Models](modules/models.md)
Model implementations and architecture-specific optimizations.

**Supported Architectures:**
- `Qwen3`: Qwen 3 series (0.6B, 1.8B, 4B, 7B)
- `Gemma3`: Gemma 3 series (2B, 7B)

### [Layers](modules/layers.md)
Optimized neural network layers and attention mechanisms.

**Key Components:**
- `TritonAttention`: Custom attention implementation
- `Sampler`: Token sampling strategies

### [Kernels](modules/kernels.md)
CUDA and Triton kernels for high-performance operations.

**Custom Kernels:**
- `decode_attention.py`: Optimized decode-phase attention
- `extend_attention.py`: Efficient prefill attention
- `rope.py`: Rotary Position Embedding

## Navigation

Each module documentation includes:
- **Overview**: Purpose and responsibilities
- **Architecture**: Internal structure and components
- **API Reference**: Classes, methods, and functions
- **Usage Examples**: Common patterns and code samples
- **Performance Notes**: Optimization tips and considerations

## Quick Reference

```python
# Import core components
from sfllm.engine import InferenceEngine
from sfllm.serving import EngineServer
from sfllm.models import Qwen3ForCausalLM

# Initialize inference engine
engine = InferenceEngine(
    model_path="Qwen/Qwen3-0.6B",
    dtype="float16",
    max_context_length=8192
)

# Start serving
from sfllm.serving.app import create_app
app = create_app(engine)
```

For detailed API documentation, see the individual module pages.