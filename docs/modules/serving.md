# Serving Module

The Serving module provides a FastAPI-based web server with OpenAI-compatible REST APIs, enabling easy integration with existing applications and tools.

## Overview

The Serving module offers:

- **OpenAI-Compatible APIs**: Drop-in replacement for OpenAI API endpoints
- **Streaming Support**: Real-time token streaming via Server-Sent Events
- **Multiprocess Architecture**: Separate inference process for better isolation
- **Built-in Client**: Python client library for easy integration
- **Performance Monitoring**: Built-in benchmarking and metrics

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Client Layer                           │
├─────────────────────────────────────────────────────────────┤
│  HTTP Clients  │  Python Client  │  OpenAI SDK  │  curl    │
└─────────────────────────────────────────────────────────────┘
                               │
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Server (app.py)                  │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  /v1/chat/  │  │/v1/complete │  │ /v1/models  │        │
│  │completions  │  │   tions     │  │             │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
│  ┌─────────────────────────────────────────────────────────┤
│  │              Message Processing                         │
│  │  • Request validation  • Response formatting           │
│  └─────────────────────────────────────────────────────────┘
└─────────────────────────────────────────────────────────────┘
                               │
┌─────────────────────────────────────────────────────────────┐
│               EngineServer (engine_server.py)               │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┤
│  │            Multiprocess Communication                   │
│  │  • Request queuing  • Response streaming  • Heartbeat  │
│  └─────────────────────────────────────────────────────────┘
└─────────────────────────────────────────────────────────────┘
                               │
┌─────────────────────────────────────────────────────────────┐
│                InferenceEngine (Separate Process)           │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### FastAPI Application (app.py)

The main web server providing OpenAI-compatible endpoints.

**Application Setup:**
```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="SFLLM API Server",
    description="High-performance LLM serving with OpenAI compatibility",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for web applications
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Initialize engine server
engine_server = EngineServer(server_args)
```

**Chat Completions Endpoint:**
```python
@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest) -> Union[ChatCompletionResponse, StreamingResponse]:
    """Create a chat completion.
    
    Compatible with OpenAI's Chat Completions API.
    Supports both streaming and non-streaming responses.
    """
    
    # Convert chat messages to prompt
    prompt = format_chat_prompt(request.messages, request.model)
    
    # Create sampling parameters
    sampling_params = SamplingParams(
        max_new_tokens=request.max_tokens or 256,
        temperature=request.temperature or 1.0,
        top_p=request.top_p or 1.0,
        frequency_penalty=request.frequency_penalty or 0.0,
        presence_penalty=request.presence_penalty or 0.0,
        stop=request.stop
    )
    
    # Submit request to engine
    request_id = await engine_server.add_request(
        prompt=prompt,
        sampling_params=sampling_params
    )
    
    if request.stream:
        return StreamingResponse(
            stream_chat_completion(request_id, request),
            media_type="text/plain"
        )
    else:
        return await generate_chat_completion(request_id, request)
```

**Text Completions Endpoint:**
```python
@app.post("/v1/completions")
async def create_completion(request: CompletionRequest) -> Union[CompletionResponse, StreamingResponse]:
    """Create a text completion.
    
    Compatible with OpenAI's Completions API.
    """
    
    sampling_params = SamplingParams(
        max_new_tokens=request.max_tokens or 256,
        temperature=request.temperature or 1.0,
        top_p=request.top_p or 1.0,
        n=request.n or 1,
        stop=request.stop,
        echo=request.echo or False
    )
    
    request_id = await engine_server.add_request(
        prompt=request.prompt,
        sampling_params=sampling_params
    )
    
    if request.stream:
        return StreamingResponse(
            stream_completion(request_id, request),
            media_type="text/plain"
        )
    else:
        return await generate_completion(request_id, request)
```

**Models Endpoint:**
```python
@app.get("/v1/models")
async def list_models() -> ModelListResponse:
    """List available models."""
    
    model_info = await engine_server.get_model_info()
    
    return ModelListResponse(
        object="list",
        data=[
            ModelInfo(
                id=model_info["name"],
                object="model",
                created=int(time.time()),
                owned_by="sfllm",
                permission=[],
                root=model_info["name"],
                parent=None
            )
        ]
    )
```

### Streaming Implementation

**Server-Sent Events:**
```python
async def stream_chat_completion(request_id: str, request: ChatCompletionRequest):
    """Stream chat completion chunks using Server-Sent Events."""
    
    try:
        async for chunk in engine_server.generate_stream(request_id):
            # Format as OpenAI-compatible streaming response
            response_chunk = ChatCompletionStreamResponse(
                id=request_id,
                object="chat.completion.chunk",
                created=int(time.time()),
                model=request.model,
                choices=[
                    StreamChoice(
                        index=0,
                        delta=StreamDelta(content=chunk.token),
                        finish_reason=chunk.finish_reason
                    )
                ]
            )
            
            # Send as Server-Sent Event
            yield f"data: {response_chunk.json()}\n\n"
            
            if chunk.finish_reason:
                break
    
    except Exception as e:
        # Send error as SSE
        error_chunk = {
            "error": {
                "message": str(e),
                "type": "server_error"
            }
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
    
    finally:
        yield "data: [DONE]\n\n"
```

**Client-Side Streaming:**
```javascript
// JavaScript client example
const response = await fetch('/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        model: 'qwen3-0.6b',
        messages: [{ role: 'user', content: 'Hello!' }],
        stream: true
    })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    
    const chunk = decoder.decode(value);
    const lines = chunk.split('\n');
    
    for (const line of lines) {
        if (line.startsWith('data: ')) {
            const data = line.slice(6);
            if (data === '[DONE]') return;
            
            try {
                const parsed = JSON.parse(data);
                const content = parsed.choices[0]?.delta?.content;
                if (content) {
                    console.log(content); // Process token
                }
            } catch (e) {
                console.error('Parse error:', e);
            }
        }
    }
}
```

### EngineServer (engine_server.py)

Manages communication between the web server and inference engine.

**Multiprocess Architecture:**
```python
import multiprocessing as mp
from multiprocessing import Queue, Process
import asyncio
import threading

class EngineServer:
    def __init__(self, server_args: ServerArgs):
        # Force spawn method for CUDA compatibility
        if mp.get_start_method(allow_none=True) != 'spawn':
            mp.set_start_method('spawn', force=True)
        
        self.server_args = server_args
        self.request_queue = Queue()
        self.response_queues = {}
        self.engine_process = None
        
        # Start inference engine in separate process
        self.start_engine_process()
```

**Request Processing:**
```python
async def add_request(
    self, 
    prompt: str, 
    sampling_params: SamplingParams
) -> str:
    """Add a request to the engine queue."""
    
    request_id = str(uuid.uuid4())
    
    # Create response queue for this request
    response_queue = Queue()
    self.response_queues[request_id] = response_queue
    
    # Send request to engine process
    request = {
        'type': 'generate',
        'request_id': request_id,
        'prompt': prompt,
        'sampling_params': sampling_params.dict()
    }
    
    self.request_queue.put(request)
    return request_id

async def generate_stream(self, request_id: str) -> AsyncIterator[GenerationChunk]:
    """Stream generation results for a request."""
    
    if request_id not in self.response_queues:
        raise ValueError(f"Request {request_id} not found")
    
    response_queue = self.response_queues[request_id]
    
    # Poll queue in separate thread to avoid blocking
    def get_response():
        try:
            return response_queue.get(timeout=1.0)
        except:
            return None
    
    while True:
        # Run queue polling in thread pool
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, get_response)
        
        if response is None:
            await asyncio.sleep(0.01)  # Small delay
            continue
        
        if response['type'] == 'token':
            yield GenerationChunk(
                token=response['token'],
                finish_reason=None
            )
        elif response['type'] == 'finished':
            yield GenerationChunk(
                token="",
                finish_reason=response['finish_reason']
            )
            break
        elif response['type'] == 'error':
            raise RuntimeError(response['message'])
```

**Engine Process Management:**
```python
def start_engine_process(self):
    """Start the inference engine in a separate process."""
    
    self.engine_process = Process(
        target=self._run_engine,
        args=(self.server_args, self.request_queue, self.response_queues)
    )
    self.engine_process.start()

def _run_engine(
    self, 
    server_args: ServerArgs, 
    request_queue: Queue,
    response_queues: Dict[str, Queue]
):
    """Run inference engine in separate process."""
    
    # Initialize engine
    engine = InferenceEngine(
        model_path=server_args.model,
        dtype=server_args.dtype,
        max_context_length=server_args.max_context_length,
        cuda_graph_max_bs=server_args.cuda_graph_max_bs
    )
    
    # Process requests
    while True:
        try:
            request = request_queue.get(timeout=1.0)
            
            if request['type'] == 'generate':
                self._handle_generate_request(engine, request, response_queues)
            elif request['type'] == 'shutdown':
                break
                
        except Exception as e:
            continue  # Timeout, continue polling
```

### Client Library (client.py)

Python client for easy integration with SFLLM server.

**Basic Client:**
```python
import httpx
import json
from typing import AsyncIterator, Optional

class SFLLMClient:
    def __init__(self, base_url: str = "http://localhost:8081"):
        self.base_url = base_url
        self.client = httpx.AsyncClient()
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "qwen3-0.6b",
        stream: bool = False,
        **kwargs
    ):
        """Create a chat completion."""
        
        request_data = {
            "model": model,
            "messages": messages,
            "stream": stream,
            **kwargs
        }
        
        if stream:
            return self._stream_chat_completion(request_data)
        else:
            response = await self.client.post(
                f"{self.base_url}/v1/chat/completions",
                json=request_data
            )
            return response.json()
    
    async def _stream_chat_completion(self, request_data: dict) -> AsyncIterator[dict]:
        """Stream chat completion responses."""
        
        async with self.client.stream(
            "POST",
            f"{self.base_url}/v1/chat/completions",
            json=request_data
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]  # Remove "data: " prefix
                    
                    if data == "[DONE]":
                        break
                    
                    try:
                        yield json.loads(data)
                    except json.JSONDecodeError:
                        continue
```

**Usage Examples:**
```python
# Non-streaming chat
client = SFLLMClient("http://localhost:8081")

response = await client.chat_completion(
    messages=[
        {"role": "user", "content": "What is the capital of France?"}
    ],
    model="qwen3-0.6b",
    max_tokens=100
)

print(response["choices"][0]["message"]["content"])

# Streaming chat
async for chunk in client.chat_completion(
    messages=[
        {"role": "user", "content": "Tell me a story"}
    ],
    stream=True
):
    if chunk["choices"][0]["delta"].get("content"):
        print(chunk["choices"][0]["delta"]["content"], end="")
```

### Message Formatting (message_formatter.py)

Converts chat messages to model-specific prompt formats.

**Chat Template System:**
```python
class MessageFormatter:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.template = self._get_chat_template(model_name)
    
    def format_messages(self, messages: List[Dict[str, str]]) -> str:
        """Convert chat messages to prompt string."""
        
        if self.model_name.startswith("qwen"):
            return self._format_qwen_messages(messages)
        elif self.model_name.startswith("gemma"):
            return self._format_gemma_messages(messages)
        else:
            return self._format_generic_messages(messages)
    
    def _format_qwen_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format messages for Qwen models."""
        
        formatted_parts = []
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                formatted_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
            elif role == "user":
                formatted_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == "assistant":
                formatted_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
        
        # Add assistant prompt
        formatted_parts.append("<|im_start|>assistant\n")
        
        return "\n".join(formatted_parts)
```

### Benchmarking Tools (benchmark.py)

Performance testing and load generation tools.

**Benchmark Runner:**
```python
import asyncio
import aiohttp
import time
import statistics
from dataclasses import dataclass
from typing import List

@dataclass
class BenchmarkResult:
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_time: float
    avg_latency: float
    p50_latency: float
    p95_latency: float
    p99_latency: float
    throughput: float

class SFLLMBenchmark:
    def __init__(self, base_url: str = "http://localhost:8081"):
        self.base_url = base_url
    
    async def run_benchmark(
        self,
        concurrency: int = 10,
        requests: int = 100,
        prompt: str = "The future of AI is",
        max_tokens: int = 50
    ) -> BenchmarkResult:
        """Run benchmark with specified parameters."""
        
        semaphore = asyncio.Semaphore(concurrency)
        latencies = []
        successful = 0
        failed = 0
        
        start_time = time.time()
        
        async def single_request():
            nonlocal successful, failed
            
            async with semaphore:
                request_start = time.time()
                
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            f"{self.base_url}/v1/completions",
                            json={
                                "model": "qwen3-0.6b",
                                "prompt": prompt,
                                "max_tokens": max_tokens
                            }
                        ) as response:
                            await response.json()
                            successful += 1
                            latencies.append(time.time() - request_start)
                            
                except Exception as e:
                    failed += 1
        
        # Execute all requests
        tasks = [single_request() for _ in range(requests)]
        await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        
        # Calculate statistics
        return BenchmarkResult(
            total_requests=requests,
            successful_requests=successful,
            failed_requests=failed,
            total_time=total_time,
            avg_latency=statistics.mean(latencies) if latencies else 0,
            p50_latency=statistics.median(latencies) if latencies else 0,
            p95_latency=statistics.quantiles(latencies, n=20)[18] if latencies else 0,
            p99_latency=statistics.quantiles(latencies, n=100)[98] if latencies else 0,
            throughput=successful / total_time if total_time > 0 else 0
        )
```

**Command Line Interface:**
```bash
# Basic benchmark
python python/sfllm/serving/benchmark.py \
  --url http://localhost:8081 \
  --concurrency 10 \
  --requests 100

# Advanced benchmark
python python/sfllm/serving/benchmark.py \
  --url http://localhost:8081 \
  --concurrency 1,5,10,20,50 \
  --requests 200 \
  --prompt "Write a short story about" \
  --max-tokens 100 \
  --output-file benchmark_results.json
```

## Configuration

### Server Arguments

```python
@dataclass
class ServerArgs:
    model: str = "Qwen/Qwen3-0.6B"
    host: str = "0.0.0.0" 
    port: int = 8081
    dtype: str = "auto"
    max_context_length: int = 8192
    max_running_tokens: int = 16384
    cuda_graph_max_bs: int = 32
    log_level: str = "INFO"
    
    # Performance settings
    disable_cuda_graph: bool = False
    enable_chunked_prefill: bool = False
    max_batch_size: int = 32
    
    # API settings
    enable_cors: bool = True
    api_key: Optional[str] = None
    allow_credentials: bool = True
```

### Production Deployment

**Uvicorn Configuration:**
```python
import uvicorn

# High-performance settings
uvicorn.run(
    "sfllm.serving.app:app",
    host="0.0.0.0",
    port=8081,
    workers=1,  # Single worker for GPU sharing
    loop="uvloop",  # Faster event loop on Linux
    access_log=False,  # Disable for performance
    server_header=False,
    date_header=False
)
```

**Docker Deployment:**
```dockerfile
FROM nvidia/cuda:11.8-runtime-ubuntu20.04

# Install SFLLM
COPY . /app
WORKDIR /app
RUN pip install -e .

# Expose port
EXPOSE 8081

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
  CMD curl -f http://localhost:8081/health || exit 1

# Start server
CMD ["python", "python/sfllm/serving/app.py", "--model", "Qwen/Qwen3-0.6B"]
```

The Serving module provides a complete, production-ready web API for SFLLM with OpenAI compatibility, streaming support, and comprehensive monitoring capabilities.