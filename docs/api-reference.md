# API Reference

SFLLM provides OpenAI-compatible REST APIs for text generation. All endpoints support both streaming and non-streaming responses.

## Base URL

```
http://localhost:8081
```

## Authentication

Currently, SFLLM does not require authentication. For production deployments, consider adding authentication via a reverse proxy.

## Common Parameters

All endpoints share these common parameters:

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `model` | string | Model identifier | Required |
| `temperature` | float | Sampling temperature (0.0-2.0) | 0.7 |
| `top_p` | float | Nucleus sampling parameter | 1.0 |
| `max_new_tokens` | integer | Maximum tokens to generate | 128 |
| `stream` | boolean | Enable streaming response | false |

## Endpoints

### Health Check

Check server status and readiness.

```http
GET /health
```

#### Response

```json
{
  "status": "healthy",
  "timestamp": 1698765432
}
```

---

### Chat Completions

Generate responses in a conversational format.

```http
POST /v1/chat/completions
```

#### Request Body

```json
{
  "model": "string",
  "messages": [
    {
      "role": "system|user|assistant",
      "content": "string"
    }
  ],
  "stream": false,
  "temperature": 0.7,
  "top_p": 1.0,
  "max_new_tokens": 128
}
```

#### Parameters

| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| `model` | string | Model identifier | ✅ |
| `messages` | array | Conversation history | ✅ |
| `messages[].role` | string | Message role: `system`, `user`, or `assistant` | ✅ |
| `messages[].content` | string | Message content | ✅ |
| `stream` | boolean | Enable streaming response | ❌ |
| `temperature` | float | Controls randomness (0.0-2.0) | ❌ |
| `top_p` | float | Nucleus sampling threshold | ❌ |
| `max_new_tokens` | integer | Maximum tokens to generate (1-4096) | ❌ |

#### Non-Streaming Response

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1698765432,
  "model": "qwen3-0.6b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Generated response text"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 25,
    "total_tokens": 37
  }
}
```

#### Streaming Response

When `stream: true`, the response is sent as Server-Sent Events:

```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1698765432,"model":"qwen3-0.6b","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1698765432,"model":"qwen3-0.6b","choices":[{"index":0,"delta":{"content":" there!"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1698765432,"model":"qwen3-0.6b","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

#### Example Requests

**Basic Chat Completion:**
```bash
curl -X POST "http://localhost:8081/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-0.6b",
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "max_new_tokens": 50
  }'
```

**Streaming Chat:**
```bash
curl -X POST "http://localhost:8081/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-0.6b",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Tell me a joke"}
    ],
    "stream": true,
    "max_new_tokens": 100
  }'
```

---

### Text Completions

Generate text completions from a prompt.

```http
POST /v1/completions
```

#### Request Body

```json
{
  "model": "string",
  "prompt": "string",
  "stream": false,
  "temperature": 0.7,
  "top_p": 1.0,
  "max_new_tokens": 128
}
```

#### Parameters

| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| `model` | string | Model identifier | ✅ |
| `prompt` | string | Input text prompt | ✅ |
| `stream` | boolean | Enable streaming response | ❌ |
| `temperature` | float | Controls randomness (0.0-2.0) | ❌ |
| `top_p` | float | Nucleus sampling threshold | ❌ |
| `max_new_tokens` | integer | Maximum tokens to generate (1-4096) | ❌ |

#### Non-Streaming Response

```json
{
  "id": "cmpl-abc123",
  "object": "text_completion",
  "created": 1698765432,
  "model": "qwen3-0.6b",
  "choices": [
    {
      "text": "Generated completion text",
      "index": 0,
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 8,
    "completion_tokens": 20,
    "total_tokens": 28
  }
}
```

#### Streaming Response

```
data: {"id":"cmpl-abc123","object":"text_completion","created":1698765432,"model":"qwen3-0.6b","choices":[{"text":"Generated","index":0,"logprobs":null,"finish_reason":null}]}

data: {"id":"cmpl-abc123","object":"text_completion","created":1698765432,"model":"qwen3-0.6b","choices":[{"text":" text","index":0,"logprobs":null,"finish_reason":null}]}

data: {"id":"cmpl-abc123","object":"text_completion","created":1698765432,"model":"qwen3-0.6b","choices":[{"text":"","index":0,"logprobs":null,"finish_reason":"stop"}]}

data: [DONE]
```

#### Example Requests

**Basic Completion:**
```bash
curl -X POST "http://localhost:8081/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-0.6b",
    "prompt": "The future of artificial intelligence is",
    "max_new_tokens": 100,
    "temperature": 0.8
  }'
```

**Streaming Completion:**
```bash
curl -X POST "http://localhost:8081/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-0.6b",
    "prompt": "Once upon a time",
    "stream": true,
    "max_new_tokens": 200
  }'
```

## Error Responses

All endpoints return errors in this format:

```json
{
  "error": {
    "message": "Error description",
    "type": "invalid_request_error",
    "code": "validation_error"
  }
}
```

### Common Error Codes

| Status Code | Error Type | Description |
|-------------|------------|-------------|
| 400 | `invalid_request_error` | Malformed request or invalid parameters |
| 422 | `validation_error` | Request validation failed |
| 500 | `internal_server_error` | Server error during processing |
| 504 | `timeout_error` | Request processing timeout |

### Example Error Response

```bash
# Invalid temperature value
curl -X POST "http://localhost:8081/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-0.6b",
    "messages": [{"role": "user", "content": "Hello"}],
    "temperature": 5.0
  }'

# Response (400 Bad Request)
{
  "error": {
    "message": "Temperature must be between 0.0 and 2.0",
    "type": "invalid_request_error",
    "code": "validation_error"
  }
}
```

## Rate Limiting

Currently, SFLLM does not implement rate limiting. For production use, consider implementing rate limiting at the reverse proxy level.

## SDKs and Libraries

### Python

Using the OpenAI Python client:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8081/v1",
    api_key="not-needed"
)

# Chat completion
response = client.chat.completions.create(
    model="qwen3-0.6b",
    messages=[
        {"role": "user", "content": "Hello, world!"}
    ],
    max_tokens=100
)

print(response.choices[0].message.content)

# Streaming chat
stream = client.chat.completions.create(
    model="qwen3-0.6b",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True,
    max_tokens=200
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
```

### JavaScript/Node.js

```javascript
const OpenAI = require('openai');

const client = new OpenAI({
  baseURL: 'http://localhost:8081/v1',
  apiKey: 'not-needed'
});

async function chatCompletion() {
  const response = await client.chat.completions.create({
    model: 'qwen3-0.6b',
    messages: [
      { role: 'user', content: 'Hello, world!' }
    ],
    max_tokens: 100
  });
  
  console.log(response.choices[0].message.content);
}
```

### cURL Examples

See the individual endpoint documentation above for cURL examples.

## WebSocket Support

SFLLM currently does not support WebSocket connections. All streaming is handled via HTTP Server-Sent Events (SSE).

## Changelog

- **v0.1.0**: Initial API implementation with chat completions and text completions
- Support for streaming and non-streaming responses
- OpenAI-compatible request/response formats