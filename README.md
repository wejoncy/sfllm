# Multimodal LLM Serving Framework for Gemma 3-4B-IT

This project implements a serving framework for Google's Gemma 3-4B-IT model that follows the OpenAI API protocol with support for both text and image inputs.

## Features

- Compatible with OpenAI API format
- Supports chat completions (`/v1/chat/completions`)
- Supports text completions (`/v1/completions`)
- Supports multimodal inputs (images + text)
- Custom endpoint for direct image uploads (`/v1/images_and_text`)
- Health check endpoint (`/health`)
- Optimized for Google's Gemma 3-4B-IT model
- Uses Google's vision models for image processing
- Performance optimization with CUDA Graphs for token generation
- Pre-allocated KV cache to reduce memory fragmentation

## Setup

1. Install the dependencies:
```
pip install -r requirements.txt
```

2. Run the server:
```
python app.py
```

The server will be available at http://localhost:8000

## API Usage

### Chat Completions

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-3-4b-it",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "temperature": 0.7,
    "max_tokens": 1024
  }'
```

### Text Completions

```bash
curl -X POST "http://localhost:8000/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-3-4b-it",
    "prompt": "Once upon a time",
    "temperature": 0.7,
    "max_tokens": 1024
  }'
```

### Multimodal Inputs (Images + Text)

```bash
curl -X POST "http://localhost:8000/v1/images_and_text" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@/path/to/your/image.jpg" \
  -F 'payload={
    "model": "google/gemma-3-4b-it",
    "text": "Describe the image",
    "temperature": 0.7,
    "max_tokens": 1024
  }'
```

## Docker Deployment

Build the Docker image:
```
docker build -t llm-serving .
```

Run the container:
```
docker run -p 8000:8000 llm-serving
```