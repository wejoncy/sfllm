# Multimodal LLM Serving Framework for Gemma 3-4B-IT

This project implements a serving framework for Google's Gemma 3-4B-IT model that follows the OpenAI API protocol with support for both text and image inputs.

## Features

- Compatible with OpenAI API format
- Supports chat completions (`/v1/chat/completions`)

## Setup

1. Install the dependencies:
```
pip install -r requirements.txt
```

2. Run the server:
```
python app.py
```

The server will be available at http://localhost:6006

## API Usage

### Chat Completions

```bash
curl -X POST "http://localhost:6006/v1/chat/completions" \
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