import requests
import time
import json
from typing import Dict, Optional


def generate(        
        prompt: str, 
        model: str = "gemma-3-4b-it", 
        max_tokens: int = 64,
        temperature: float = 0.8,
        top_p: float = 0.95,
        stream: bool = False):
    payload = {
        "model": model,
        "messages":[
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt}
            ]
        }
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "stream": stream
        
    }
    session = requests.Session()
    response = session.post(
        f"http://localhost:8000/v1/chat/completions",
        json=payload
    )
    response.raise_for_status()
    result = response.json()
    return result.get("text", "") if "text" in result else result

# Example usage
if __name__ == "__main__":
    
    # Generate text (non-streaming)
    prompt = "Explain quantum computing in simple terms"
    start = time.time()
    response = generate(prompt)
    end = time.time()
    print(f"Time taken: {end - start:.2f} seconds")
    print(f"Response: {response}")
    
    print()
