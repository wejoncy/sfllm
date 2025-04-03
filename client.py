import requests
import time
import json
from typing import Dict, Optional
import argparse


class Client:
    def __init__(self, host: str):
        self.host = host

    def generate(self, prompt: str, model: str = "gemma-3-4b-it", max_tokens: int = 64,
                 temperature: float = 0.8, top_p: float = 0.95, stream: bool = False) -> Dict:
        payload = {
            "model": model,
            "messages": [
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
            f"{self.host}/v1/chat/completions",
            json=payload
        )
        response.raise_for_status()
        result = response.json()
        return result.get("text", "") if "text" in result else result
# Example usage
if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Gemma Model Client")
    argparser.add_argument("--host", type=str, default=f"http://104.208.77.11:8080", help="Host URL")
    args = argparser.parse_args()
    client = Client(host=args.host)
    # Generate text (non-streaming)
    prompt = "Explain quantum computing in simple terms"
    start = time.time()
    response = client.generate(prompt)
    end = time.time()
    print(f"Time taken: {end - start:.2f} seconds")
    print(f"Response: {response}")
    
    print()
