#!/usr/bin/env python3
"""
Test script for streaming responses
"""

import requests
import json

def test_streaming_chat():
    """Test streaming chat completions"""
    print("Testing streaming chat completions...")
    
    url = "http://localhost:8081/v1/chat/completions"
    data = {
        "model": "test-model",
        "messages": [
            {"role": "user", "content": "Tell me a short story about a robot."}
        ],
        "stream": True,
        "max_new_tokens": 100
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, json=data, headers=headers, stream=True)
        
        if response.status_code == 200:
            print("Stream started successfully!")
            print("Response: ", end='', flush=True)
            position=0
            # The server returns JSON objects directly, not SSE format
            for line in response.iter_lines(decode_unicode=True):
                if line.strip():  # Skip empty lines
                    try:
                        chunk = json.loads(line)
                        if 'choices' in chunk and len(chunk['choices']) > 0:
                            text = chunk['choices'][0].get('text', '')
                            if text:
                                print(text[position:], end="", flush=True)
                                position= len(text)
                            
                            # Check if this is the final chunk
                            finish_reason = chunk['choices'][0].get('finish_reason')
                            if finish_reason == 'stop':
                                print("\nStream completed!")
                                break
                    except json.JSONDecodeError as e:
                        print(f"\nFailed to parse JSON: {line}")
                        print(f"Error: {e}")
            print()
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Error testing streaming: {e}")

def test_non_streaming_chat():
    """Test non-streaming chat completions"""
    print("Testing non-streaming chat completions...")
    
    url = "http://localhost:8081/v1/chat/completions"
    data = {
        "model": "test-model",
        "messages": [
            {"role": "user", "content": "Hello, how are you?"}
        ],
        "stream": False,
        "max_new_tokens": 50
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, json=data, headers=headers)
        
        if response.status_code == 200:
            result = response.json()
            # If the result is a string, parse it as JSON
            if isinstance(result, str):
                result = json.loads(result.strip())
            print("Non-streaming response:")
            print(json.dumps(result, indent=2))
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Error testing non-streaming: {e}")

def test_streaming_completions():
    """Test streaming completions"""
    print("Testing streaming completions...")
    
    url = "http://localhost:8081/v1/completions"
    data = {
        "model": "test-model",
        "prompt": "Once upon a time in a galaxy far, far away",
        "stream": True,
        "max_new_tokens": 100
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, json=data, headers=headers, stream=True)
        
        if response.status_code == 200:
            print("Completion stream started successfully!")
            print("Response: ", end='', flush=True)
            
            # The server returns JSON objects directly, not SSE format
            position = 0
            for line in response.iter_lines(decode_unicode=True):
                if line.strip():  # Skip empty lines
                    try:
                        chunk = json.loads(line)
                        if 'choices' in chunk and len(chunk['choices']) > 0:
                            text = chunk['choices'][0].get('text', '')
                            if text:
                                print(text[position:], end="", flush=True)
                                position = len(text)

                            # Check if this is the final chunk
                            finish_reason = chunk['choices'][0].get('finish_reason')
                            if finish_reason == 'stop':
                                print("\nStream completed!")
                                break
                    except json.JSONDecodeError as e:
                        print(f"\nFailed to parse JSON: {line}")
                        print(f"Error: {e}")
            print()
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Error testing streaming completions: {e}")

if __name__ == "__main__":
    print("Starting streaming API tests...\n")
    
    # Test health endpoint first
    try:
        health_response = requests.get("http://localhost:8081/health")
        if health_response.status_code == 200:
            print("✅ Server is healthy")
        else:
            print("❌ Server health check failed")
            exit(1)
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        exit(1)
    
    print("\n" + "="*50)
    test_non_streaming_chat()
    
    print("\n" + "="*50)
    test_streaming_chat()
    
    print("\n" + "="*50)
    test_streaming_completions()
    
    print("\n🎉 All tests completed!")