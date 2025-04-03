from fastapi import FastAPI, HTTPException, Request, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
import uvicorn
import json
import asyncio
import base64
from io import BytesIO
import os
from contextlib import asynccontextmanager
from serving.model_loader import load_model
from serving.text_generation import generate_text
from serving.image_processor import process_image, init_image_processor
from serving.models import ChatRequest, CompletionRequest

# Global model variable
OnServingModel = None

# Lifespan context manager for startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model and initialize resources
    global OnServingModel
    OnServingModel = load_model()
    
    # Create a temp directory for image uploads if it doesn't exist
    os.makedirs("temp_images", exist_ok=True)
    
    # Initialize the image processor
    await init_image_processor()
    
    yield  # This is where the app runs
    
    # Shutdown: Clean up resources
    # (No cleanup needed for now, but could be added here)

app = FastAPI(title="Multimodal LLM Serving Framework", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    if OnServingModel is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Format messages into a prompt with processed images
    prompt = ""
    image_embeddings = []
    
    # If model is Gemma, add special formatting - Gemma uses a specific chat template
    if "gemma" in request.model.lower():
        for msg in request.messages:
            if msg.role == "system":
                if isinstance(msg.content, str):
                    prompt += f"<start_of_turn>system\n{msg.content}<end_of_turn>\n"
                else:
                    # Handle content list for system message
                    system_content = ""
                    for item in msg.content:
                        if item.type == "text":
                            system_content += f"{item.text} "
                    prompt += f"<start_of_turn>system\n{system_content.strip()}<end_of_turn>\n"
            
            elif msg.role == "user":
                prompt += "<start_of_turn>user\n"
                if isinstance(msg.content, str):
                    prompt += f"{msg.content}"
                else:
                    # Handle content list for user message
                    for item in msg.content:
                        if item.type == "text":
                            prompt += f"{item.text} "
                        elif item.type == "image_url":
                            # Process image from URL
                            if "url" in item.image_url:
                                image_url = item.image_url["url"]
                                
                                # Check if it's a base64 image
                                if image_url.startswith("data:image/"):
                                    # Extract base64 data
                                    image_data = image_url.split(",")[1]
                                    image_embedding = await process_image(base64.b64decode(image_data))
                                else:
                                    # Regular URL
                                    image_embedding = await process_image(image_url)
                                    
                                image_embeddings.append(image_embedding)
                                prompt += "[IMAGE] "
                prompt += "<end_of_turn>\n"
            
            elif msg.role == "assistant":
                if isinstance(msg.content, str):
                    prompt += f"<start_of_turn>model\n{msg.content}<end_of_turn>\n"
                else:
                    # Handle content list for assistant message
                    asst_content = ""
                    for item in msg.content:
                        if item.type == "text":
                            asst_content += f"{item.text} "
                    prompt += f"<start_of_turn>model\n{asst_content.strip()}<end_of_turn>\n"
        
        prompt += "<start_of_turn>model\n"
    else:
        # Use the existing general formatting for other models
        for msg in request.messages:
            if msg.role == "system":
                if isinstance(msg.content, str):
                    prompt += f"System: {msg.content}\n"
                else:
                    # Handle content list for system message
                    for item in msg.content:
                        if item.type == "text":
                            prompt += f"System: {item.text}\n"
            
            elif msg.role == "user":
                if isinstance(msg.content, str):
                    prompt += f"User: {msg.content}\n"
                else:
                    # Handle content list for user message
                    prompt += "User: "
                    for item in msg.content:
                        if item.type == "text":
                            prompt += f"{item.text} "
                        elif item.type == "image_url":
                            # Process image from URL
                            if "url" in item.image_url:
                                image_url = item.image_url["url"]
                                
                                # Check if it's a base64 image
                                if image_url.startswith("data:image/"):
                                    # Extract base64 data
                                    image_data = image_url.split(",")[1]
                                    image_embedding = await process_image(base64.b64decode(image_data))
                                else:
                                    # Regular URL
                                    image_embedding = await process_image(image_url)
                                    
                                image_embeddings.append(image_embedding)
                                prompt += "[IMAGE] "
                    prompt += "\n"
            
            elif msg.role == "assistant":
                if isinstance(msg.content, str):
                    prompt += f"Assistant: {msg.content}\n"
                else:
                    # Handle content list for assistant message
                    prompt_part = "Assistant: "
                    for item in msg.content:
                        if item.type == "text":
                            prompt_part += f"{item.text} "
                    prompt += prompt_part + "\n"
        
        prompt += "Assistant: "
    
    # Generate text with image context if available
    response_text = await generate_text(
        model=OnServingModel,
        prompt=prompt,
        image_embeddings=image_embeddings if image_embeddings else None,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        is_gemma="gemma" in request.model.lower(),
        use_cuda_graph=True  # Enable CUDA Graph optimization
    )
    
    # Format the response according to OpenAI's format
    response = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": int(asyncio.get_event_loop().time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": len(prompt),
            "completion_tokens": len(response_text),
            "total_tokens": len(prompt) + len(response_text)
        }
    }
    
    return response

@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    if OnServingModel is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Generate text
    response_text = await generate_text(
        model=OnServingModel,
        prompt=request.prompt,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        is_gemma="gemma" in request.model.lower(),
        use_cuda_graph=True  # Enable CUDA Graph optimization
    )
    
    # Format the response according to OpenAI's format
    response = {
        "id": "cmpl-123",
        "object": "text_completion",
        "created": int(asyncio.get_event_loop().time()),
        "model": request.model,
        "choices": [
            {
                "text": response_text,
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": len(request.prompt),
            "completion_tokens": len(response_text),
            "total_tokens": len(request.prompt) + len(response_text)
        }
    }
    
    return response

@app.post("/v1/images_and_text")
async def images_and_text(
    prompt: str = Form(...),
    image: UploadFile = File(...),
    model: str = Form("google/gemma-3-4b-it"),
    temperature: float = Form(0.7),
    max_tokens: int = Form(1024)
):
    """Custom endpoint for direct image+text input"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Read and process the image
    image_content = await image.read()
    image_embedding = await process_image(image_content)
    
    # Format prompt for Gemma if using Gemma
    if "gemma" in model.lower():
        formatted_prompt = f"<start_of_turn>user\n{prompt} [IMAGE]<end_of_turn>\n<start_of_turn>model\n"
    else:
        formatted_prompt = prompt
    
    # Generate text with image context
    response_text = await generate_text(
        model=OnServingModel,
        prompt=formatted_prompt,
        image_embeddings=[image_embedding],
        temperature=temperature,
        max_tokens=max_tokens,
        is_gemma="gemma" in model.lower(),
        use_cuda_graph=True  # Enable CUDA Graph optimization
    )
    
    return {
        "response": response_text,
        "model": model
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
