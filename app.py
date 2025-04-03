from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import asyncio
from fastapi.encoders import jsonable_encoder
from contextlib import asynccontextmanager
from serving.model_loader import load_model
from serving.text_generation import generate_text
from serving.models import ChatRequest, CompletionRequest

# Global model variable
OnServingModel = load_model()   

app = FastAPI(title="Multimodal LLM Serving Framework")

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
    json_body = jsonable_encoder(request)

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
                    else:
                        assert False, "Image URLs are not supported in this version"
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

    
    # Generate text with image context if available
    response_text = await generate_text(
        model=OnServingModel,
        prompt=prompt,
        messages=json_body['messages'],
        image_embeddings=None,
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
    }
    
    return response

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=6006, reload=False)
