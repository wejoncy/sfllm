from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import asyncio
from fastapi.encoders import jsonable_encoder
from serving.model_loader import ForwardModel
from serving.text_generation import generate_text
from serving.req_protocol import ChatRequest, CompletionRequest
import argparse

# Global model variable
OnServingModel = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    global OnServingModel
    OnServingModel = ForwardModel()
    yield


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

    inputs = OnServingModel.tokenize(prompt, request.messages)

    # Generate text with image context if available
    engine_out = await generate_text(
        model=OnServingModel,
        inputs=inputs,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        # use_cuda_graph=True  # Enable CUDA Graph optimization
    )
    response_text = engine_out["text"]
    usage = engine_out["usage"]
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
        "usage":usage
    }
    
    return response

@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    if OnServingModel is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    inputs = OnServingModel.tokenize(request.prompt)
    # Generate text
    engine_out = await generate_text(
        model=OnServingModel,
        inputs=inputs,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        # use_cuda_graph=True  # Enable CUDA Graph optimization
    )
    response_text = engine_out["text"]
    usage = engine_out["usage"]
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
        "usage":usage
    }
    
    return response


@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    # Parse command line arguments
    argparser = argparse.ArgumentParser(description="Gemma Model Client")
    argparser.add_argument("--port", type=int, default=6006, help="Port number")
    args = argparser.parse_args()
    uvicorn.run("app:app", host="0.0.0.0", port=args.port, reload=False)
