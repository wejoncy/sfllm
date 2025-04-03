from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import time
from fastapi.encoders import jsonable_encoder
import argparse

from serving.req_protocol import ChatRequest, CompletionRequest
from serving.queue_management import QueueManager
from serving.inference_engine import InferenceWorker
from message_formatter import format_chat_messages
from config import REQUEST_TIMEOUT

# Global variables
queue_manager = QueueManager()
inference_worker = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model and start the queue processor
    global queue_manager, inference_worker
    
    # Create and start the inference worker
    inference_worker = InferenceWorker(queue_manager)
    await inference_worker.start()
    
    yield
    
    # Shutdown
    if inference_worker:
        await inference_worker.stop()


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
    # Format messages into a prompt with processed images
    prompt = format_chat_messages(request.messages)
    json_body = jsonable_encoder(request)
    
    # Prepare request data for the queue
    request_data = {
        "type": "chat",
        "prompt": prompt,
        "messages": request.messages,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "max_tokens": request.max_tokens,
        "model": request.model
    }
    
    # Submit to queue and wait for response
    try:
        request_id = queue_manager.submit_request(request_data)
        response = await queue_manager.get_response(request_id, timeout=REQUEST_TIMEOUT)
        if isinstance(response, dict) and "error" in response:
            raise HTTPException(
                status_code=500,
                detail=response.get("error", "Unknown error during inference")
            )
        
        response_text = response["text"]
        usage = response["usage"]
        # Format the response
        response = {
            "id": f"chatcmpl-{request_id[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request_data.get("model"),
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
            "usage": usage
        }

            
        return response
        
    except TimeoutError:
        raise HTTPException(
            status_code=504,
            detail="Request timed out. The model may be overloaded."
        )

@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    # Prepare request data for the queue
    request_data = {
        "type": "completion",
        "prompt": request.prompt,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "max_tokens": request.max_tokens,
        "model": request.model
    }
    
    # Submit to queue and wait for response
    try:
        request_id = queue_manager.submit_request(request_data)
        output = await queue_manager.get_response(request_id, timeout=REQUEST_TIMEOUT)
        if isinstance(output, dict) and "error" in output:
            raise HTTPException(
                status_code=500,
                detail=output.get("error", "Unknown error during inference")
            )
            
        response_text = output["text"]
        usage = output["usage"]
        # Format the response
        response = {
            "id": f"cmpl-{request_id[:8]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": request_data.get("model"),
            "choices": [
                {
                    "text": response_text,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop"
                }
            ],
            "usage": usage
        }
        

        return response
        
    except TimeoutError:
        raise HTTPException(
            status_code=504,
            detail="Request timed out. The model may be overloaded."
        )

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "queue_size": queue_manager.input_queue.qsize(),
        "workers": len(inference_worker.worker_threads) if inference_worker else 0
    }

if __name__ == "__main__":
    # Parse command line arguments
    argparser = argparse.ArgumentParser(description="Gemma Model Client")
    argparser.add_argument("--port", type=int, default=8080, help="Port number")
    args = argparser.parse_args()
    uvicorn.run("app:app", host="0.0.0.0", port=args.port, reload=False)
