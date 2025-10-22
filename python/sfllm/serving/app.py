#!/usr/bin/env python3
"""
Usage: python app.py --model <model_name_or_path> [--port <port_number>]
Starts the FastAPI server for LLM serving.
in the client
powershell
curl -Uri "http://127.0.0.1:8080/v1/completions" `   \
      -Method POST `    -Headers @{ "Content-Type" = "application/json" } `  \
      -Body '{"prompt": "how are you?","model":"1", "max_new_tokens":100}'

bash
`curl -X POST "http://127.0.0.1:8080/v1/completions"      -H "Content-Type: application/json"      -d '{"prompt": "how are you?", "model": "1", "max_new_tok
ens": 100}'`
"""
import multiprocessing as mp
from contextlib import asynccontextmanager
import fastapi
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn
import asyncio
import time
import argparse

from sfllm.serving.req_protocol import ChatRequest, CompletionRequest
from sfllm.serving.engine_server import EngineServer
from message_formatter import format_chat_messages
from config import REQUEST_TIMEOUT
from sfllm.server_args import ServerArgs

def create_app(server_args):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Load the ML model and start the queue processor
        inference_worker = EngineServer(server_args)
        app.state.inference_worker = inference_worker

        inference_worker.start()
        asyncio.get_running_loop().create_task(inference_worker.worker_response_loop())

        while not inference_worker.is_ready():
            await asyncio.sleep(0.1)
        yield
        print("Shutting down inference worker...")
        # Shutdown
        if inference_worker:
            await inference_worker.stop()

    app = FastAPI(title="LLM Serving Framework", lifespan=lifespan)

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Streaming response
    async def generate_stream(request_id, generator):
        import json
        async for chunk in generator:
            # Format streaming response according to OpenAI format
            stream_chunk = {
                "id": request_id,
                "object": "text_completion",
                "choices": [
                    {
                        "text": chunk.get("text", ""),
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": "stop"
                        if chunk.get("status", "RUNNING") in ["COMPLETED", "FAILED"]
                        else None,
                    }
                ],
            }

            # Convert to JSON string and add newline for proper streaming
            yield json.dumps(stream_chunk) + "\n\n"

    @app.post("/v1/chat/completions")
    async def chat_completions(request: fastapi.Request, body: ChatRequest):
        # Format messages into a prompt with processed images
        prompt = format_chat_messages(body.messages)
        
        # Prepare request data for the queue
        request_data = {
            "type": "chat",
            "prompt": prompt,
            "messages": body.messages,
            "temperature": body.temperature,
            "top_p": body.top_p,
            "max_new_tokens": body.max_new_tokens,
            "model": body.model,
            "stream": body.stream,
        }
        inference_worker = app.state.inference_worker
        
        # Submit to queue and wait for response
        request_id = await inference_worker.submit_request(request_data)

        try:
            if body.stream:
                return StreamingResponse(generate_stream(request_id, inference_worker.get_stream_response(request_id)), media_type="text/plain")
            else:
                # Non-streaming response
                async for _v in generate_stream(
                    request_id,
                    inference_worker.get_stream_response(
                        request_id, timeout=REQUEST_TIMEOUT
                    ),
                ):
                    response = _v

                return response
            
        except TimeoutError:
            raise HTTPException(
                status_code=504,
                detail="Request timed out. The model may be overloaded."
            )

    @app.post("/v1/completions")
    async def completions(request: fastapi.Request, body: CompletionRequest):
        # Prepare request data for the queue
        request_data = {
            "type": "completion",
            "prompt": body.prompt,
            "temperature": body.temperature,
            "top_p": body.top_p,
            "max_new_tokens": body.max_new_tokens,
            "model": body.model,
            "stream": body.stream,
        }
        
        # Submit to queue and wait for response
        try:
            inference_worker = app.state.inference_worker
            request_id = await inference_worker.submit_request(request_data)

            if body.stream:
                return StreamingResponse(
                    generate_stream(request_id,
                        inference_worker.get_stream_response(request_id, timeout=REQUEST_TIMEOUT),
                    ),
                    media_type="text/json",
                )
            else:
                # Non-streaming response
                async for _v in generate_stream(request_id,
                        inference_worker.get_stream_response(request_id, timeout=REQUEST_TIMEOUT),
                    ):
                    response = _v
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
            "timestamp": int(time.time())
        }
    return app

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    parser.add_argument("--port", type=int, default=8081, help="Port number")
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    app = create_app(server_args)
    uvicorn.run(app, host="0.0.0.0", port=args.port, reload=False)
