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

streaming
bash
`
curl http://localhost:30000/generate   -H "Content-Type: application/json"   -d '{"text": "how are you?","stream": true,"sampling_params": { "max_new_tokens": 160, "temperature": 1 }}'
`
"""
import multiprocessing as mp
from contextlib import asynccontextmanager
import fastapi
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn
import asyncio
import json

import argparse

from sfllm.serving.req_protocol import ChatRequest, CompletionRequest, GenerateReqInput
from sfllm.serving.engine_server import EngineServer
from sfllm.server_args import ServerArgs
from sfllm.version import __version__
from typing import AsyncIterator

def create_app(server_args):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Load the ML model and start the queue processor
        inference_worker = EngineServer(server_args)
        app.state.inference_worker = inference_worker

        inference_worker.start()
        asyncio.get_running_loop().create_task(inference_worker.worker_response_loop())
        asyncio.get_running_loop().create_task(inference_worker.auto_clean_resource_loop())

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

    def format_OPENAI_complete(response):
        # Format streaming response according to OpenAI format
        stream_chunk = {
            "id": response['request_id'],
            "object": "text_completion",
            "choices": [
                {
                    "text": response.get("text", ""),
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop"
                    if response.get("status", "RUNNING") in ["COMPLETED", "FAILED"]
                    else None,
                }
            ],
        }

        # Convert to JSON string and add newline for proper streaming
        return stream_chunk

    async def generate_response(worker, req: GenerateReqInput, endpoint: str):
        request_id:int = await worker.submit_request(req)

        if req.stream:
            async def stream_results() -> AsyncIterator[bytes]:
                try:
                    async for out in worker.get_response(request_id, streaming=True):
                        yield (b"data: " + json.dumps(out).encode("utf-8") + b"\n\n")
                except ValueError as e:
                    out = {"error": {"message": str(e)}, 'request_id': request_id}
                    if endpoint == "/v1/chat/completions" or endpoint == "/v1/completions":
                        yield (
                            b"data: "
                            + json.dumps(format_OPENAI_complete(out)).encode("utf-8")
                            + b"\n\n"
                        )
                    else:
                        yield (b"data: " + json.dumps(out).encode("utf-8") + b"\n\n")
                yield b"data: [DONE]\n\n"

            background_tasks = fastapi.BackgroundTasks()
            return StreamingResponse(
                stream_results(),
                media_type="text/event-stream",
                background=worker.create_abort_task(request_id, background_tasks),
            )
        else:
            try:
                # Non-streaming response
                async for _v in worker.get_response(request_id):
                    response = _v
            except ValueError as e:
                    response = {"error": {"message": str(e)}, 'request_id': request_id}
            if endpoint == "/v1/chat/completions" or endpoint == "/v1/completions":
                response = format_OPENAI_complete(response)
            return response


    @app.api_route("/generate", methods=["POST", "PUT"])
    async def generate_request(request: fastapi.Request, body: GenerateReqInput):
        """Handle a generate request."""
        inference_worker = app.state.inference_worker
        return await generate_response(inference_worker, body, "/generate")

    @app.post("/v1/chat/completions")
    async def chat_completions(request: fastapi.Request, body: ChatRequest):
        # Format messages into a prompt with processed images
        req = GenerateReqInput.from_basemodel(body)
        inference_worker = app.state.inference_worker
        # Submit to queue and wait for response

        return await generate_response(inference_worker, req, "/v1/chat/completions")

    @app.post("/v1/completions")
    async def completions(request: fastapi.Request, body: CompletionRequest):
        # Submit to queue and wait for response
        inference_worker = app.state.inference_worker
        req = GenerateReqInput.from_basemodel(base_model=body)

        return await generate_response(inference_worker, req, "/v1/completions")

    @app.get("/health")
    async def health():
        return fastapi.responses.Response(status_code=200)

    @app.get("/get_model_info")
    async def get_model_info():
        """Get the model information."""
        result = {
            "model_path": server_args.model_path,
        }
        return result

    @app.get("/get_server_info")
    async def get_server_info():
        # Returns interna states per DP.
        internal_states= [{}]
        return {
            "internal_states": internal_states,
            "version": __version__,
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
