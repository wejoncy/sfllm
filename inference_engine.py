import asyncio
import queue
import threading
import time
from typing import Dict, Any, List

from serving.model_loader import ForwardModel
from serving.text_generation import generate_text

class InferenceWorker:
    """Worker that processes inference requests from a queue."""
    
    def __init__(self, queue_manager):
        """
        Initialize the inference worker.
        
        Args:
            queue_manager: The QueueManager instance
        """
        self.queue_manager = queue_manager
        self.model = ForwardModel()
        self.running = False
        self.worker_threads = []
        self.response_processor = None
    
    async def process_request(self, request_id: str, request_data: Dict[str, Any]):
        """Process a single inference request."""
        try:
            request_type = request_data.get("type")
            temperature=request_data.get("temperature", 0.7)
            top_p=request_data.get("top_p", 0.95)
            max_tokens=request_data.get("max_tokens", 100)
            prompt = request_data.get("prompt", "")
            messages = None
            if request_type == "chat":
                # Handle chat completion
                messages = request_data.get("messages", [])
            else:
                # Handle text completion
                prompt = request_data.get("prompt", "")

            inputs = self.model.tokenize(prompt, messages)   
            # Generate text
            engine_out = await generate_text(
                model=self.model,
                inputs=inputs,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            
            # Submit the response
            self.queue_manager.submit_response(request_id, engine_out)
            
        except Exception as e:
            error_response = {
                "error": str(e),
                "status": "error"
            }
            self.queue_manager.submit_response(request_id, error_response)
    
    async def worker_loop(self):
        """Worker coroutine that processes requests from the queue."""
        while self.running:
            try:
                # Get the next request with a timeout
                request_id, request_data = self.queue_manager.get_next_request(timeout=0.1)
                
                # Process the request
                await self.process_request(request_id, request_data)
                
            except queue.Empty:
                # No request available, just continue
                await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                # Exit cleanly if the task is cancelled
                break
            except Exception as e:
                print(f"Error in worker loop: {e}")
    
    async def start(self):
        """Start the inference workers."""
        self.running = True
        
        # Start worker tasks
        worker = asyncio.create_task(self.worker_loop())
        self.worker_threads.append(worker)

        print(f"Started inference workers")
    
    async def stop(self):
        """Stop the inference workers."""
        self.running = False
        
        # Cancel all worker tasks
        for worker in self.worker_threads:
            worker.cancel()
            
        # Wait for tasks to complete cancellation
        for worker in self.worker_threads:
            try:
                await worker
            except asyncio.CancelledError:
                pass
                
        self.worker_threads = []
        print("Stopped inference workers")
