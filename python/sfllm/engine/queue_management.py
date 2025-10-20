import queue
import asyncio
import uuid
import threading
import time
from typing import Dict, Any, Optional, Tuple, List

class QueueManager:
    """Manages input and output queues for the inference engine."""
    
    def __init__(self, max_queue_size: int = 100):
        self.input_queue = queue.Queue(maxsize=max_queue_size)
        self.output_queue = queue.Queue(maxsize=max_queue_size)
        self.request_map = {}  # Maps request_id to response objects
        self.lock = threading.Lock()
    
    def submit_request(self, request_data: Dict[str, Any]) -> str:
        """
        Submit a request to the input queue.
        
        Args:
            request_data: Request parameters 
            
        Returns:
            request_id: Unique ID for this request
        """
        request_id = str(uuid.uuid4())
        self.input_queue.put((request_id, request_data))
        return request_id
    
    def size(self) -> int:
        """
        Get the size of the input queue.
        
        Returns:
            Size of the input queue
        """
        return self.input_queue.qsize()
    
    def get_next_request(self) -> Tuple[str, Dict[str, Any]]:
        """
        Get the next request from the input queue.
        
        Returns:
            Tuple of (request_id, request_data)
        """
        req = None
        try:
            req = self.input_queue.get(block=False)
        except queue.Empty:
            pass
        return req
    
    def submit_response(self, request_id: str, response: Any) -> None:
        """
        Submit a response to the output queue.
        
        Args:
            request_id: ID of the original request
            response: Response data
        """
        with self.lock:
            if request_id in self.request_map:
                # Replace the event with the actual response
                event = self.request_map[request_id]
                self.request_map[request_id] = response
                # Signal that the response is ready
                event.set()
        
    async def get_response(self, request_id: str, timeout: float = 30.0) -> Any:
        """
        Wait for and return the response for a specific request.
        
        Args:
            request_id: The request ID to wait for
            timeout: Maximum time to wait in seconds
            
        Returns:
            The response data
            
        Raises:
            TimeoutError: If the response doesn't arrive within timeout
        """
        end_time = time.time() + timeout
        
        # Register that we're waiting for this request
        with self.lock:
            waiting_event = threading.Event()
            self.request_map[request_id] = waiting_event
        
        # Check if the response is already in the output queue
        try:
            while time.time() < end_time:
                # Check if our response has been processed
                if waiting_event.is_set():
                    with self.lock:
                        response = self.request_map.pop(request_id)
                        return response
                
                # Wait a bit before checking again
                await asyncio.sleep(0.1)
                
            # Timed out
            with self.lock:
                if request_id in self.request_map:
                    self.request_map.pop(request_id)
            raise TimeoutError(f"Request {request_id} timed out after {timeout} seconds")
            
        except Exception as e:
            with self.lock:
                if request_id in self.request_map:
                    self.request_map.pop(request_id)
            raise e
