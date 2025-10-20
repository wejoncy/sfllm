import asyncio
from typing import Dict, Any, List, Tuple

from sfllm.engine.model_runner import ModelRunner, generate_text, batch_generate_text
from sfllm.engine.scheduler import Scheduler
from sfllm.engine.sampling_params import SamplingParams
from sfllm.engine.sequence import Sequence,SequenceGroup

class InferenceEngine:
    """Worker that processes inference requests from a queue."""
    
    def __init__(self, model_path: str):
        """
        Initialize the inference worker.
        """
        self.model_runner = ModelRunner(model_path)
        self.running = False
        self.worker_threads = []
        self.scheduler = Scheduler()
        self.finished_sequences = []
    
    def post_forward(self, sequence_group: SequenceGroup, token_ids: List[int]) -> None:
        """Post-process the model outputs and update the sequences."""
        idx = 0
        for sequence in sequence_group:
            sequence.new_tokens = token_ids[idx: idx + 1]
            sequence.tokens.extend(sequence.new_tokens)
            idx += 1

            if len(sequence.tokens) < sequence.sampling_params.max_tokens:
                sequence.status = "RUNNING"
                self.scheduler.running_queue.put(sequence)
            else:
                sequence.status = "COMPLETED"
                self.finished_sequences.append(sequence)


    def add_request(self, prompt: str, sampling_params: SamplingParams) -> None:
        """Add a new inference request to the queue."""
        sequence = Sequence(prompt, sampling_params)
        sequence.tokens = self.model_runner.tokenize(prompt)
        sequence.new_tokens = sequence.tokens
        self.scheduler.add_request(sequence)
    
    def step(self):
        """Process a single inference request."""
        seq_group = self.scheduler.schedule()
        if seq_group.empty():
            return
        token_ids = self.model_runner.forward(seq_group)
        self.post_forward(seq_group, token_ids)

    def generate(self, prompt: List[str]|str, sampling_params: SamplingParams) -> Dict[str, Any]:
        """Generate text for inference requests."""
        if isinstance(prompt, str):
            prompt = [prompt]
        for p in prompt:
            self.add_request(p, sampling_params)
        while not self.scheduler.is_done():
            self.step()
        
        for sequence in self.finished_sequences:
            sequence.generated_text = self.model_runner.detokenize(sequence.tokens)
            print(f"Generated text for sequence {sequence.generated_text}")
        return {}

    async def process_requests(self, requests: List[Tuple[str, Dict[str, Any]]]):
        """Process a single inference request."""
        batch_requests = []
        for request_id, request_data in requests:
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
            batch_requests.append((request_id, temperature, top_p, max_tokens, inputs))
        try:
            policy_grouped_requests = self.sort_policy.sort_requests(batch_requests)
            for token_len, grouped_request in policy_grouped_requests.items():
                print(f"Processing batch of {len(grouped_request)} requests with token length {token_len}")
                batch_output = await batch_generate_text(
                    model=self.model,
                    batch_inputs=grouped_request,
                )
                for engine_out, request in zip(batch_output, grouped_request):
                    # Submit the response
                    request_id = request[0]
                    self.queue_manager.submit_response(request_id, engine_out)
        except Exception as e:
            error_response = {
                "error": str(e),
                "status": "error"
            }
            for token_len, grouped_request in policy_grouped_requests.items():
                self.queue_manager.submit_response(grouped_request[0], error_response)
    
    async def worker_loop(self):
        """Worker coroutine that processes requests from the queue."""
        while self.running:
            try:
                running_requests = self.scheduler.schedule()
                if len(running_requests) == 0:
                    # No request available, just continue
                    await asyncio.sleep(0.1)
                    continue
                print("running_requests", len(running_requests))
                print('pending_request:', self.queue_manager.size())
                # Process the request
                await self.process_requests(running_requests)
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

        print("Started inference workers")
    
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


if __name__ == "__main__":
    # Example usage
    engine = InferenceEngine("/home/jicwen/work/Qwen3-0.6B/")
    # engine.add_request("Hello, world!", SamplingParams())
    engine.generate("Hello, world!", SamplingParams())
    print("Inference step completed.")

        