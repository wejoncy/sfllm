import asyncio
import argparse
from typing import Dict, Any, List, Tuple

from sfllm.engine.model_runner import ModelRunner
from sfllm.engine.scheduler import Scheduler
from sfllm.engine.sampling_params import SamplingParams
from sfllm.engine.sequence import Sequence,SequenceGroup
from sfllm.server_args import ServerArgs
from sfllm.utils import configure_logger


class InferenceEngine:
    """Worker that processes inference requests from a queue."""
    
    def __init__(self, server_args: ServerArgs):
        """
        Initialize the inference worker.
        """
        configure_logger(server_args)
        self.model_runner = ModelRunner(server_args)
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
        sequence.prompt_token_len = len(sequence.tokens)
        self.scheduler.add_request(sequence)
    
    def step(self):
        """Process a single inference request."""
        seq_group = self.scheduler.schedule()
        if seq_group.empty():
            return
        token_ids = self.model_runner.forward(seq_group)
        self.scheduler.metrics.refresh(seq_group)
        self.post_forward(seq_group, token_ids)

    def generate(self, prompt: List[str]|str, sampling_params: SamplingParams) -> Dict[str, Any]:
        """Generate text for inference requests."""
        seq_outputs = {}
        if isinstance(prompt, str):
            prompt = [prompt]
        for p in prompt:
            self.add_request(p, sampling_params)
        while not self.scheduler.is_done():
            self.step()
        
        for sequence in self.finished_sequences:
            sequence.generated_text = self.model_runner.detokenize(
                sequence.tokens[sequence.prompt_token_len:],
            )
            seq_outputs[sequence.sequence_id] = {"prompt": sequence.prompt, "generated_text": sequence.generated_text}
        return seq_outputs
    
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
    import sys
    sys.argv = ["", "--model", "/home/jicwen/work/Qwen3-0.6B/"]
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    server_args.disable_cuda_graph = True
    engine = InferenceEngine(server_args)
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    # engine.add_request("Hello, world!", SamplingParams())
    outputs = engine.generate(prompts, SamplingParams(max_tokens=300, top_k=1))
    for k, output in outputs.items():
        print("===============================")
        print(f"Prompt: {output['prompt']}\nGenerated text: {output['generated_text']}")
    print("Inference step completed.")
