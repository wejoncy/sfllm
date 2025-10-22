import logging
import argparse
from typing import Dict, Any, List, Tuple

from sfllm.engine.model_runner import ModelRunner
from sfllm.engine.scheduler import Scheduler
from sfllm.engine.sampling_params import SamplingParams
from sfllm.engine.sequence import Sequence,SequenceGroup, SequenceStatus
from sfllm.server_args import ServerArgs
from sfllm.utils import configure_logger
import multiprocessing

logger = logging.getLogger(__name__)
class InferenceEngine:
    """Worker that processes inference requests from a queue."""
    
    def __init__(self, server_args: ServerArgs):
        """
        Initialize the inference worker.
        """
        configure_logger(server_args)
        self.model_runner = ModelRunner(server_args)
        self.server_args = server_args
        self.running = False
        self.scheduler = Scheduler(self.model_runner.get_max_context_length())
        self.finished_sequences = []

    def post_forward(self, sequence_group: SequenceGroup, token_ids: List[int], failed_sequences: List[Sequence]) -> None:
        """Post-process the model outputs and update the sequences."""
        idx = 0
        for sequence in sequence_group:
            sequence.new_tokens = token_ids[idx: idx + 1]
            sequence.tokens.extend(sequence.new_tokens)
            idx += 1

            if len(sequence.tokens) - sequence.prompt_token_len < sequence.sampling_params.max_new_tokens:
                sequence.status = SequenceStatus.RUNNING
                self.scheduler.running_queue.put(sequence)
            else:
                sequence.status = SequenceStatus.COMPLETED
                self.finished_sequences.append(sequence)
                self.scheduler.free_sequence_resources(sequence)
            
        for sequence in failed_sequences:
            sequence.status = SequenceStatus.FAILED
            self.finished_sequences.append(sequence)
            self.scheduler.free_sequence_resources(sequence)


    def new_request(self, prompt: str|Tuple[str, List[int]], sampling_params: SamplingParams) -> int:
        if isinstance(prompt, str):
            sequence = Sequence(prompt, sampling_params)
            sequence.tokens = self.model_runner.tokenize(prompt)
            sequence.new_tokens = sequence.tokens
            sequence.prompt_token_len = len(sequence.tokens)
        else:
            assert isinstance(prompt, tuple), "Prompt must be a string or a tuple of (str, List[int])"
            sequence = Sequence(prompt[0], sampling_params, input_ids=prompt[1])

        return sequence

    def add_request(
        self,
        prompt: str | Tuple[str, List[int]] | Sequence,
        sampling_params: SamplingParams = SamplingParams(),
    ) -> int:
        """Add a new inference request to the queue."""
        if isinstance(prompt, Sequence):
            sequence = prompt
        else:
            sequence = self.new_request(prompt, sampling_params)
        self.scheduler.add_request(sequence)
        return sequence.sequence_id
    
    def step(self):
        """Process a single inference request."""
        seq_group, failed_sequences = self.scheduler.schedule()
        self.scheduler.metrics.refresh(seq_group)
        token_ids = []
        if not seq_group.empty():
            token_ids = self.model_runner.forward(seq_group)        
        self.post_forward(seq_group, token_ids, failed_sequences)
        return seq_group, failed_sequences

    def generate(self, prompt: List[str]|str, sampling_params: SamplingParams, stream:bool=False) -> Dict[str, Any]:
        """Generate text for inference requests."""
        
        if isinstance(prompt, str):
            prompt = [prompt]
        for p in prompt:
            self.add_request(p, sampling_params)
        while not self.scheduler.is_done():
            seq_group, failed_sequences = self.step()
            if stream:
                seq_outputs = {}
                for sequence in seq_group:
                    if sequence.status == SequenceStatus.RUNNING:
                        generated_text = self.model_runner.detokenize(
                            sequence.new_tokens,
                        )
                        seq_outputs[sequence.sequence_id] = {"prompt": sequence.prompt, "text": generated_text}
                        yield seq_outputs

        if not stream:
            seq_outputs = {}
            for sequence in self.finished_sequences:
                sequence.generated_text = self.model_runner.detokenize(
                    sequence.tokens[sequence.prompt_token_len:],
                )
                seq_outputs[sequence.sequence_id] = {
                    "prompt": sequence.prompt,
                    "text": sequence.generated_text,
                }
            return seq_outputs




if __name__ == "__main__":
    # Example usage
    import sys
    sys.argv = ["", "--model", "/home/jicwen/work/Qwen3-0.6B/"]
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    # server_args.disable_cuda_graph = True
    engine = InferenceEngine(server_args)
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    # engine.add_request("Hello, world!", SamplingParams())
    outputs = engine.generate(prompts, SamplingParams(max_new_tokens=30, top_k=1), stream=True)
    for output in outputs:
        for _, output_d in output.items():
            print(f"Prompt: {output_d['prompt']}\nGenerated text: {output_d['text']}")
    print("Inference step completed.")
