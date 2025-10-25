
"""
nsys profile  --force-overwrite=true  -o baseline-report  --trace=cuda,nvtx,osrt,cudnn --cuda-graph-trace=node  python python/sfllm/engine/inference_engine.py
"""
import logging
from typing import Dict, Any, List, Tuple

from sfllm.engine.model_runner import ModelRunner
from sfllm.engine.scheduler import Scheduler
from sfllm.engine.sampling_params import SamplingParams
from sfllm.engine.sequence import RequestSequence, SequenceGroup, SequenceStatus, AbortSequence
from sfllm.server_args import ServerArgs
from sfllm.utils.nutils import configure_logger

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
        self.scheduler = Scheduler(server_args)
        self.model_runner.set_mem_pool(
            self.scheduler.block_memory_manager.physical_memory_pool
        )
        self.model_runner.capture_graph()

    def post_forward(self, sequence_group: SequenceGroup, token_ids: List[int], failed_sequences: List[RequestSequence]) -> None:
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
                self.scheduler.free_sequence_resources(sequence)
                sequence.status = SequenceStatus.COMPLETED
                # abort request may have req_id added after completed, so we need to check again
                sid = next(iter(self.scheduler.abort_requests), None)
                # 100 should be safe to set as buffer
                if sid is not None and sid + 100 < sequence.sequence_id:
                    self.scheduler.abort_requests.remove(sid)


    def new_request(self, prompt: str|Tuple[str, List[int]], sampling_params: SamplingParams) -> int:
        if isinstance(prompt, str):
            sequence = RequestSequence(prompt, sampling_params)
            sequence.init(self.model_runner.model.tokenizer)
        else:
            assert isinstance(prompt, tuple), "Prompt must be a string or a tuple of (str, List[int])"
            sequence = RequestSequence(prompt[0], sampling_params, input_ids=prompt[1])

        return sequence

    def add_request(
        self,
        prompt: str | Tuple[str, List[int]] | RequestSequence,
        sampling_params: SamplingParams = SamplingParams(),
    ) -> int:
        """Add a new inference request to the queue."""
        if isinstance(prompt, RequestSequence):
            sequence = prompt
        elif isinstance(prompt, AbortSequence):
            self.scheduler.add_abort_request(prompt.sequence_id)
            return prompt.sequence_id
        else:
            sequence = self.new_request(prompt, sampling_params)
        self.scheduler.add_request(sequence)
        return sequence.sequence_id
    
    def step(self):
        """Process a single inference request."""
        seq_group, failed_sequences = self.scheduler.schedule()
        token_ids = []
        if not seq_group.empty():
            token_ids = self.model_runner.forward(seq_group)        
        self.post_forward(seq_group, token_ids, failed_sequences)
        seq_group.append(failed_sequences)
        return seq_group

    def generate(self, prompt: List[str]|str, sampling_params: SamplingParams, stream:bool=False) -> Dict[str, Any]:
        """Generate text for inference requests."""
        
        if isinstance(prompt, str):
            prompt = [prompt]
        for p in prompt:
            self.add_request(p, sampling_params)

        while not self.scheduler.is_done():
            seq_group = self.step()
            seq_outputs = {}
            for sequence in seq_group:
                if stream:
                    if sequence.status == SequenceStatus.RUNNING:
                        generated_text = self.model_runner.detokenize(
                            sequence.new_tokens,
                        )
                        seq_outputs[sequence.sequence_id] = {"prompt": sequence.prompt, "text": generated_text}
                        yield seq_outputs
                elif sequence.status in [SequenceStatus.COMPLETED, SequenceStatus.FAILED]:
                    sequence.generated_text = self.model_runner.detokenize(
                        sequence.tokens[sequence.prompt_token_len:],
                    )
                    yield {
                        sequence.sequence_id: {
                            "prompt": sequence.prompt,
                            "text": sequence.generated_text,
                        }
                    }
