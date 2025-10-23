import multiprocessing
from sfllm.engine.inference_engine import InferenceEngine
import asyncio
import transformers
import time
from typing import Dict, Any
from sfllm.engine.sampling_params import SamplingParams
from sfllm.engine.sequence import SequenceStatus,Sequence

class EngineServer:
    def __init__(self, server_args):
        self.token_2_worker_queue = multiprocessing.Queue()
        self.worker_2_token_queue = multiprocessing.Queue()
        self.req_to_state: Dict[str, Any] = {}
        self.shared_event = multiprocessing.Manager().Event()
        self.server_args = server_args
        self.ready_flag = multiprocessing.Value("b", False)
        self.worker_threads = []
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(server_args.model_path)
        self.inference_engine = None

    async def submit_request(self, request_data: Dict[str, Any]) -> str:
        """Submit a new inference request and return the request ID."""
        import uuid
        tokenizer = self.tokenizer
        request_id = str(uuid.uuid4().hex) + "_" + str(time.time())
        sampling_params = SamplingParams(top_p=request_data.get("top_p", 1.0),
                                         max_new_tokens=request_data.get("max_new_tokens", 128))
        sequence = Sequence(
            request_data["prompt"],
            sampling_params,
            input_ids=tokenizer.encode(request_data["prompt"]),
        )
        self.req_to_state[sequence.sequence_id] = {
            "event": asyncio.Event(),
            "response": {"text": ""},
            "created_at": time.time(),
            "status": "PENDING",
            "request_id": request_id,
        }
        self.token_2_worker_queue.put(sequence)
        return sequence.sequence_id

    async def get_stream_response(self, sequence_id: str, timeout: int = 30) -> Any:
        """Get the response for a submitted inference request."""
        while self.req_to_state[sequence_id]["status"] not in ["COMPLETED", "FAILED"]:
            await asyncio.wait_for(
                self.req_to_state[sequence_id]["event"].wait(), timeout=timeout
            )
            self.req_to_state[sequence_id]["event"].clear()
            response = self.req_to_state[sequence_id]["response"]
            yield response
        self.req_to_state[sequence_id] = None

    @staticmethod
    def event_run_loop(self):
        self.inference_engine = InferenceEngine(self.server_args)
        self.ready_flag.value = True
        while True:
            if not self.running:
                break
            try:
                req_sequence = self.token_2_worker_queue.get(block=False)
                self.inference_engine.add_request(req_sequence)
            except:  # noqa: E722
                pass

            seq_group = self.inference_engine.step()
            if len(seq_group) == 0:
                time.sleep(0.1)
                continue

            seq_outputs = {}
            for sequence in seq_group:
                seq_outputs[sequence.sequence_id] = {
                    # "prompt": sequence.prompt,
                    "tokens": sequence.new_tokens,
                    "status": str(sequence.status),
                }
            self.worker_2_token_queue.put((seq_outputs))

    async def worker_response_loop(self):
        tokenizer = self.tokenizer
        while self.running:
            if self.worker_threads[-1].is_alive() is False:
                print("Inference worker process has stopped unexpectedly.")
                self.running = False
                self.worker_threads[-1].join()
                break
            try:
                response = self.worker_2_token_queue.get(block=False)
            except:  # noqa: E722
                await asyncio.sleep(0.1)
                continue
            for sequence_id, output in response.items():
                if sequence_id in self.req_to_state:
                    generated_text = tokenizer.decode(output["tokens"],skip_special_tokens=True)
                    self.req_to_state[sequence_id]["response"]["text"] += generated_text
                    self.req_to_state[sequence_id]["status"] = output["status"]
                    self.req_to_state[sequence_id]["event"].set()


    def start(self):
        """Start the inference workers."""
        multiprocessing.set_start_method("spawn", force=True)
        self.running = True
        # Start worker tasks
        worker = multiprocessing.Process(target=self.event_run_loop, args=(self,))
        worker.start()
        self.worker_threads.append(worker)
    
    async def stop(self):
        """Stop the inference workers."""
        self.running = False
        await asyncio.sleep(1)  # Give some time for workers to notice the running flag change
        # Terminate all worker tasks
        for worker in self.worker_threads:
            worker.terminate()
            
        # Wait for tasks to complete cancellation
        for worker in self.worker_threads:
            worker.join()

        self.worker_threads = []
        print("Stopped inference workers")
    
    def is_ready(self) -> bool:
        """Check if the inference engine is ready."""
        return self.ready_flag.value