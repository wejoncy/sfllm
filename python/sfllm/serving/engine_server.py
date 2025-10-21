import multiprocessing
from sfllm.engine.inference_engine import InferenceEngine
import asyncio
import transformers
import time
from typing import Dict, Any
from sfllm.engine.sampling_params import SamplingParams

class EngineServer:
    def __init__(self, server_args):
        self.token_2_worker_queue = multiprocessing.Queue()
        self.worker_2_token_queue = multiprocessing.Queue()
        self.req_to_state: Dict[str, Any] = {}
        self.shared_event = multiprocessing.Manager().Event()
        self.server_args = server_args
        self.ready_flag = multiprocessing.Value("b", False)
        self.worker_threads = []
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            server_args.model_path, use_fast=False
        )
        self.inference_engine = None

    async def submit_request(self, request_data: Dict[str, Any]) -> str:
        """Submit a new inference request and return the request ID."""
        import json
        tokenizer = self.tokenizer
        request_id = (
            str(multiprocessing.current_process().pid)
            + "_"
            + str(time.time())
        )
        msg = {
            "request_id": request_id,
            "prompt": request_data["prompt"],
            "input_ids": tokenizer.encode(request_data["prompt"]),
            "sampling_params": {
                "top_p": request_data.get("top_p", 1.0),
                "max_new_tokens": request_data.get("max_new_tokens", 128),
            },
        }
        self.req_to_state[request_id] = {
            "event": asyncio.Event(),
            "response": {},
            "created_at": time.time(),
            "status": "PENDING",
        }
        self.token_2_worker_queue.put(json.dumps(msg))
        return request_id

    async def get_stream_response(self, request_id: str, timeout: int = 30) -> Any:
        """Get the response for a submitted inference request."""
        while self.req_to_state[request_id]["status"] != "COMPLETED":
            await asyncio.wait_for(
                self.req_to_state[request_id]["event"].wait(), timeout=timeout
            )
            self.req_to_state[request_id]["event"].clear()
            yield self.req_to_state[request_id]["response"]

    @staticmethod
    def worker_loop(self):
        self.inference_engine = InferenceEngine(self.server_args)
        self.ready_flag.value = True
        import json
        while True:
            if not self.running:
                break
            try:
                msg = self.token_2_worker_queue.get(block=False)
                request = json.loads(msg)
                prompt = request["prompt"]
                input_ids = request["input_ids"]
                sampling_params = SamplingParams.from_dict(request["sampling_params"])
                self.inference_engine.add_request((prompt, input_ids), sampling_params)
            except:
                pass

            seq_group = self.inference_engine.step()
            if len(seq_group) == 0:
                time.sleep(0.1)
                continue

            seq_outputs = {}
            for sequence in seq_group:
                seq_outputs[request["request_id"]] = {
                    # "prompt": sequence.prompt,
                    "tokens": sequence.new_tokens,
                    "status": sequence.status,
                }
            self.worker_2_token_queue.put(json.dumps(seq_outputs))

    async def worker_response_loop(self):
        import json
        tokenizer = self.tokenizer
        while self.running:
            if self.worker_threads[-1].is_alive() is False:
                print("Inference worker process has stopped unexpectedly.")
                self.running = False
                break
            try:
                msg = self.worker_2_token_queue.get(block=False)
            except:
                await asyncio.sleep(0.1)
                continue
            response = json.loads(msg)
            for request_id, output in response.items():
                if request_id in self.req_to_state:
                    if "text" not in self.req_to_state[request_id]["response"]:
                        self.req_to_state[request_id]["response"]["text"] = ""
                    generated_text = tokenizer.decode(output["tokens"])
                    self.req_to_state[request_id]["response"]["text"] += generated_text
                    self.req_to_state[request_id]["status"] = output["status"]
                    self.req_to_state[request_id]["event"].set()


    def start(self):
        """Start the inference workers."""
        multiprocessing.set_start_method("spawn", force=True)
        self.running = True
        # Start worker tasks
        worker = multiprocessing.Process(target=self.worker_loop, args=(self,))
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