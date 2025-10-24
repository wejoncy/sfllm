import multiprocessing
import asyncio
import time
from typing import Dict, Any
from sfllm.engine.sampling_params import SamplingParams
from sfllm.engine.sequence import RequestSequence, AbortSequence,SequenceStatus
from sfllm.serving.req_protocol import GenerateReqInput
from sfllm.serving.tokenizer_manager import TokenizerManager


class EngineServer:
    def __init__(self, server_args):
        self.tokenizer_input_queue = multiprocessing.Queue()
        self.tokenizer_output_queue = multiprocessing.Queue()
        self.req_to_state: Dict[str, Any] = {}
        self.shared_event = multiprocessing.Manager().Event()
        self.server_args = server_args
        self.ready_flag = multiprocessing.Value("b", False)
        self.worker_threads = []
        self.tokenizer_manager = TokenizerManager(self.server_args)
        self.tokenizer_manager.set_tokenizer_queues(
            self.tokenizer_input_queue, self.tokenizer_output_queue
        )

    async def submit_request(self, request: GenerateReqInput) -> str:
        """Submit a new inference request and return the request ID."""
        import uuid
        request_id = str(uuid.uuid4().hex) + "_" + str(time.time())
        sampling_params = SamplingParams(
            top_p=request.sampling_params.get("top_p", 1.0), max_new_tokens=request.sampling_params.get("max_new_tokens", 1024)
        )
        sequence = RequestSequence(
            request.text,
            sampling_params,
            input_ids=request.input_ids,
        )
        self.req_to_state[sequence.sequence_id] = {
            "event": asyncio.Event(),
            "response": {
                "text": "",
                "output_ids": [],
                "meta_info": {
                    "id": sequence.sequence_id,
                    "prompt_length": sequence.prompt_token_len,
                    "completion_tokens": 0,
                },
            },
            "status": "PENDING",
            "request_id": request_id,
        }
        self.tokenizer_input_queue.put(sequence)
        return sequence.sequence_id

    async def get_response(self, sequence_id: str, timeout: int = -1, streaming: bool = False) -> Any:
        """Get the response for a submitted inference request."""
        while self.req_to_state[sequence_id]["status"] not in [SequenceStatus.COMPLETED, SequenceStatus.FAILED]:
            await asyncio.wait_for(
                self.req_to_state[sequence_id]["event"].wait(), timeout=timeout
            )
            self.req_to_state[sequence_id]["event"].clear()
            response = self.req_to_state[sequence_id]["response"]
            if streaming:
                self.req_to_state[sequence_id]["response"]["output_ids"] = []
                yield response
        response = self.req_to_state[sequence_id]["response"]
        self.req_to_state.pop(sequence_id)
        yield response

    async def auto_clean_resource_loop(self):
        while self.running:
            await asyncio.sleep(600)  # Clean every 10 minutes
            to_delete = []
            for sequence_id, state in self.req_to_state.items():
                if (
                    state["status"] in [SequenceStatus.COMPLETED, SequenceStatus.FAILED]
                    and state["finished_time"] + 60 < time.time()
                ):
                    to_delete.append(sequence_id)
            for sequence_id in to_delete:
                self.req_to_state.pop(sequence_id)

    async def worker_response_loop(self):
        while self.running:
            if self.worker_threads[-1].is_alive() is False:
                print("Inference worker process has stopped unexpectedly.")
                self.running = False
                self.worker_threads[-1].join()
                break
            try:
                response = self.tokenizer_output_queue.get(block=False)
            except:  # noqa: E722
                await asyncio.sleep(0.1)
                continue
            for sequence_id, output in response.items():
                assert sequence_id in self.req_to_state
                this_response = self.req_to_state[sequence_id]["response"]
                this_response["text"] += output["text"]
                this_response["output_ids"].append(output["output_ids"])
                this_response["meta_info"]["completion_tokens"] = output["completion_tokens"]

                self.req_to_state[sequence_id]["status"] = output["status"]
                self.req_to_state[sequence_id]["event"].set()
                self.req_to_state[sequence_id]["finished_time"] = time.time()


    def abort_request(self, rid: int):
        if rid not in self.req_to_state:
            return
        abort_req = AbortSequence(rid)
        self.tokenizer_input_queue.put(abort_req)

    def create_abort_task(self, rid: int, background_tasks):
        # Abort the request if the client is disconnected.
        async def abort_request():
            await asyncio.sleep(2)
            self.abort_request(rid)

        background_tasks.add_task(abort_request)
        return background_tasks

    def start(self):
        """Start the inference workers."""
        multiprocessing.set_start_method("spawn", force=True)
        self.running = True
        # Start worker tasks
        worker = multiprocessing.Process(
            target=self.tokenizer_manager.tokenizer_event_run_loop,
            args=(self.tokenizer_manager,),
        )
        worker.start()
        self.tokenizer_manager.start()
        self.worker_threads.append(worker)
    
    async def stop(self):
        """Stop the inference workers."""
        self.tokenizer_manager.stop()
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
        return self.tokenizer_manager.ready_flag.value