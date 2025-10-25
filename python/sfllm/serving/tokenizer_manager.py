import asyncio
import time
import multiprocessing
from sfllm.engine.sequence import AbortSequence, DecodeSequence, RequestSequence
from sfllm.engine.sampling_params import SamplingParams
from sfllm.serving.req_protocol import GenerateReqInput
from sfllm.engine.inference_engine import InferenceEngine


class TokenizerManager:
    def __init__(self, server_args):
        self.server_args = server_args
        self.tokenizer = None
        self.inferengine_input_queue = multiprocessing.Queue()
        # self.inferengine_output_queue = multiprocessing.Queue()
        self.running = False
        self.worker_threads = []
        self.tokenizer_input_queue = None
        self.tokenizer_output_queue = None
        self.ready_flag = multiprocessing.Value("b", False)

    def set_tokenizer_queues(self, input_queue, output_queue):
        self.tokenizer_input_queue = input_queue
        self.tokenizer_output_queue = output_queue

    def load_tokenizer(self):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.server_args.model_path, trust_remote_code=True,)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @staticmethod
    def inferengine_event_run_loop(self):
        self.inference_engine = InferenceEngine(self.server_args)
        self.ready_flag.value = True
        import queue
        while True:
            if not self.running:
                break
            try:
                for i in range(2):
                    req_sequence = self.inferengine_input_queue.get_nowait()
                    self.inference_engine.add_request(req_sequence)
            except queue.Empty:  # noqa: E722
                pass

            seq_group = self.inference_engine.step()
            if len(seq_group) == 0:
                time.sleep(0.1)
                continue

            seq_outputs = []
            for sequence in seq_group:
                decode_seq = DecodeSequence(sequence)
                seq_outputs.append(decode_seq)
            self.tokenizer_input_queue.put(seq_outputs)


    @staticmethod
    def tokenizer_event_run_loop(self):
        self.load_tokenizer()
        self.running = True
        while self.running:
            if not self.running:
                break
            try:
                out_sequence = self.tokenizer_input_queue.get()
                if isinstance(out_sequence, AbortSequence):
                    self.inferengine_input_queue.put(out_sequence)
                elif isinstance(out_sequence, RequestSequence):
                    out_sequence.init(self.tokenizer)
                    self.inferengine_input_queue.put(out_sequence)
                elif isinstance(out_sequence, list):
                    assert isinstance(out_sequence[0], DecodeSequence)
                    seq_outputs = {}
                    for seq in out_sequence:
                        generated_text = self.tokenizer.decode(
                            seq.tokens, skip_special_tokens=True
                        )
                        seq_outputs[seq.sequence_id] = {
                            "text": generated_text,
                            "output_ids": seq.tokens,
                            "completion_tokens": seq.completion_tokens,
                            "status": seq.status,
                        }
                    self.tokenizer_output_queue.put(seq_outputs)
                else:
                    raise ValueError("Unknown sequence type received in tokenizer_event_run_loop.")
            except Exception:
                exit(-1)

    def start(self):
        """Start the inference workers."""
        self.running = True
        # Start worker tasks
        worker = multiprocessing.Process(target=self.inferengine_event_run_loop, args=(self,))
        worker.start()
        self.worker_threads.append(worker)

    def stop(self):
        """Stop the inference workers."""
        self.running = False
        for worker in self.worker_threads:
            worker.terminate()
            worker.join()