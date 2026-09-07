import asyncio
import time
import logging
from tokenizers.decoders import DecodeStream
import torch.multiprocessing as multiprocessing
from sfllm.engine.sequence import AbortSequence, DecodeSequence, RequestSequence
from sfllm.engine.sampling_params import SamplingParams
from sfllm.serving.req_protocol import GenerateReqInput
from sfllm.engine.inference_engine import InferenceEngine
from sfllm.server_args import set_global_server_args_for_scheduler

logger = logging.getLogger(__name__)

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
        self.decode_states = {}

    def set_tokenizer_queues(self, input_queue, output_queue):
        self.tokenizer_input_queue = input_queue
        self.tokenizer_output_queue = output_queue

    def load_tokenizer(self):
        from transformers import AutoConfig, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.server_args.model_path, trust_remote_code=True,)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model_config = AutoConfig.from_pretrained(self.server_args.model_path)
        config_values = vars(model_config)
        text_config = config_values.get("text_config") or model_config
        model_eos = vars(text_config).get("eos_token_id")
        if isinstance(model_eos, int):
            model_eos = (model_eos,)
        self.eos_token_ids = frozenset(model_eos or ())
        if self.tokenizer.eos_token_id is not None:
            self.eos_token_ids |= {self.tokenizer.eos_token_id}

    @staticmethod
    def inferengine_event_run_loop(self):
        # ``spawn`` starts a fresh interpreter, so module globals set by
        # ServerArgs.__post_init__ in the HTTP parent are not inherited.
        set_global_server_args_for_scheduler(self.server_args)
        self.inference_engine = InferenceEngine(self.server_args)
        self.ready_flag.value = True
        import queue
        import threading
        thread = None
        if not self.server_args.disable_overlap:
            th_event = threading.Event()
            thread = threading.Thread(target=self.inference_engine.event_loop_overlap, args=(th_event,))
            thread.start()
        while True:
            if not self.running or (thread is not None and not thread.is_alive()):
                logger.error("Inference engine event loop stopped unexpectedly.")
                break
            try:
                for i in range(10):
                    req_sequence = self.inferengine_input_queue.get_nowait()
                    self.inference_engine.add_request(req_sequence)
            except queue.Empty:  # noqa: E722
                pass
            
            if not self.server_args.disable_overlap:
                seq_group = self.inference_engine.step_overlap(timeout=0.1)
            else:
                seq_group = self.inference_engine.step()
            if len(seq_group) == 0:
                time.sleep(0.1)
                continue

            seq_outputs = []
            for sequence in seq_group:
                decode_seq = DecodeSequence(sequence)
                seq_outputs.append(decode_seq)
            self.tokenizer_input_queue.put(seq_outputs)
        if not self.server_args.disable_overlap:
            th_event.set()
            thread.join()


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
                    self.decode_states.pop(out_sequence.sequence_id, None)
                    self.inferengine_input_queue.put(out_sequence)
                elif isinstance(out_sequence, RequestSequence):
                    out_sequence.init(self.tokenizer)
                    out_sequence.sampling_params.stop_token_ids |= self.eos_token_ids
                    stop_token_sequences = []
                    for stop in out_sequence.sampling_params.stop:
                        token_ids = tuple(
                            self.tokenizer.encode(stop, add_special_tokens=False)
                        )
                        if token_ids:
                            stop_token_sequences.append(token_ids)
                    out_sequence.sampling_params.stop_token_sequences = tuple(
                        stop_token_sequences
                    )
                    self.decode_states[out_sequence.sequence_id] = (
                        DecodeStream(skip_special_tokens=True), [], 0
                    )
                    self.inferengine_input_queue.put(out_sequence)
                elif isinstance(out_sequence, list):
                    assert isinstance(out_sequence[0], DecodeSequence)
                    seq_outputs = {}
                    for seq in out_sequence:
                        state = self.decode_states.get(seq.sequence_id)
                        if state is None:
                            continue  # Output already in flight when aborted.
                        decoder, token_ids, text_offset = state
                        token_ids.extend(seq.tokens)
                        finished = not seq.status.is_active()
                        if finished:
                            generated_text = self.tokenizer.decode(
                                token_ids, skip_special_tokens=True
                            )[text_offset :]
                            self.decode_states.pop(seq.sequence_id)
                        else:
                            generated_text = decoder.step(
                                self.tokenizer.backend_tokenizer, seq.tokens
                            ) or ""
                            self.decode_states[seq.sequence_id] = (
                                decoder, token_ids, text_offset + len(generated_text)
                            )
                        seq_outputs[seq.sequence_id] = {
                            "text": generated_text,
                            "output_ids": seq.tokens,
                            "completion_tokens": seq.completion_tokens,
                            "status": seq.status,
                        }
                    if seq_outputs:
                        self.tokenizer_output_queue.put(seq_outputs)
                else:
                    raise ValueError("Unknown sequence type received in tokenizer_event_run_loop.")
            except Exception as e:
                print(f"Error occurred in tokenizer_event_run_loop: {e}")
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