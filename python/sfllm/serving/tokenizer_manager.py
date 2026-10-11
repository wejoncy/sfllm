import logging
import queue
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from tokenizers.decoders import DecodeStream
from transformers import AutoConfig, AutoTokenizer, GenerationConfig
import torch.multiprocessing as multiprocessing
from sfllm.engine.sequence import AbortSequence, DecodeSequence, RequestSequence, SequenceStatus
from jinja2 import TemplateError
from sfllm.serving.req_protocol import ControlRequest
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
        try:
            generation_eos = GenerationConfig.from_pretrained(
                self.server_args.model_path
            ).eos_token_id
        except OSError:
            generation_eos = None
        if isinstance(generation_eos, int):
            generation_eos = (generation_eos,)
        self.eos_token_ids |= frozenset(generation_eos or ())

    def encode_chat(self, sequence, tokenizer=None):
        kwargs = dict(sequence.chat_template_kwargs)
        # These options define the token sequence consumed by the engine.
        kwargs.update(
            tokenize=True, add_generation_prompt=True,
            return_dict=False, return_tensors=None,
        )
        return (tokenizer or self.tokenizer).apply_chat_template(sequence.messages, **kwargs)

    @staticmethod
    def inferengine_event_run_loop(self):
        # Keep model/kernel imports in the GPU process, outside HTTP/tokenizer GC.
        from sfllm.engine.inference_engine import InferenceEngine, OVERLAP_IDLE_WAIT_SECONDS

        # ``spawn`` starts a fresh interpreter, so module globals set by
        # ServerArgs.__post_init__ in the HTTP parent are not inherited.
        set_global_server_args_for_scheduler(self.server_args)
        self.inference_engine = InferenceEngine(self.server_args)
        self.ready_flag.value = True
        import threading
        thread = None
        if not self.server_args.disable_overlap:
            th_event = threading.Event()
            thread = threading.Thread(
                target=self.inference_engine.event_loop_overlap,
                args=(th_event, self.inferengine_input_queue),
            )
            thread.start()
        while True:
            if not self.running or (thread is not None and not thread.is_alive()):
                logger.error("Inference engine event loop stopped unexpectedly.")
                break
            if not self.server_args.disable_overlap:
                seq_group = self.inference_engine.step_overlap(
                    timeout=OVERLAP_IDLE_WAIT_SECONDS
                )
            else:
                try:
                    for _ in range(self.server_args.max_running_requests):
                        request = self.inferengine_input_queue.get_nowait()
                        self.inference_engine.add_request(request)
                        if isinstance(request, ControlRequest):
                            self.tokenizer_output_queue.put(
                                self.inference_engine.output_batch_queue.get_nowait()
                            )
                except queue.Empty:
                    pass
                seq_group = self.inference_engine.step()
            if len(seq_group) == 0:
                continue

            if isinstance(seq_group, tuple):
                self.tokenizer_output_queue.put(seq_group)
            else:
                self.tokenizer_input_queue.put(seq_group)
        if not self.server_args.disable_overlap:
            th_event.set()
            thread.join()


    def _encode_request(self, sequence, tokenizer):
        if sequence.messages is None:
            sequence.init(tokenizer)
        else:
            try:
                input_ids = self.encode_chat(sequence, tokenizer)
            except (TemplateError, ValueError, TypeError) as exc:
                self.tokenizer_output_queue.put({sequence.sequence_id: {
                    "text": "", "output_ids": [],
                    "prompt_length": 0, "completion_tokens": 0,
                    "status": SequenceStatus.FAILED, "error": str(exc),
                }})
                return
            sequence.init(input_ids)
        sequence.sampling_params.stop_token_ids |= self.eos_token_ids
        stop_token_sequences = []
        for stop in sequence.sampling_params.stop:
            token_ids = tuple(
                tokenizer.encode(stop, add_special_tokens=False)
            )
            if token_ids:
                stop_token_sequences.append(token_ids)
        sequence.sampling_params.stop_token_sequences = tuple(
            stop_token_sequences
        )
        self.decode_states[sequence.sequence_id] = (
            DecodeStream(skip_special_tokens=True), [], 0
        )
        self.inferengine_input_queue.put(sequence)

    @staticmethod
    def tokenizer_event_run_loop(self):
        self.load_tokenizer()
        self.running = True
        with ThreadPoolExecutor(max_workers=self.server_args.tokenizer_worker_num) as executor:
            self._run_tokenizer_loop(executor)

    def _run_tokenizer_loop(self, executor):
        # HF tokenizers mutate padding/truncation settings during encoding.
        tokenizers = (self.tokenizer,) + tuple(
            deepcopy(self.tokenizer)
            for _ in range(self.server_args.tokenizer_worker_num - 1)
        )
        next_message = None
        while self.running:
            try:
                out_sequence = (
                    next_message if next_message is not None
                    else self.tokenizer_input_queue.get()
                )
                next_message = None
                if isinstance(out_sequence, ControlRequest):
                    if out_sequence is ControlRequest.FLUSH_CACHE and self.decode_states:
                        self.tokenizer_output_queue.put((out_sequence, False))
                    else:
                        self.inferengine_input_queue.put(out_sequence)
                elif isinstance(out_sequence, AbortSequence):
                    self.decode_states.pop(out_sequence.sequence_id, None)
                    self.inferengine_input_queue.put(out_sequence)
                elif isinstance(out_sequence, RequestSequence):
                    requests = [out_sequence]
                    # Use already queued prompts; never wait to fill a group.
                    for _ in range(1, len(tokenizers)):
                        try:
                            next_message = self.tokenizer_input_queue.get_nowait()
                        except queue.Empty:
                            break
                        if not isinstance(next_message, RequestSequence):
                            break
                        requests.append(next_message)
                        next_message = None
                    if len(requests) > 1:
                        # Workers send immediately. Drain before processing controls
                        # or decode messages, which own the same request states.
                        for _ in executor.map(self._encode_request, requests, tokenizers):
                            pass
                    else:
                        self._encode_request(out_sequence, self.tokenizer)
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
                            "prompt_length": seq.prompt_token_len,
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
