import logging
import torch
import bisect
import tqdm
import transformers
from typing import Dict, List, Any

from sfllm.model_loader.model_loader import initialize_model
from sfllm.engine.shedule_batch import ScheduleBatch,BatchResult
from sfllm.engine.forward_params import ForwardMode, ForwardBatch
from sfllm.engine.memory_pool import BlockMemoryManager
from sfllm.layers.sampler import Sampler, SamplingBatchInfo
from sfllm.server_args import ServerArgs
from sfllm.utils.nutils import DEFAULT_CUDA_GRAPH_BATCH_SIZES, MAX_PROCESSED_TOKENS

logger = logging.getLogger(__name__)

class ModelRunner:
    def __init__(self, server_args: ServerArgs, device_id: int = 0, is_draft: bool = False):
        self.is_draft = is_draft
        if is_draft:
            model_path = server_args.draft_model_path
        else:
            model_path = server_args.model_path
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
        self.model = initialize_model(model_path, server_args.dtype)
        self.device_id = device_id
        self.sampler = Sampler(self.model.config)
        self.rank = 0
        self.dtype = self.model.dtype
        self.server_args = server_args

        max_batch_size = server_args.cuda_graph_max_bs
        self.input_ids = torch.empty((max_batch_size,), dtype=torch.long, device=self.device_id)
        self.position_ids = torch.empty((max_batch_size,), dtype=torch.long, device=self.device_id)
        # we mange eagle related cuda graph buffers in EagleWorker 
        self.create_cudagraph_buffers()
        self.init_attn_backend_buffers()

    def get_config(self):
        return self.model.config

    def init_attn_backend_buffers(self):
        # attn backend related buffers
        config = self.model.config
        max_kv_splits = 16
        max_batch_size = 512
        self.attn_logits = torch.empty(
            (
                max_batch_size*2,
                config.num_attention_heads,
                max_kv_splits,
                config.head_dim,
            ),
            dtype=torch.float32,
            device="cuda",
        )
        self.attn_lse = torch.empty(
            (max_batch_size*2, config.num_attention_heads, max_kv_splits),
            dtype=torch.float32,
            device="cuda",
        )

    def init_memory_pool(self):
        self.block_memory_manager = BlockMemoryManager(
            self.server_args, self.get_config()
        )

    def wrap_attn_backend(self, attn_logits, attn_lse):
        self.attn_logits = attn_logits
        self.attn_lse = attn_lse

    def create_cudagraph_buffers(self):
        # cuda graph
        device_id = self.device_id
        server_args = self.server_args
        self.compute_stream = torch.cuda.Stream(device=device_id)
        self.copy_in_stream = torch.cuda.Stream(device=device_id)
        self.cuda_graph_max_bs = server_args.cuda_graph_max_bs
        ind = bisect.bisect_right(DEFAULT_CUDA_GRAPH_BATCH_SIZES, self.cuda_graph_max_bs)
        self.capture_batch_size = DEFAULT_CUDA_GRAPH_BATCH_SIZES[:ind]
        self.capture_batch_size = self.capture_batch_size[:ind]

        max_batch_size = max(self.capture_batch_size)
        self.out_cache_loc = torch.zeros((max_batch_size,), dtype=torch.int64, device=self.device_id)
        self.output_logits = {}
        self.cuda_graphs = {}

        self.num_kv_splits_buffer = torch.zeros((MAX_PROCESSED_TOKENS,), dtype=torch.int32, device="cuda")+2
        self.kv_indptr_buffer = torch.zeros((max_batch_size+2,), dtype=torch.int32, device="cuda")
        self.kv_indices_buffer = torch.zeros((MAX_PROCESSED_TOKENS,), dtype=torch.int64, device="cuda")
        self.qo_indptr_buffer = torch.zeros((max_batch_size+2,), dtype=torch.int32, device="cuda")
        self.mask_indptr_buffer = torch.zeros((max_batch_size+2,), dtype=torch.int32, device="cuda")

        self.graph_pool = torch.cuda.graph_pool_handle()

    def init_capture_graph(self):
        if self.server_args.disable_cuda_graph is False:
            self.capture_graph()

    def get_max_context_length(self):
        return self.model.config.max_position_embeddings
    
    def prepare_inputs(self, scheduled_batch: ScheduleBatch) -> Dict[str, Any]:
        batch_size = len(scheduled_batch)
        forward_batch = scheduled_batch.forward_batch
        padded_batch_size = batch_size + forward_batch.padded_token
        total_seq_len = forward_batch.kv_indices.shape[0]

        forward_batch.attn_logits = self.attn_logits
        forward_batch.attn_lse = self.attn_lse

        if forward_batch.forward_mode != ForwardMode.DECODE:
            ori_data = forward_batch.kv_indptr
            forward_batch.kv_indptr = self.kv_indptr_buffer[: batch_size + 1]
            forward_batch.kv_indptr[1:].copy_(ori_data, non_blocking=True)
            ori_data = forward_batch.kv_indices
            forward_batch.kv_indices = self.kv_indices_buffer[:total_seq_len]
            forward_batch.kv_indices.copy_(ori_data, non_blocking=True)
            ori_data = forward_batch.qo_indptr
            forward_batch.qo_indptr = self.qo_indptr_buffer[: batch_size + 1]
            forward_batch.qo_indptr[1:].copy_(ori_data, non_blocking=True)
        else:
            if self.is_draft:
                padded_batch_size = scheduled_batch.input_ids.shape[-1]
            ori_data = forward_batch.kv_indptr
            forward_batch.kv_indptr = self.kv_indptr_buffer[: padded_batch_size + 1]
            forward_batch.kv_indptr[1:].copy_(ori_data, non_blocking=True)
            ori_data = forward_batch.kv_indices
            forward_batch.kv_indices = self.kv_indices_buffer[:total_seq_len]
            forward_batch.kv_indices.copy_(ori_data, non_blocking=True)
            forward_batch.num_kv_splits = self.num_kv_splits_buffer[:padded_batch_size]
        
        if forward_batch.mask_indptr is not None:
            ori_data = forward_batch.mask_indptr
            forward_batch.mask_indptr = self.mask_indptr_buffer[: padded_batch_size + 1]
            forward_batch.mask_indptr[1:].copy_(ori_data, non_blocking=True)
        return {
            "input_ids": scheduled_batch.input_ids,
            "position_ids": scheduled_batch.position_ids,
            # "attention_mask": attention_mask,
            "forward_batch": forward_batch,
        }

    def prepare_replay(self, inputs: Dict[str, Any], scheduled_batch: ScheduleBatch):
        bs_size = len(scheduled_batch)
        forward_batch = inputs["forward_batch"]
        padded_bs_size = bs_size + forward_batch.padded_token
        self.input_ids[:padded_bs_size].copy_(inputs["input_ids"], non_blocking=True)
        self.position_ids[:padded_bs_size].copy_(
            inputs["position_ids"], non_blocking=True
        )
        self.out_cache_loc[:padded_bs_size].copy_(
            forward_batch.out_cache_loc, non_blocking=True
        )
        # self.attention_mask[:bs_size].copy_(inputs["attention_mask"], non_blocking=True)


    def forward(self, scheduled_batch: ScheduleBatch):
        forward_batch = scheduled_batch.forward_batch
        inputs = self.prepare_inputs(scheduled_batch)

        bs_size = len(scheduled_batch)
        pad_bs_size = forward_batch.padded_token + bs_size
        aux_hidden_states = None
        if (
            forward_batch.forward_mode == ForwardMode.DECODE
            and self.cuda_graphs.get(pad_bs_size) is not None
        ):
            self.prepare_replay(inputs, scheduled_batch)
            self.cuda_graphs[pad_bs_size].replay()
            logits, aux_hidden_states = self.output_logits[pad_bs_size]
        else:
            logits, aux_hidden_states = self.model(**inputs)
        if forward_batch.padded_token > 0:
            logits = logits[:bs_size]
            aux_hidden_states = aux_hidden_states[:bs_size] if aux_hidden_states is not None else None

        token_ids = (
            self.sampler(logits, forward_batch.sampling_batch_info) if self.rank == 0 else None
        )
        return BatchResult(token_ids, logits, aux_hidden_states)

    def profile_run(self, forward_batch: ForwardBatch = None):
        logger.info("Profiling run to reserve activation memory...")
        self.input_ids.fill_(0)
        self.position_ids.fill_(0)
        if forward_batch is None:
            forward_batch = ForwardBatch(None)
        qo_indptr_buffer = self.input_ids.view(dtype=torch.int32)
        forward_batch.qo_indptr = qo_indptr_buffer[: 1]
        forward_batch.qo_indptr[0] = 0
        with torch.no_grad():
            self.model(
                self.input_ids*0,
                position_ids=self.position_ids*0,
                forward_batch=forward_batch,
            )


    def capture_graph(self):
        memory_pool = self.block_memory_manager
        batch_size = 1
        forward_batch = ForwardBatch(memory_pool)
        forward_batch.attn_logits = self.attn_logits
        forward_batch.attn_lse = self.attn_lse
        input_ids = self.input_ids[:batch_size]
        forward_batch.kv_indptr = self.kv_indptr_buffer[: batch_size + 1]
        forward_batch.kv_indices = self.kv_indices_buffer[
            : input_ids.shape[0]
        ]
        forward_batch.num_kv_splits = self.num_kv_splits_buffer[:batch_size]
        forward_batch.forward_mode = ForwardMode.DECODE
        forward_batch.out_cache_loc = forward_batch.kv_indices
        self.compute_stream.synchronize()

        with torch.no_grad():
            self.model(
                self.input_ids[:batch_size],
                position_ids=self.position_ids[:batch_size],
                forward_batch=forward_batch,
            )

        for batch_size in tqdm.tqdm(self.capture_batch_size, desc="Capturing CUDA Graphs"):
            forward_batch.kv_indptr = self.kv_indptr_buffer[: batch_size + 1]
            forward_batch.out_cache_loc = self.out_cache_loc[:batch_size]
            torch.cuda.synchronize()
            cudagraph = torch.cuda.CUDAGraph()
            # attention_mask = torch.empty((batch_size), dtype=torch.long, device=self.device_id)
            with torch.cuda.graph(cudagraph, stream=self.compute_stream, pool=self.graph_pool):
                output = self.model(
                    self.input_ids[:batch_size],
                    position_ids=self.position_ids[:batch_size],
                    forward_batch=forward_batch,
                )
            torch.cuda.synchronize()
            self.output_logits[batch_size] = output
            self.cuda_graphs[batch_size] = cudagraph
            
        self.cuda_graphs[1].replay()
    
    def tokenize(self, prompt):
        return self.tokenizer.encode(prompt)


    def detokenize(self, tokens):
        return self.tokenizer.decode(
            tokens, skip_special_tokens=True, spaces_between_special_tokens=True
        )
