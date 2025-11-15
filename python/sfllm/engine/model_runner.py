import logging
import torch
import bisect
import tqdm
import transformers
import gc
from contextlib import contextmanager
from typing import Dict, List, Any

from sfllm.model_loader.model_loader import initialize_model
from sfllm.engine.schedule_batch import ScheduleBatch,BatchResult
from sfllm.engine.forward_params import ForwardMode, ForwardBatch
from sfllm.engine.memory_pool import BlockMemoryManager
from sfllm.layers.sampler import Sampler
from sfllm.server_args import ServerArgs
from sfllm.utils.nutils import DEFAULT_CUDA_GRAPH_BATCH_SIZES, MAX_PROCESSED_TOKENS
import sfllm.utils.nutils as nutils

logger = logging.getLogger(__name__)

@contextmanager
def freeze_gc(enable_cudagraph_gc: bool):
    """
    Optimize garbage collection during CUDA graph capture.
    Clean up, then freeze all remaining objects from being included
    in future collections if GC is disabled during capture.
    """
    gc.collect()
    should_freeze = not enable_cudagraph_gc
    if should_freeze:
        gc.freeze()
    try:
        yield
    finally:
        if should_freeze:
            gc.unfreeze()
            gc.collect()


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

        expand_scale = server_args.speculative_num_steps * server_args.speculative_eagle_topk if is_draft else server_args.speculative_num_draft_tokens
        max_batch_size = server_args.cuda_graph_max_bs*expand_scale
        self.input_ids = torch.empty((max_batch_size,), dtype=torch.long, device=self.device_id)
        self.position_ids = torch.empty((max_batch_size,), dtype=torch.long, device=self.device_id)
        self.out_cache_loc = torch.zeros((max_batch_size,), dtype=torch.int64, device=self.device_id)
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

    def init_memory_pool(self, num_blocks: int = None):
        self.block_memory_manager = BlockMemoryManager(
            self.server_args, self.get_config(), num_blocks=num_blocks
        )

    def wrap_target_model(self, target_model_runner: 'ModelRunner'):
        self.attn_logits = target_model_runner.attn_logits
        self.attn_lse = target_model_runner.attn_lse
        self.compute_stream = target_model_runner.compute_stream
        self.graph_pool = target_model_runner.graph_pool

    def create_cudagraph_buffers(self):
        # cuda graph
        device_id = self.device_id
        server_args = self.server_args
        if not self.is_draft:
            self.compute_stream = torch.cuda.Stream(device=device_id)
            self.graph_pool = torch.cuda.graph_pool_handle()
        self.cuda_graph_max_bs = server_args.cuda_graph_max_bs
        ind = bisect.bisect_right(DEFAULT_CUDA_GRAPH_BATCH_SIZES, self.cuda_graph_max_bs)
        self.capture_batch_size = DEFAULT_CUDA_GRAPH_BATCH_SIZES[:ind]
        self.capture_batch_size = self.capture_batch_size[:ind]

        expand_scale = self.server_args.speculative_num_steps* self.server_args.speculative_eagle_topk if self.is_draft else 1
        max_batch_size = max(self.capture_batch_size)*expand_scale
        # for draft extend, out_cache_loc size = batch_size * speculative_num_steps*topk
        self.output_logits = {}
        self.output_logits_target_verify = {}
        self.output_logits_extend = {}
        self.cuda_graphs = {}
        self.cuda_graphs_extend = {}
        self.cuda_graphs_target_verify = {}

        self.num_kv_splits_buffer = torch.zeros((MAX_PROCESSED_TOKENS,), dtype=torch.int32, device="cuda")+2
        self.kv_indptr_buffer = torch.zeros((max_batch_size+2,), dtype=torch.int32, device="cuda")
        self.kv_indices_buffer = torch.zeros((MAX_PROCESSED_TOKENS,), dtype=torch.int64, device="cuda")
        self.qo_indptr_buffer = torch.zeros((max_batch_size+4,), dtype=torch.int32, device="cuda")
        self.mask_indptr_buffer = torch.zeros((max_batch_size+2,), dtype=torch.int32, device="cuda")
        self.custom_mask_buffer = torch.zeros((MAX_PROCESSED_TOKENS*4096//200+2,), dtype=torch.bool, device="cuda")

        if self.is_draft:
            self.hidden_states_buffer = torch.empty(
                (max_batch_size*(self.server_args.speculative_num_steps+1), self.get_config().hidden_size*3),
                dtype=self.dtype,
                device=self.device_id
            )

    def init_capture_cudagraph(self, forward_mode: ForwardMode = ForwardMode.DECODE):
        if self.server_args.disable_cuda_graph is False:
            if forward_mode == ForwardMode.DRAFT_EXTEND:
                self.capture_cudagraph_draft_extend()
            elif forward_mode == ForwardMode.TARGET_VERIFY:
                self.capture_cudagraph_target_verify()
            else:
                self.capture_cudagraph_decode()

    def get_max_context_length(self):
        return self.model.config.max_position_embeddings
    
    def prepare_inputs(self, scheduled_batch: ScheduleBatch) -> Dict[str, Any]:
        batch_size = len(scheduled_batch)
        forward_batch = scheduled_batch.forward_batch
        padded_batch_size = batch_size + forward_batch.padded_token
        forward_batch.attn_logits = self.attn_logits
        forward_batch.attn_lse = self.attn_lse
        forward_batch.num_kv_splits = self.num_kv_splits_buffer[:padded_batch_size]

        assert forward_batch.out_cache_loc is None or forward_batch.out_cache_loc.dtype == torch.long
        assert forward_batch.qo_indptr is None or forward_batch.qo_indptr.dtype == torch.int32
        assert forward_batch.kv_indptr is None or forward_batch.kv_indptr.dtype == torch.int32
        assert forward_batch.mask_indptr is None or forward_batch.mask_indptr.dtype == torch.int32
        assert forward_batch.kv_indices is None or forward_batch.kv_indices.dtype == torch.long
        assert forward_batch.custom_mask is None or forward_batch.custom_mask.dtype == torch.bool


    def prepare_replay(self, scheduled_batch: ScheduleBatch):
        batch_size = len(scheduled_batch)
        forward_batch = scheduled_batch.forward_batch
        total_seq_len = forward_batch.kv_indices.shape[0]
        padded_bs_size = batch_size + forward_batch.padded_token
        input_ids = scheduled_batch.input_ids
        position_ids = scheduled_batch.position_ids
        num_tokens = input_ids.shape[0]
        self.input_ids[:num_tokens].copy_(input_ids)
        self.position_ids[:num_tokens].copy_(position_ids)
        self.out_cache_loc[:num_tokens].copy_(forward_batch.out_cache_loc)
        self.kv_indices_buffer[:total_seq_len].copy_(forward_batch.kv_indices)
        self.kv_indptr_buffer[: padded_bs_size + 1].copy_(forward_batch.kv_indptr)
        if forward_batch.forward_mode == ForwardMode.DRAFT_EXTEND:
            padded_token_nums = (self.server_args.speculative_num_steps+1)*padded_bs_size
            self.input_ids[num_tokens:padded_token_nums].fill_(0)
            self.position_ids[num_tokens:padded_token_nums].fill_(0)
            self.out_cache_loc[num_tokens:padded_token_nums].fill_(0)
            self.qo_indptr_buffer[: padded_bs_size + 1].copy_(forward_batch.qo_indptr)
            self.hidden_states_buffer[:num_tokens].copy_(forward_batch.spec_info.hidden_states)
        elif forward_batch.forward_mode == ForwardMode.TARGET_VERIFY:
            self.qo_indptr_buffer[: padded_bs_size + 1].copy_(forward_batch.qo_indptr)
            # shape:seq_lens_sum * num_verify_tokens+ num_verify_tokens * num_verify_tokens * bs,
            self.custom_mask_buffer[:forward_batch.custom_mask.shape[0]].copy_(forward_batch.custom_mask)
            self.mask_indptr_buffer[: padded_bs_size + 1].copy_(forward_batch.mask_indptr)            


    def profile_run(self, forward_batch: ForwardBatch = None):
        logger.info("Profiling run to reserve activation memory...")
        profile_batch = 64
        self.input_ids.fill_(0)
        self.position_ids.fill_(0)
        if forward_batch is None:
            forward_batch = ForwardBatch(None)
        qo_indptr_buffer = self.input_ids.view(dtype=torch.int32)
        forward_batch.qo_indptr = qo_indptr_buffer[: 1]
        forward_batch.qo_indptr[0] = 0
        with torch.no_grad():
            self.model(
                self.input_ids[:profile_batch]*0,
                position_ids=self.position_ids[:profile_batch]*0,
                forward_batch=forward_batch,
            )

    @torch.inference_mode()
    def capture_cudagraph_decode(self):
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

        self.model(
            self.input_ids[:batch_size],
            position_ids=self.position_ids[:batch_size],
            forward_batch=forward_batch,
        )
        with freeze_gc(False):
            for batch_size in tqdm.tqdm(list(reversed(self.capture_batch_size)), desc="Capturing CUDA Graphs"):
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

    @torch.inference_mode()
    def capture_cudagraph_target_verify(self):
        draft_tokens_expand = self.server_args.speculative_num_draft_tokens
        memory_pool = self.block_memory_manager
        batch_size = 3
        forward_batch = ForwardBatch(memory_pool)
        forward_batch.forward_mode = ForwardMode.TARGET_VERIFY
        # forward_batch.attn_logits = self.attn_logits
        # forward_batch.attn_lse = self.attn_lse

        forward_batch.qo_indptr = self.qo_indptr_buffer[: batch_size + 1]
        forward_batch.kv_indptr = self.kv_indptr_buffer[: batch_size + 1]
        forward_batch.kv_indices = self.kv_indices_buffer[: batch_size + 1]
        forward_batch.num_kv_splits = self.num_kv_splits_buffer[:batch_size]
        forward_batch.out_cache_loc = self.out_cache_loc[:batch_size*draft_tokens_expand]
        forward_batch.custom_mask = self.custom_mask_buffer[: batch_size]
        forward_batch.mask_indptr = self.mask_indptr_buffer[: batch_size + 1]
        forward_batch.max_extend_len = draft_tokens_expand
        forward_batch.max_kv_split = 16
        self.compute_stream.synchronize()

        self.model(
            self.input_ids[:batch_size*draft_tokens_expand],
            position_ids=self.position_ids[:batch_size*draft_tokens_expand],
            forward_batch=forward_batch,
        )

        for batch_size in tqdm.tqdm(list(reversed(self.capture_batch_size)), desc="Capturing CUDA Graphs"):
            forward_batch.kv_indptr = self.kv_indptr_buffer[: batch_size + 1]
            forward_batch.qo_indptr = self.qo_indptr_buffer[: batch_size + 1]
            forward_batch.out_cache_loc = self.out_cache_loc[:batch_size*draft_tokens_expand]
            torch.cuda.synchronize()
            cudagraph = torch.cuda.CUDAGraph()
            # attention_mask = torch.empty((batch_size), dtype=torch.long, device=self.device_id)
            with torch.cuda.graph(cudagraph, stream=self.compute_stream, pool=self.graph_pool):
                output = self.model(
                    self.input_ids[:batch_size*draft_tokens_expand],
                    position_ids=self.position_ids[:batch_size*draft_tokens_expand],
                    forward_batch=forward_batch,
                )
            torch.cuda.synchronize()
            self.output_logits_target_verify[batch_size] = output
            self.cuda_graphs_target_verify[batch_size] = cudagraph
            
        self.cuda_graphs_target_verify[1].replay()
    
    @torch.inference_mode()
    def capture_cudagraph_draft_extend(self):
        from sfllm.spec_decoding.spec_utils import EagleSpecInput
        memory_pool = self.block_memory_manager
        batch_size = 1
        token_nums = batch_size*(1+self.server_args.speculative_num_steps)
        forward_batch = ForwardBatch(memory_pool)
        # forward_batch.attn_logits = self.attn_logits
        # forward_batch.attn_lse = self.attn_lse
        forward_batch.kv_indptr = self.kv_indptr_buffer[: batch_size + 1]
        forward_batch.kv_indices = self.kv_indices_buffer[: batch_size]
        forward_batch.qo_indptr = self.qo_indptr_buffer[: batch_size + 1]
        forward_batch.num_kv_splits = self.num_kv_splits_buffer[:batch_size]
        forward_batch.forward_mode = ForwardMode.DRAFT_EXTEND
        forward_batch.max_extend_len = self.server_args.speculative_num_steps+1
        forward_batch.out_cache_loc = self.out_cache_loc[:token_nums]

        if self.is_draft:
            forward_batch.spec_info = EagleSpecInput(
                hidden_states=self.hidden_states_buffer[:token_nums],
            )

        self.model(
            self.input_ids[:token_nums],
            position_ids=self.position_ids[:token_nums],
            forward_batch=forward_batch,
        )
        self.compute_stream.synchronize()
        for batch_size in tqdm.tqdm(list(reversed(range(1, 32))), desc="Capturing CUDA Graphs"):
            token_nums = batch_size*(1+self.server_args.speculative_num_steps)
            forward_batch.kv_indptr = self.kv_indptr_buffer[: batch_size + 1]
            forward_batch.kv_indices = self.kv_indices_buffer
            forward_batch.out_cache_loc = self.out_cache_loc[:token_nums]
            forward_batch.qo_indptr = self.qo_indptr_buffer[: batch_size + 1]
            if self.is_draft:
                forward_batch.spec_info.hidden_states = self.hidden_states_buffer[:token_nums]

            cudagraph = torch.cuda.CUDAGraph()

            with torch.cuda.graph(cudagraph, stream=self.compute_stream, pool=self.graph_pool):
                output = self.model(
                    self.input_ids[:token_nums],
                    position_ids=self.position_ids[:token_nums],
                    forward_batch=forward_batch,
                )
            torch.cuda.synchronize()
            self.output_logits_extend[batch_size] = output
            self.cuda_graphs_extend[batch_size] = cudagraph
            
        self.cuda_graphs_extend[1].replay()
    
    def tokenize(self, prompt):
        return self.tokenizer.encode(prompt)


    def detokenize(self, tokens):
        return self.tokenizer.decode(
            tokens, skip_special_tokens=True, spaces_between_special_tokens=True
        )

    @torch.inference_mode()
    def forward(self, scheduled_batch: ScheduleBatch):
        forward_batch = scheduled_batch.forward_batch
        self.prepare_inputs(scheduled_batch)

        bs_size = len(scheduled_batch)
        num_tokens = scheduled_batch.input_ids.shape[0]
        pad_bs_size = forward_batch.padded_token + bs_size
        aux_hidden_states = None
        if (
            forward_batch.forward_mode == ForwardMode.DECODE
            and self.cuda_graphs.get(pad_bs_size) is not None
        ):
            self.prepare_replay(scheduled_batch)
            self.cuda_graphs[pad_bs_size].replay()
            logits, aux_hidden_states = self.output_logits[pad_bs_size]
        elif (forward_batch.forward_mode == ForwardMode.DRAFT_EXTEND
              and self.cuda_graphs_extend.get(pad_bs_size) is not None
              ):
            self.prepare_replay(scheduled_batch)
            self.cuda_graphs_extend[pad_bs_size].replay()
            logits, aux_hidden_states = self.output_logits_extend[pad_bs_size]
            aux_hidden_states = [i[:num_tokens] for i in aux_hidden_states]
        elif (forward_batch.forward_mode == ForwardMode.TARGET_VERIFY
              and self.cuda_graphs_target_verify.get(pad_bs_size) is not None
              ):
            self.prepare_replay(scheduled_batch)
            self.cuda_graphs_target_verify[pad_bs_size].replay()
            logits, aux_hidden_states = self.output_logits_target_verify[pad_bs_size]
        else:
            logits, aux_hidden_states = self.model(input_ids=scheduled_batch.input_ids,
                                                   position_ids=scheduled_batch.position_ids,
                                                   forward_batch=forward_batch)

        if self.server_args.enable_debug and not torch.cuda.is_current_stream_capturing():  # print debug log
            # debug mode to compare with non-cuda graph results
            logits_ref, aux_hidden_states_ref = self.model(input_ids=scheduled_batch.input_ids,
                                                           position_ids=scheduled_batch.position_ids,
                                                           forward_batch=forward_batch)
            assert torch.allclose(logits, logits_ref, atol=2e-2)
            if aux_hidden_states is not None:
                for h1, h2 in zip(aux_hidden_states, aux_hidden_states_ref):
                    assert torch.allclose(h1, h2, atol=8e-2)
        
        if forward_batch.padded_token > 0:
            logits = logits[:bs_size]
            aux_hidden_states = aux_hidden_states[:bs_size] if aux_hidden_states is not None else None

        token_ids = (
            self.sampler(
                logits, forward_batch.sampling_batch_info) if self.rank == 0 else None
        )
        return BatchResult(token_ids, logits, aux_hidden_states)
