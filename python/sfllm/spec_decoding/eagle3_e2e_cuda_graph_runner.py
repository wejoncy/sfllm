
from sfllm.engine.forward_params import ForwardMode, ForwardBatch
from typing import Callable
import tqdm
import torch

class EagleE2ECudaGraphRunner():
    def __init__(self, draft_model_runner, target_model_runner, model_func:Callable):
        self.device_id = draft_model_runner.device_id
        self.draft_model_runner = draft_model_runner
        self.target_model_runner = target_model_runner
        self.model_func = model_func
        self.server_args = draft_model_runner.server_args
        self.topk = self.server_args.speculative_eagle_topk
        self.spec_mem_pool = draft_model_runner.block_memory_manager
        self.target_mem_pool = target_model_runner.block_memory_manager
        self.compute_stream = draft_model_runner.compute_stream


        steps = self.server_args.speculative_num_steps - 1
        indptr_shape = list(draft_model_runner.kv_indptr_buffer.shape)
        indptr_shape.insert(0, steps)
        indices_shape = list(draft_model_runner.kv_indices_buffer.shape)
        indices_shape.insert(0, steps)
        self.spec_kv_indptr_buffer = draft_model_runner.kv_indptr_buffer
        self.spec_kv_indptr_buffer_s = torch.zeros(indptr_shape, dtype=torch.int32, device=draft_model_runner.device_id)
        self.spec_kv_indices_buffer_s = torch.zeros(indices_shape, dtype=torch.long, device=draft_model_runner.device_id)
        self.spec_qo_indptr_buffer = draft_model_runner.qo_indptr_buffer
        self.spec_kv_indices_buffer = draft_model_runner.kv_indices_buffer
        self.num_kv_splits_buffer = draft_model_runner.num_kv_splits_buffer
        self.hidden_states_buffer = draft_model_runner.hidden_states_buffer
        self.input_ids = draft_model_runner.input_ids
        self.verified_id = torch.zeros((4096,), dtype=torch.long, device=self.device_id)
        self.position_ids = draft_model_runner.position_ids
        self.spec_kv_indices_mtd_buffer = torch.zeros_like(draft_model_runner.kv_indices_buffer)
        self.spec_out_cache_loc = draft_model_runner.out_cache_loc
        self.spec_position_ids_extend = torch.zeros_like(self.position_ids)
        self.accept_length_buffer = torch.zeros((128,), dtype=torch.int64, device=draft_model_runner.device_id)
        #==================
        self.kv_indptr_buffer = target_model_runner.kv_indptr_buffer
        self.kv_indices_buffer = target_model_runner.kv_indices_buffer
        self.qo_indptr_buffer = target_model_runner.qo_indptr_buffer
        self.out_cache_loc = target_model_runner.out_cache_loc
        self.mask_indptr_buffer = target_model_runner.mask_indptr_buffer
        self.seq_lens_buffer = torch.zeros((128,), dtype=torch.int64, device="cuda")
        self.cuda_graphs = {}
        self.graph_pool  = draft_model_runner.graph_pool
        self.graph_outputs = {}
        # new buffers
        self.logits_buffer = torch.empty(
            (self.server_args.cuda_graph_max_bs, 
             draft_model_runner.get_config().draft_vocab_size), dtype=torch.float16, device=draft_model_runner.device_id)

    def prepare_cudagraph_inputs_for_capture(self, batch_size:int):
        from sfllm.spec_decoding.spec_utils import EagleSpecInput
        from sfllm.engine.schedule_batch import ScheduleBatch

        scheduled_batch = ScheduleBatch([0]*batch_size, None)
        spec_forward_batch = ForwardBatch(self.spec_mem_pool)
        # for draft extend
        token_nums = batch_size*(1+self.server_args.speculative_num_steps)
        spec_info = EagleSpecInput(hidden_states=self.hidden_states_buffer[:token_nums])
        spec_info.verified_id = self.verified_id[:token_nums]  # input_ids
        spec_forward_batch.position_ids_extend = self.spec_position_ids_extend[:token_nums]# position_ids
        spec_forward_batch.out_cache_loc = self.spec_out_cache_loc[:token_nums] 
        spec_forward_batch.kv_indptr = self.spec_kv_indptr_buffer[: batch_size + 1]
        spec_forward_batch.kv_indices = self.spec_kv_indices_buffer[: batch_size] # ....
        spec_forward_batch.qo_indptr = self.spec_qo_indptr_buffer[: batch_size + 1]
        spec_forward_batch.num_kv_splits = self.num_kv_splits_buffer[:batch_size]
        spec_forward_batch.forward_mode = ForwardMode.DRAFT_EXTEND
        spec_forward_batch.max_extend_len = self.server_args.speculative_num_steps+1
        spec_forward_batch.seq_lens_sum = 0 # never used in cuda graph

        # multi-step draft decode
        scheduled_batch.position_ids = self.position_ids[:batch_size]
        spec_forward_batch.kv_indices_mtd = self.spec_kv_indices_mtd_buffer[:batch_size] # fake, should be sequence length , but it does not matter
        spec_forward_batch.spec_info = spec_info
        spec_info.accept_length = self.accept_length_buffer[:batch_size]
        # target_forward_batch.kv_indptr = self.kv_indptr_buffer[: batch_size + 1] # share with verify
        # for build_tree_kernel_efficient
        scheduled_batch.input_ids = self.input_ids[:batch_size]


        # target forward_batch
        draft_tokens_expand = self.server_args.speculative_num_draft_tokens
        target_forward_batch = ForwardBatch(self.target_mem_pool)

        target_forward_batch.forward_mode = ForwardMode.TARGET_VERIFY
        target_forward_batch.qo_indptr = self.qo_indptr_buffer[: batch_size + 1]
        target_forward_batch.kv_indptr = self.kv_indptr_buffer[: batch_size + 1]
        target_forward_batch.kv_indices = self.kv_indices_buffer[: batch_size + 1]#...
        target_forward_batch.num_kv_splits = self.num_kv_splits_buffer[:batch_size]
        target_forward_batch.out_cache_loc = self.out_cache_loc[:batch_size*draft_tokens_expand]
        # target_forward_batch.custom_mask = self.custom_mask_buffer[: batch_size]
        target_forward_batch.mask_indptr = self.mask_indptr_buffer[: batch_size + 1]
        target_forward_batch.max_extend_len = draft_tokens_expand
        target_forward_batch.max_kv_split = 16
        target_forward_batch.seq_lens_sum = 16
        target_forward_batch.seq_lens = self.seq_lens_buffer[:batch_size]

        scheduled_batch.forward_batch_spec = spec_forward_batch
        scheduled_batch.forward_batch = target_forward_batch
        scheduled_batch.spec_info = spec_info
        return scheduled_batch

    @torch.inference_mode()
    def init_cuda_graph(self):
        from sfllm.engine.model_runner import freeze_gc
        scheduled_batch = self.prepare_cudagraph_inputs_for_capture(batch_size=1)
        old_DEBUG = self.server_args.enable_debug
        self.server_args.enable_debug = False
        self.model_func(scheduled_batch)
        self.server_args.enable_debug = old_DEBUG
        self.compute_stream.synchronize()
        with freeze_gc(False):
            for batch_size in tqdm.tqdm(list(reversed(range(1, 32))), desc="Capturing CUDA Graphs"):
                (scheduled_batch) = self.prepare_cudagraph_inputs_for_capture(batch_size)
                cudagraph = torch.cuda.CUDAGraph()

                with torch.cuda.graph(cudagraph, stream=self.compute_stream, pool=self.graph_pool):
                    output = self.model_func(scheduled_batch)
                torch.cuda.synchronize()
                self.graph_outputs[batch_size] = output
                self.cuda_graphs[batch_size] = cudagraph
           
        self.cuda_graphs[1].replay()

    # @torch.compile
    def prepare_replay(self, scheduled_batch):
        batch_size = len(scheduled_batch)
        spec_info = scheduled_batch.spec_info
        # for draft extend
        pad_token_nums = batch_size*(1+self.server_args.speculative_num_steps)
        token_nums = spec_info.verified_id.shape[-1]
        forward_batch_spec = scheduled_batch.forward_batch_spec
        self.verified_id[:token_nums].copy_(spec_info.verified_id)
        self.verified_id[token_nums:pad_token_nums].fill_(0)
        self.spec_position_ids_extend[:token_nums].copy_(forward_batch_spec.position_ids_extend)
        self.spec_position_ids_extend[token_nums:pad_token_nums].fill_(0)
        self.spec_out_cache_loc[:token_nums].copy_(forward_batch_spec.out_cache_loc)
        self.spec_out_cache_loc[token_nums:pad_token_nums].fill_(0)
        self.spec_kv_indptr_buffer[: batch_size + 1].copy_(forward_batch_spec.kv_indptr)
        ind_size = scheduled_batch.forward_batch_spec.kv_indices.shape[0]
        self.spec_kv_indices_buffer[: ind_size].copy_(forward_batch_spec.kv_indices)
        self.spec_qo_indptr_buffer[: batch_size + 1].copy_(forward_batch_spec.qo_indptr)
        self.hidden_states_buffer[:token_nums].copy_(spec_info.hidden_states)

        # multi-step draft decode
        self.position_ids[:batch_size].copy_(scheduled_batch.position_ids)
        ind_size_mtd = scheduled_batch.forward_batch_spec.kv_indices_mtd.shape[0]
        self.spec_kv_indices_mtd_buffer[:ind_size_mtd].copy_(forward_batch_spec.kv_indices_mtd)
        self.accept_length_buffer[:batch_size].copy_(spec_info.accept_length)
        # for build_tree_kernel_efficient
        self.input_ids[:batch_size].copy_(scheduled_batch.input_ids)

        # target verify

        verify_forward_batch = scheduled_batch.forward_batch
        self.qo_indptr_buffer[:batch_size+1].copy_(verify_forward_batch.qo_indptr)
        self.kv_indptr_buffer[:batch_size+1].copy_(verify_forward_batch.kv_indptr)
        ind_size_verify = verify_forward_batch.kv_indices.shape[0]
        self.kv_indices_buffer[:ind_size_verify].copy_(verify_forward_batch.kv_indices)
        self.out_cache_loc[:batch_size*self.server_args.speculative_num_draft_tokens].copy_(
            verify_forward_batch.out_cache_loc)
        self.mask_indptr_buffer[:batch_size+1].copy_(verify_forward_batch.mask_indptr)
        self.seq_lens_buffer[:batch_size].copy_(verify_forward_batch.seq_lens)


    @torch.inference_mode()
    def forward(self, scheduled_batch):
        batch_size = len(scheduled_batch)
        if batch_size in self.cuda_graphs:
            self.prepare_replay(scheduled_batch)
            self.cuda_graphs[batch_size].replay()
            output = self.graph_outputs[batch_size]
        else:
            output = self.model_func(scheduled_batch)
        # we can't guarantee they are exactly the same for all steps
        # output1 = self.model_func(spec_info, scheduled_batch)
        # assert all([torch.allclose(o1, o2) for o1, o2 in zip(output1, output)])
        return output