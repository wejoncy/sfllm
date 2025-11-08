
from sfllm.engine.forward_params import ForwardMode, ForwardBatch
from typing import Callable
import tqdm
import torch

class EagleCudaGraphRunner():
    def __init__(self, draft_model_runner, model_func:Callable):
        self.draft_model_runner = draft_model_runner
        self.model_func = model_func
        self.server_args = draft_model_runner.server_args
        self.topk = self.server_args.speculative_eagle_topk
        self.block_memory_manager = draft_model_runner.block_memory_manager
        self.compute_stream = draft_model_runner.compute_stream
        self.attn_logits = draft_model_runner.attn_logits
        self.attn_lse = draft_model_runner.attn_lse
        steps = self.server_args.speculative_num_steps - 1
        indptr_shape = list(draft_model_runner.kv_indptr_buffer.shape)
        indptr_shape.insert(0, steps)
        indices_shape = list(draft_model_runner.kv_indices_buffer.shape)
        indices_shape.insert(0, steps)
        self.kv_indptr_buffer = draft_model_runner.kv_indptr_buffer
        self.kv_indptr_buffer_s = torch.zeros(indptr_shape, dtype=torch.int32, device=draft_model_runner.device_id)
        self.kv_indices_buffer_s = torch.zeros(indices_shape, dtype=torch.long, device=draft_model_runner.device_id)
        self.qo_indptr_buffer = draft_model_runner.qo_indptr_buffer
        self.num_kv_splits_buffer = draft_model_runner.num_kv_splits_buffer
        self.hidden_states_buffer = draft_model_runner.hidden_states_buffer.view(-1, draft_model_runner.get_config().hidden_size)
        self.input_ids = draft_model_runner.input_ids
        self.position_ids = draft_model_runner.position_ids
        self.kv_indices_mtd_buffer = draft_model_runner.out_cache_loc
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
        forward_batch = ForwardBatch(None)
        forward_batch.seq_lens_sum = 0 # never used in cuda graph
        forward_batch.kv_indices_mtd = self.kv_indices_mtd_buffer[:batch_size] # fake, should be sequence length , but it does not matter
        forward_batch.kv_indptr = self.kv_indptr_buffer[:batch_size+1]
        # out_cache_loc_tensor = self.out_cache_loc_buffer[:batch_size*self.topk*running_steps]
        scheduled_batch.position_ids = self.position_ids[:batch_size]
        spec_info = EagleSpecInput(
            hidden_states=self.hidden_states_buffer[:batch_size], logits=self.logits_buffer[:batch_size],
        )
        scheduled_batch.forward_batch_spec = forward_batch
        scheduled_batch.forward_batch = forward_batch
        return spec_info, scheduled_batch

    @torch.inference_mode()
    def init_cuda_graph(self):
        spec_info, scheduled_batch = self.prepare_cudagraph_inputs_for_capture(batch_size=1)
        self.model_func(spec_info, scheduled_batch)
        self.compute_stream.synchronize()
        for batch_size in tqdm.tqdm(range(1, 32), desc="Capturing CUDA Graphs"):
            (spec_info, scheduled_batch) = self.prepare_cudagraph_inputs_for_capture(batch_size)
            cudagraph = torch.cuda.CUDAGraph()

            with torch.cuda.graph(cudagraph, stream=self.compute_stream, pool=self.graph_pool):
                output = self.model_func(spec_info, scheduled_batch)
            torch.cuda.synchronize()
            self.graph_outputs[batch_size] = output
            self.cuda_graphs[batch_size] = cudagraph
            
        self.cuda_graphs[1].replay()


    def prepare_replay(self, spec_info, scheduled_batch):
        batch_size = len(scheduled_batch)
        self.hidden_states_buffer[:batch_size].copy_(spec_info.hidden_states)
        self.logits_buffer[:batch_size].copy_(spec_info.logits)
        self.position_ids[:batch_size].copy_(scheduled_batch.position_ids)
        self.kv_indptr_buffer[:batch_size + 1].copy_(scheduled_batch.forward_batch.kv_indptr)
        self.kv_indices_mtd_buffer[:scheduled_batch.forward_batch_spec.kv_indices_mtd.shape[0]
            ].copy_(scheduled_batch.forward_batch_spec.kv_indices_mtd)

        # for i, forward_batch in enumerate(attn_metadatas):
        #     self.kv_indptr_buffer_s[i][: expand_batch_size + 1].copy_(forward_batch.kv_indptr)
        #     num_of_indices = forward_batch.kv_indices.numel() # for each step
        #     self.kv_indices_buffer_s[i][: num_of_indices].copy_(forward_batch.kv_indices)
        #     # forward_batch.qo_indptr = self.qo_indptr_buffer[: expand_batch_size + 1]
        #     # self.num_kv_splits_buffer[:expand_batch_size].copy_(forward_batch.num_kv_splits)


    @torch.inference_mode()
    def forward(self, spec_info, scheduled_batch):
        batch_size = len(scheduled_batch)
        if batch_size in self.cuda_graphs:
            self.prepare_replay(spec_info, scheduled_batch)
            self.cuda_graphs[batch_size].replay()
            output = self.graph_outputs[batch_size]
        else:
            output = self.model_func(spec_info, scheduled_batch)
        # we can't guarantee they are exactly the same for all steps
        # output1 = self.model_func(spec_info, scheduled_batch)
        # assert all([torch.allclose(o1, o2) for o1, o2 in zip(output1, output)])
        return output