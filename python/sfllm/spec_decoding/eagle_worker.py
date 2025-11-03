

import torch
from typing import List
from sfllm.engine.shedule_batch import ScheduleBatch,BatchResult
from sfllm.engine.forward_params import ForwardMode, ForwardBatch
from sfllm.server_args import ServerArgs
from sfllm.engine.model_runner import ModelRunner
from sfllm.spec_decoding.spec_utils import (EagleSpecInput,
                                            EagleVerifyInput,
                                            select_top_k_tokens,
                                            fast_topk,
                                            organize_draft_results,
                                            build_tree_kernel_efficient)
import transformers


class EagleWorker:
    def __init__(self,server_args:ServerArgs):
        self.draft_model_runner = ModelRunner(server_args, is_draft=True)
        self.target_model_runner = ModelRunner(server_args)
        self.target_model_runner.model.set_eagle3_layers_to_capture()
        self.server_args = server_args
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        server_args.model_config = self.target_model_runner.model.config
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(server_args.model_path)
        self.draft_model_runner.wrap_attn_backend(self.target_model_runner.attn_logits, self.target_model_runner.attn_lse)

        #======init
        self.profile_run()
        self.target_model_runner.init_memory_pool()
        self.draft_model_runner.init_memory_pool()

        self.hot_token_id = self.draft_model_runner.model.hot_token_id.to("cuda")

        ####debug only
        self.draft_batch_out = None

    
    def profile_run(self):
        forward_batch = ForwardBatch(None)
        forward_batch.spec_info = EagleSpecInput()
        hs = self.draft_model_runner.model.config.hidden_size
        dtype = self.draft_model_runner.dtype
        hidden_states = torch.empty((64, hs*3), dtype=dtype, device="cuda")
        forward_batch.spec_info.hidden_states = hidden_states
        self.draft_model_runner.profile_run(forward_batch)
        self.target_model_runner.profile_run()
    
    @property
    def main_mem_pool(self):
        return self.target_model_runner.block_memory_manager


    def init_capture_graph(self):
        pass

    @torch.inference_mode()
    def forward(self, scheduled_batch:ScheduleBatch):
        # Implement the forward pass logic here
        if scheduled_batch.forward_batch.forward_mode == ForwardMode.EXTEND:
            batch_output = self.target_model_runner.forward(scheduled_batch)
            spec_info = EagleSpecInput(hidden_states=torch.concatenate(batch_output.aux_hidden_states,dim=-1))
            scheduled_batch.forward_batch.spec_info = spec_info
            draft_batch_out = self.draft_forward_extend(batch_output, scheduled_batch)
            spec_info.hidden_states =  torch.concatenate(draft_batch_out.aux_hidden_states,dim=-1)
            spec_info.next_token_logits = draft_batch_out.next_token_logits
            batch_output.spec_info = spec_info
            self.draft_batch_out = draft_batch_out
            return batch_output
        elif scheduled_batch.forward_batch.forward_mode == ForwardMode.DECODE:
            self.multi_step_speculative_decode(scheduled_batch)
    
    def draft_forward_extend(self, batch_output:BatchResult, scheduled_batch:ScheduleBatch):
        next_token_ids = batch_output.next_token_ids
        #build inputs for draft model
        input_ids = scheduled_batch.input_ids
        input_ids = torch.cat((input_ids[1:], next_token_ids), dim=-1)

        scheduled_batch.forward_batch.past_key_values = self.draft_model_runner.block_memory_manager.kv_buffers
        scheduled_batch.input_ids = input_ids
        
        token_allocator = self.draft_model_runner.block_memory_manager
        out_cache_loc = token_allocator.alloc_block([-1]*input_ids.shape[0], hashv=0)
        old_forward_batch = scheduled_batch.forward_batch

        forward_batch = ForwardBatch(token_allocator)
        forward_batch.out_cache_loc = torch.tensor(out_cache_loc, dtype=torch.int64, device="cuda")
        forward_batch.qo_indptr = old_forward_batch.qo_indptr[1:]
        forward_batch.kv_indptr = old_forward_batch.kv_indptr[1:]
        forward_batch.kv_indices = old_forward_batch.kv_indices
        forward_batch.spec_info = old_forward_batch.spec_info

        scheduled_batch.forward_batch = forward_batch
        draft_batch_out = self.draft_model_runner.forward(scheduled_batch)
        scheduled_batch.forward_batch = old_forward_batch
        return draft_batch_out

    def multi_step_speculative_decode(self, scheduled_batch:ScheduleBatch):
        verify_input = self.draft_propose(scheduled_batch)
        self.verify_propose(scheduled_batch, verify_input)
    
    def draft_propose(self, scheduled_batch:ScheduleBatch):
        # Implement the draft propose logic here

        # Return values
        score_list: List[torch.Tensor] = []
        token_list: List[torch.Tensor] = []
        parents_list: List[torch.Tensor] = []


        seq_len = scheduled_batch.position_ids
        #prepare kv cache loc
        orig_forward_batch = scheduled_batch.forward_batch
        attn_metadatas = []
        bs = scheduled_batch.input_ids.shape[0]
        total_tokens = bs * self.topk * self.speculative_num_steps
        token_allocator = self.draft_model_runner.block_memory_manager
        assert token_allocator.can_alloc(total_tokens)
        out_cache_loc = token_allocator.alloc_block([-1]*total_tokens, hashv=0)
        out_cache_loc_tensor = torch.tensor(out_cache_loc, dtype=torch.int32, device="cuda").view(self.speculative_num_steps, -1)
        cur_kv_indptr = scheduled_batch.forward_batch.kv_indptr[..., None]
        scheduled_batch.position_ids = scheduled_batch.position_ids.repeat_interleave(self.topk, dim=0)
        past_len = cur_kv_indptr[0, 0]
        for i in range(self.speculative_num_steps):
            forward_batch = ForwardBatch(self.draft_model_runner.block_memory_manager)
            forward_batch.forward_mode = ForwardMode.DECODE
            forward_batch.kv_indptr = cur_kv_indptr.expand(-1, self.topk).cumsum(dim=1).flatten()
            kv_indices = torch.arange(1, past_len+1+i, device="cuda")
            kv_indices = kv_indices[None].repeat(self.topk, 1)
            for tpki, sb_kv_indices in enumerate(kv_indices):
                sb_kv_indices[-i-1:] = out_cache_loc_tensor[tpki,:i+1]
            forward_batch.kv_indices = kv_indices.flatten()

            cur_kv_indptr = cur_kv_indptr + 1
            attn_metadatas.append(forward_batch)
        
        out_cache_loc = out_cache_loc_tensor
        out_cache_loc = out_cache_loc.reshape(
            bs, self.topk, self.speculative_num_steps
        )
        out_cache_loc = out_cache_loc.permute((2, 0, 1)).reshape(
            self.speculative_num_steps, -1
        )
        #TODO debug only
        draft_batch_out = self.draft_batch_out
        probs = torch.softmax(draft_batch_out.next_token_logits[-1:], dim=-1)
        hidden_states = draft_batch_out.aux_hidden_states[0][-1:]
        topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
        spec_info = EagleSpecInput(hidden_states=hidden_states)
        spec_info.verified_id = scheduled_batch.input_ids
        # Forward multiple steps
        scores = None
        for i in range(self.speculative_num_steps):
            input_ids, hidden_states, scores, tree_info, selected_input_index = select_top_k_tokens(
                i, topk_p, topk_index, hidden_states, scores, self.topk
            )
            score_list.append(tree_info[0])
            token_list.append(tree_info[1])
            parents_list.append(tree_info[2])

            # We don't need to run the last forward. we get 1 token from draft prefill and (#spec steps - 1) tokens here
            if i == self.speculative_num_steps - 1:
                break

            # Set inputs
            scheduled_batch.forward_batch = attn_metadatas[i]
            if i > 0:
                last_kv_indices = attn_metadatas[i-1].kv_indices.view(self.topk,-1)
                cur_kv_indices = attn_metadatas[i].kv_indices.view(self.topk,-1)
                cur_kv_indices[:, :-1] = last_kv_indices[selected_input_index]
            scheduled_batch.input_ids = input_ids
            scheduled_batch.forward_batch.out_cache_loc = out_cache_loc[i]
            scheduled_batch.position_ids.add_(1)
            spec_info.hidden_states = hidden_states
            scheduled_batch.forward_batch.spec_info = spec_info

            # Run forward
            logits_output = self.draft_model_runner.forward(scheduled_batch)
            probs = torch.softmax(logits_output.next_token_logits, dim=-1)
            topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
            if self.hot_token_id is not None:
                topk_index = self.hot_token_id[topk_index]
            hidden_states = logits_output.aux_hidden_states[0]

        parent_list, top_scores_index, draft_tokens = organize_draft_results(
            score_list, token_list, parents_list, self.speculative_num_draft_tokens
        )

        # return parent_list, top_scores_index, draft_tokens
        (
            tree_mask,
            position,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            draft_tokens,
        ) = build_tree_kernel_efficient(
            spec_info.verified_id,
            parent_list,
            top_scores_index,
            draft_tokens,
            seq_len,
            seq_len.sum(),
            self.topk,
            self.speculative_num_steps,
            self.speculative_num_draft_tokens,
        )
        return EagleVerifyInput(
            draft_token=draft_tokens,
            custom_mask=tree_mask,
            positions=position,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            retrive_cum_len=None,
            spec_steps=self.speculative_num_steps,
            topk=self.topk,
            draft_token_num=self.server_args.speculative_num_draft_tokens,
        )


    def verify_propose(self, scheduled_batch:ScheduleBatch, verify_input:EagleVerifyInput):
        # Implement the verification logic here
        # Forward
        batch_result = self.target_worker.forward_batch_generation(
            model_worker_batch, is_verify=True
        )