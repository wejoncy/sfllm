

import logging
import torch
from typing import List
from sfllm.engine.schedule_batch import ScheduleBatch,BatchResult
from sfllm.engine.forward_params import ForwardMode, ForwardBatch
from sfllm.server_args import ServerArgs
from sfllm.engine.model_runner import ModelRunner
from sfllm.spec_decoding.spec_utils import (EagleSpecInput,
                                            EagleVerifyInput,
                                            select_top_k_tokens,
                                            fast_topk,
                                            organize_draft_results,
                                            build_tree_kernel_efficient,
                                            generate_kv_indices_for_mtd)
from sfllm.spec_decoding.draft_cuda_graph_runner import EagleCudaGraphRunner
from sfllm.spec_decoding.eagle3_e2e_cuda_graph_runner import EagleE2ECudaGraphRunner
import transformers

ALIGN_EAGLE_WITH_SGLANG_ = False
logger = logging.getLogger(__name__)

for_comparation = None
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
        self.draft_model_runner.wrap_target_model(self.target_model_runner)
        self.detokenize = self.target_model_runner.detokenize
        self.compute_stream = self.target_model_runner.compute_stream
        self.dtype = self.target_model_runner.dtype
        self.config = self.target_model_runner.model.config

        #statistics
        self.total_accepted_tokens = 0
        
        #eagle3 draft model share embedding with target model
        embed, head = self.target_model_runner.model.get_embed_and_head()
        if (
            hasattr(self.draft_model_runner.model, "load_lm_head_from_target")
            and self.draft_model_runner.model.load_lm_head_from_target
        ):
            self.draft_model_runner.model.set_embed_and_head(embed, head)
        else:
            self.draft_model_runner.model.set_embed(embed)

        #======init
        self.profile_run()
        self.target_model_runner.init_memory_pool()
        self.draft_model_runner.init_memory_pool(num_blocks=self.target_model_runner.block_memory_manager.num_blocks)
        
        self.draft_decode_forward_runner = EagleCudaGraphRunner(self.draft_model_runner, self.draft_decode_forward)
        self.eagle_e2e_runner = EagleE2ECudaGraphRunner(self.draft_model_runner, self.target_model_runner, 
                                                                   self.forward_decode_e2e)

        self.hot_token_id = self.draft_model_runner.model.hot_token_id.to("cuda")
        self.attn_metadatas = []
        # cur_kv_seqlen = seq_lens[..., None] + 1
        for i in range(self.speculative_num_steps - 1):
            forward_batch = ForwardBatch(self.draft_mem_pool)
            forward_batch.forward_mode = ForwardMode.DECODE
            self.attn_metadatas.append(forward_batch)

        # prealloc out_cache_loc for draft propose
        total_tokens = self.server_args.cuda_graph_max_bs * self.topk * (self.speculative_num_steps - 1)
        token_allocator = self.draft_mem_pool
        assert token_allocator.can_alloc(total_tokens)
        out_cache_loc = token_allocator.persist_alloc_block_from_rear(total_tokens)
        self.prealloc_out_cache_loc_tensor = torch.tensor(out_cache_loc, dtype=torch.long, device="cuda")

    
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

    @property
    def draft_mem_pool(self):
        return self.draft_model_runner.block_memory_manager

    def init_capture_cudagraph(self):
        if not self.server_args.disable_cuda_graph:
            # self.eagle_e2e_runner.init_cuda_graph()
            # self.target_model_runner.init_capture_cudagraph(forward_mode=ForwardMode.TARGET_VERIFY)
            # self.draft_model_runner.init_capture_cudagraph(forward_mode=ForwardMode.DRAFT_EXTEND)
            # self.draft_decode_forward_runner.init_cuda_graph()
            pass
        
    @torch.inference_mode()
    def forward(self, scheduled_batch:ScheduleBatch):
        # Implement the forward pass logic here
        if scheduled_batch.forward_batch.forward_mode == ForwardMode.EXTEND:
            batch_output = self.target_model_runner.forward(scheduled_batch)
            spec_info = EagleSpecInput()
            spec_info.hash = scheduled_batch.spec_info.hash
            scheduled_batch.spec_info = spec_info
            spec_info.hidden_states = torch.concatenate(batch_output.aux_hidden_states, dim=-1)
            scheduled_batch.forward_batch_spec.spec_info = spec_info
            self.draft_forward_extend(batch_output, scheduled_batch)

            spec_info.verified_id = batch_output.next_token_ids
            spec_info.accept_length = torch.zeros((len(scheduled_batch),), dtype=torch.int32, device=spec_info.logits.device) - 1
            # spec_info.accept_index = torch.ones((len(scheduled_batch),), dtype=torch.int32, device=spec_info.logits.device)
            batch_output.spec_info = spec_info
            return batch_output
        elif scheduled_batch.forward_batch.forward_mode == ForwardMode.DECODE:
            if scheduled_batch.spec_info.hidden_states.shape[-1] > self.target_model_runner.get_config().hidden_size:
                # forward_decode_e2e
                global for_comparation
                for_comparation = self.eagle_e2e_runner.forward(scheduled_batch)
                out_args = for_comparation
            elif scheduled_batch.spec_info.hidden_states.shape[-1] == self.target_model_runner.get_config().hidden_size:
                with torch.cuda.nvtx.range("eagle_spec_decode"):
                    out_args = self.multi_step_speculative_decode(scheduled_batch)

            def diff_tensors(a_tuple, b_tuple):
                if len(a_tuple) != len(b_tuple):
                    raise ValueError(
                        f"Tuple length mismatch: {len(a_tuple)} vs {len(b_tuple)}"
                    )

                for i, (ta, tb) in enumerate(zip(a_tuple, b_tuple)):
                    if i == 0:
                        ta = ta.hidden_states
                        tb = tb.hidden_states
                    if not torch.equal(ta, tb):
                        print(f"Tensor at index {i} is different.")
            # if for_comparation is not None:
            #     diff_tensors(for_comparation, out_args)
            return self.forward_decode_e2e_post_process(scheduled_batch, *out_args)
    
    def draft_forward_extend(self, batch_output:BatchResult, scheduled_batch:ScheduleBatch):
        next_token_ids = batch_output.next_token_ids
        #build inputs for draft model
        o_input_ids = scheduled_batch.input_ids
        pt = 0
        for i, extend_len in enumerate(scheduled_batch.forward_batch.extend_lens_list):
            input_ids = o_input_ids[pt : pt + extend_len]
            o_input_ids[pt : pt + extend_len] = torch.cat(
                (input_ids[1:], next_token_ids[i].reshape(1))
            )
            pt += extend_len
        with scheduled_batch.switch_spec_forward_batch():
            logits_output = self.draft_model_runner.forward(scheduled_batch)

        spec_info = scheduled_batch.spec_info
        if not ALIGN_EAGLE_WITH_SGLANG_:
            pruned_states = spec_info.hidden_states[scheduled_batch.forward_batch.qo_indptr[1:] - 1]
        else:
            pruned_states = [hd[scheduled_batch.forward_batch.qo_indptr[1:] - 1] for hd in logits_output.aux_hidden_states]
            pruned_states = torch.concatenate(pruned_states, dim=-1)
        spec_info.hidden_states = pruned_states
        spec_info.logits = logits_output.next_token_logits
        return logits_output

    def multi_step_speculative_decode(self, scheduled_batch:ScheduleBatch):
        with torch.cuda.nvtx.range("draft_propose"):
            verify_input = self.draft_propose(scheduled_batch)
        with torch.cuda.nvtx.range("verify_propose"):
            return self.verify_propose(scheduled_batch, verify_input)

    def draft_decode_forward(self, scheduled_batch: ScheduleBatch):
        # prepare inputs for draft parallel decode, those parts seems compatible with cuda graph
        bs = len(scheduled_batch)
        spec_info = scheduled_batch.spec_info
        running_steps = self.speculative_num_steps - 1
        total_tokens = bs * self.topk * running_steps

        forward_batch_spec = scheduled_batch.forward_batch_spec
        past_kv_indices = forward_batch_spec.kv_indices_mtd

        out_cache_loc_tensor = self.prealloc_out_cache_loc_tensor[:total_tokens]
        scheduled_batch.position_ids = scheduled_batch.position_ids.repeat_interleave(self.topk, dim=0)
        seq_lens_sum = scheduled_batch.forward_batch.seq_lens_sum
        kv_out_buffers = (self.draft_decode_forward_runner.kv_indptr_buffer_s, 
                   self.draft_decode_forward_runner.kv_indices_buffer_s)
        kv_indices_outs = generate_kv_indices_for_mtd(kv_out_buffers,
            scheduled_batch.forward_batch.kv_indptr, past_kv_indices, out_cache_loc_tensor, 
            seq_lens_sum, bs, self.topk, running_steps)
        # cur_kv_seqlen = seq_lens[..., None] + 1
        for i in range(running_steps):
            forward_batch = self.attn_metadatas[i]
            forward_batch.kv_indptr = kv_indices_outs[0][i][:bs*self.topk+1]
            forward_batch.kv_indices = kv_indices_outs[1][i][:(seq_lens_sum + (i + 1) * bs) * self.topk]

        # Return values
        score_list: List[torch.Tensor] = []
        token_list: List[torch.Tensor] = []
        parents_list: List[torch.Tensor] = []
        out_cache_loc = out_cache_loc_tensor

        # Reshape out_cache_loc to (running_steps, bs * topk)
        out_cache_loc = out_cache_loc.reshape(bs, self.topk, running_steps)
        out_cache_loc = out_cache_loc.permute((2, 0, 1)).reshape(running_steps, -1)
        attn_metadatas = self.attn_metadatas

        # Start decoding
        probs = torch.softmax(spec_info.logits, dim=-1)
        topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
        hidden_states = spec_info.hidden_states
        if self.hot_token_id is not None:
            topk_index = self.hot_token_id[topk_index]
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
            # if not ALIGN_EAGLE_WITH_SGLANG_ and i > 0: #TODO this action should be done
            #     last_kv_indices = attn_metadatas[i-1].kv_indices.view(self.topk,-1)
            #     cur_kv_indices = attn_metadatas[i].kv_indices.view(self.topk,-1)
            #     cur_kv_indices[:, :-1] = last_kv_indices[selected_input_index]

            scheduled_batch.input_ids = input_ids
            scheduled_batch.forward_batch.out_cache_loc = out_cache_loc[i]
            scheduled_batch.position_ids.add_(1) # why this token align to the true position while the prefill not?
            spec_info.hidden_states = hidden_states
            scheduled_batch.forward_batch.spec_info = spec_info

            # Run forward
            logits_output = self.draft_model_runner.forward(scheduled_batch)
            probs = torch.softmax(logits_output.next_token_logits, dim=-1)
            topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
            if self.hot_token_id is not None:
                topk_index = self.hot_token_id[topk_index]
            hidden_states = torch.cat(logits_output.aux_hidden_states,dim=-1)

        parent_list, top_scores_index, draft_tokens = organize_draft_results(
            score_list, token_list, parents_list, self.speculative_num_draft_tokens
        )
        return parent_list, top_scores_index, draft_tokens

    def draft_propose(self, scheduled_batch:ScheduleBatch):
        self.pre_forward_last_verify_token(scheduled_batch)
        spec_info = scheduled_batch.spec_info
        spec_info.verified_id = scheduled_batch.input_ids# TODO,only works for bs=1

        #prepare kv cache loc
        orig_forward_batch = scheduled_batch.forward_batch
        seq_lens_sum = scheduled_batch.forward_batch.seq_lens_sum
        # bs = len(scheduled_batch)
        # running_steps = self.speculative_num_steps - 1
        # total_tokens = bs * self.topk * running_steps
        # token_allocator = self.draft_mem_pool
        # assert token_allocator.can_alloc(total_tokens)
        # device = seq_lens.device
        # out_cache_loc = token_allocator.borrow_disposable_block(total_tokens)
        # out_cache_loc_tensor = torch.tensor(out_cache_loc, dtype=torch.int32, 
        #                                     pin_memory=True).to(device, non_blocking=True)
        #input:spec_info.logits,spec_info.hidden_states,attn_metadatas
        # cuda graph runner of self.draft_decode_forward
        with torch.cuda.nvtx.range("draft_decode_forward_runner_forward"):
            parent_list, top_scores_index, draft_tokens = self.draft_decode_forward_runner.forward(
                scheduled_batch)

        seq_lens = orig_forward_batch.seq_lens
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
            seq_lens,
            seq_lens_sum,
            self.topk,
            self.speculative_num_steps,
            self.speculative_num_draft_tokens,
        )
        scheduled_batch.forward_batch = orig_forward_batch
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

    def pre_forward_last_verify_token(self, scheduled_batch:ScheduleBatch):
        #decode for the latest token#######################
        # the first time will be skipped as we have run it in prefill stage
        spec_info = scheduled_batch.spec_info
        if spec_info.hidden_states.shape[-1] == self.target_model_runner.get_config().hidden_size:
            return
        forward_batch_spec = scheduled_batch.forward_batch_spec
        old_input_ids = scheduled_batch.input_ids
        old_position_ids = scheduled_batch.position_ids

        forward_batch_spec.forward_mode = ForwardMode.DRAFT_EXTEND
        scheduled_batch.input_ids = spec_info.verified_id  # the real input_ids
        scheduled_batch.position_ids = scheduled_batch.forward_batch_spec.position_ids_extend # the real position_ids
        forward_batch_spec.spec_info = spec_info

        # Run forward
        with torch.cuda.nvtx.range("pre_forward_last_verify_token"):
            with scheduled_batch.switch_spec_forward_batch():
                logits_output = self.draft_model_runner.forward(scheduled_batch)

        spec_info.hidden_states = logits_output.aux_hidden_states[0][(forward_batch_spec.qo_indptr[1:]-1)]
        spec_info.logits = logits_output.next_token_logits
        scheduled_batch.position_ids = old_position_ids
        scheduled_batch.input_ids = old_input_ids


    def verify_propose(self, scheduled_batch:ScheduleBatch, verify_input:EagleVerifyInput):
        scheduled_batch.position_ids = verify_input.positions
        scheduled_batch.input_ids = verify_input.draft_token
        forward_batch = scheduled_batch.forward_batch

        forward_batch.forward_mode = ForwardMode.TARGET_VERIFY
        forward_batch.custom_mask = verify_input.custom_mask

        with torch.cuda.nvtx.range("target_model_runner_forward"):
            logits_output = self.target_model_runner.forward(scheduled_batch)
        forward_batch.forward_mode = ForwardMode.DECODE
        verify_input.hidden_states = torch.cat(logits_output.aux_hidden_states, dim=-1)

        accept_index, accept_length, predict = verify_input.verify(scheduled_batch, logits_output, 1)
        return verify_input, logits_output.next_token_logits, accept_index, accept_length, predict

    # why this work???
    def forward_decode_e2e(self, scheduled_batch:ScheduleBatch):
        #enmulate input tensors
        """
        scheduled_batch.forward_batch_spec as forward_batch:
        --->        forward_batch.kv_indptr = self.kv_indptr_buffer[: batch_size + 1]
                    forward_batch.kv_indices = self.kv_indices_buffer[: batch_size]
                    forward_batch.qo_indptr = self.qo_indptr_buffer[: batch_size + 1]
                    forward_batch.num_kv_splits = self.num_kv_splits_buffer[:batch_size]
                    forward_batch.forward_mode = ForwardMode.EXTEND
                    forward_batch.max_extend_len = self.server_args.speculative_num_steps+1
                    forward_batch.out_cache_loc = self.out_cache_loc[:token_nums]
                    forward_batch.position_ids_extend = self.position_ids_extend[:token_nums]
                kv_indices_mtd

            scheduled_batch.input_ids = input_ids
            scheduled_batch.position_ids = position_ids
        """
        #decode for the latest token#######################begin#######
        spec_info = scheduled_batch.spec_info
        forward_batch_spec = scheduled_batch.forward_batch_spec
        old_input_ids = scheduled_batch.input_ids
        old_position_ids = scheduled_batch.position_ids

        forward_batch_spec.forward_mode = ForwardMode.DRAFT_EXTEND
        scheduled_batch.input_ids = spec_info.verified_id
        scheduled_batch.position_ids = scheduled_batch.forward_batch_spec.position_ids_extend
        forward_batch_spec.spec_info = spec_info
        # Run forward
        with scheduled_batch.switch_spec_forward_batch():
            logits_output = self.draft_model_runner.forward(scheduled_batch)
        spec_info.hidden_states = logits_output.aux_hidden_states[0][(forward_batch_spec.qo_indptr[1:]-1)]
        spec_info.logits = logits_output.next_token_logits
        scheduled_batch.position_ids = old_position_ids
        scheduled_batch.input_ids = old_input_ids

        #########end###########################################

        ##### draft propose begin ##################
        spec_info = scheduled_batch.spec_info
        # spec_info.verified_id = scheduled_batch.input_ids #TODO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        spec_info.verified_id = spec_info.verified_id[scheduled_batch.forward_batch_spec.qo_indptr[1:]-1]

        #prepare kv cache loc
        orig_forward_batch = scheduled_batch.forward_batch
        seq_lens_sum = scheduled_batch.forward_batch.seq_lens_sum
        bs = len(scheduled_batch)
        running_steps = self.speculative_num_steps - 1
        total_tokens = bs * self.topk * running_steps

        forward_batch_spec = scheduled_batch.forward_batch_spec
        past_kv_indices = forward_batch_spec.kv_indices_mtd

        out_cache_loc_tensor = self.prealloc_out_cache_loc_tensor[:total_tokens]
        scheduled_batch.position_ids = scheduled_batch.position_ids.repeat_interleave(self.topk, dim=0)
        seq_lens_sum = scheduled_batch.forward_batch.seq_lens_sum
        kv_out_buffers = (self.draft_decode_forward_runner.kv_indptr_buffer_s, 
                   self.draft_decode_forward_runner.kv_indices_buffer_s)
        kv_indices_outs = generate_kv_indices_for_mtd(kv_out_buffers,
            scheduled_batch.forward_batch.kv_indptr, past_kv_indices, out_cache_loc_tensor, 
            seq_lens_sum, bs, self.topk, running_steps)
        # cur_kv_seqlen = seq_lens[..., None] + 1
        for i in range(running_steps):
            forward_batch = self.attn_metadatas[i]
            forward_batch.kv_indptr = kv_indices_outs[0][i][:bs*self.topk+1]
            forward_batch.kv_indices = kv_indices_outs[1][i][:(seq_lens_sum + (i + 1) * bs) * self.topk]

        # Return values
        score_list: List[torch.Tensor] = []
        token_list: List[torch.Tensor] = []
        parents_list: List[torch.Tensor] = []
        out_cache_loc = out_cache_loc_tensor

        # Reshape out_cache_loc to (running_steps, bs * topk)
        out_cache_loc = out_cache_loc.reshape(bs, self.topk, running_steps)
        out_cache_loc = out_cache_loc.permute((2, 0, 1)).reshape(running_steps, -1)
        attn_metadatas = self.attn_metadatas

        # Start decoding
        probs = torch.softmax(spec_info.logits, dim=-1)
        topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
        hidden_states = spec_info.hidden_states
        if self.hot_token_id is not None:
            topk_index = self.hot_token_id[topk_index]
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
            # if not ALIGN_EAGLE_WITH_SGLANG_ and i > 0: #TODO this action should be done
            #     last_kv_indices = attn_metadatas[i-1].kv_indices.view(self.topk,-1)
            #     cur_kv_indices = attn_metadatas[i].kv_indices.view(self.topk,-1)
            #     cur_kv_indices[:, :-1] = last_kv_indices[selected_input_index]

            scheduled_batch.input_ids = input_ids
            scheduled_batch.forward_batch.out_cache_loc = out_cache_loc[i]
            scheduled_batch.position_ids.add_(1) # why this token align to the true position while the prefill not?
            spec_info.hidden_states = hidden_states
            scheduled_batch.forward_batch.spec_info = spec_info

            # Run forward
            logits_output = self.draft_model_runner.forward(scheduled_batch)
            probs = torch.softmax(logits_output.next_token_logits, dim=-1)
            topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
            if self.hot_token_id is not None:
                topk_index = self.hot_token_id[topk_index]
            hidden_states = torch.cat(logits_output.aux_hidden_states,dim=-1)

        parent_list, top_scores_index, draft_tokens = organize_draft_results(
            score_list, token_list, parents_list, self.speculative_num_draft_tokens
        )
        seq_lens = orig_forward_batch.seq_lens
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
            seq_lens,
            seq_lens_sum,
            self.topk,
            self.speculative_num_steps,
            self.speculative_num_draft_tokens,
            tree_mask_buf=self.target_model_runner.custom_mask_buffer
        )
        scheduled_batch.forward_batch = orig_forward_batch
        verify_input = EagleVerifyInput(
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
        ##### draft propose end ##################

        scheduled_batch.position_ids = verify_input.positions
        scheduled_batch.input_ids = verify_input.draft_token
        forward_batch = scheduled_batch.forward_batch

        forward_batch.forward_mode = ForwardMode.TARGET_VERIFY
        forward_batch.custom_mask = verify_input.custom_mask

        logits_output = self.target_model_runner.forward(scheduled_batch)
        forward_batch.forward_mode = ForwardMode.DECODE

        verify_input.hidden_states = torch.cat(logits_output.aux_hidden_states, dim=-1)

        accept_index, accept_length, predict = verify_input.verify(scheduled_batch, logits_output, 1)
        return verify_input, logits_output.next_token_logits, accept_index, accept_length, predict

        ##===========post cuda graph================
    def forward_decode_e2e_post_process(self, scheduled_batch:ScheduleBatch, verify_input:EagleVerifyInput, 
                                        next_token_logits:torch.Tensor, accept_index:torch.Tensor, 
                                        accept_length:torch.Tensor, predict:torch.Tensor):
        logits_output = BatchResult(None, next_token_logits, None)
        ret = verify_input.verify_post_process(scheduled_batch, accept_index, accept_length, predict)
        logits_output.next_token_ids = ret.verified_id # padded for async
        # logits_output.next_token_logits = logits_output.next_token_logits[ret.accepted_indices] #same
        spec_info = scheduled_batch.spec_info
        spec_info.verified_id = ret.verified_id
        # spec_info.logits = logits_output.next_token_logits
        spec_info.hidden_states = verify_input.hidden_states[ret.accepted_indices]
        spec_info.accept_length = ret.accept_length
        # spec_info.accept_index = ret.accepted_indices
        logits_output.spec_info = spec_info
        return logits_output

    def spec_postprocess(self, scheduled_batch:ScheduleBatch, batch_output:BatchResult, async_overlap:bool=False):
        draft_token_num = self.server_args.speculative_num_draft_tokens
        last_verify_id_start = 0
        
        if async_overlap:
            spec_info = batch_output.spec_info
            accept_length_cpu = spec_info.accept_length_cpu
            out_cache_loc_cpu = batch_output.out_cache_loc
        else:
            spec_info = scheduled_batch.spec_info
            accept_length_cpu = spec_info.accept_length.cpu()
            spec_info.accept_length_cpu = accept_length_cpu
            accept_length_cpu = accept_length_cpu.clamp(min=0)
            out_cache_loc_cpu = scheduled_batch.forward_batch.out_cache_loc.cpu()
            batch_output.next_token_ids = batch_output.next_token_ids[:(1+accept_length_cpu).sum()]
            spec_info.verified_id = spec_info.verified_id[:(1+accept_length_cpu).sum()]
            spec_info.hidden_states = spec_info.hidden_states[:(1+accept_length_cpu).sum()]

        # the prefill batch won't have accept_index
        # hanlde out_cache_loc_lazy
        if spec_info.out_cache_loc is not None:
            # spec_info.accept_index = spec_info.accept_index[spec_info.accept_index!=-1]
            # accept_index = spec_info.accept_index.cpu()
            # evict_mask = torch.full((len(scheduled_batch)*draft_token_num,), True, dtype=torch.bool)
            # evict_mask[accept_index] = False
            accept_out_cache_loc = spec_info.out_cache_loc.cpu() # in overlap mode,spec_info.out_cache_loc is in CPU already
            accept_cache_loc_list = accept_out_cache_loc[accept_out_cache_loc != -1].tolist()
            refused_cache_loc = list(set(out_cache_loc_cpu.tolist())-set(accept_cache_loc_list))
            assert len(refused_cache_loc)+len(accept_cache_loc_list) == len(out_cache_loc_cpu)
            self.main_mem_pool.free_block(refused_cache_loc)
            if not async_overlap:
                for idx, sequence in enumerate(scheduled_batch):
                    ############### adjust out_cache_loc ##################
                    # the final token is un-determined before the forward, 
                    # we have to alloc num-draft-token cache loc per batch. After, we need to adjust it here
                    sequence.out_cache_loc = sequence.out_cache_loc[: -draft_token_num]
                    accept_len = accept_length_cpu[idx].item()+1
                    sequence.out_cache_loc.extend(accept_cache_loc_list[: accept_len])
                    accept_cache_loc_list = accept_cache_loc_list[accept_len:]
                    # for async overlap ????
                    # sequence.accept_index = spec_info.accept_index[last_verify_id_start:end]
            else:
                num_steps = self.speculative_num_steps
                for idx, sequence in enumerate(scheduled_batch):
                    accept_len = accept_length_cpu[idx].item()
                    start = draft_token_num + num_steps
                    # -1 means the root token, we have kept it in the last step
                    sequence.out_cache_loc[-start: -start + accept_len] = accept_cache_loc_list[1: 1+accept_len]
                    sequence.out_cache_loc[-start + accept_len:-start + num_steps] = []
                    accept_cache_loc_list = accept_cache_loc_list[1+accept_len:]

                ######################################################
        # but we will alloc draft_steps+1 tokens cache loc per sequence, so also need to free the unused ones
        if async_overlap:
            num_steps = self.speculative_num_steps
            num_draft_tokens = self.server_args.speculative_num_draft_tokens
            for idx, seq in enumerate(scheduled_batch):
                ac_len = batch_output.spec_info.accept_length_cpu[idx].item()
                offset = len(seq.out_cache_loc_spec) - (num_steps-ac_len)
                seq.out_cache_loc_spec, extral_loc = seq.out_cache_loc_spec[
                    :offset], seq.out_cache_loc_spec[offset:]
                self.draft_mem_pool.free_block(extral_loc, force_sort=True)
                # we can't release the verify loc here, but we can set it as neg to indicate unknown yet
                # the root token is alway accepted, so we can keep one less
                offset = len(seq.out_cache_loc) - (num_draft_tokens-1)
                seq.out_cache_loc, seq.out_cache_loc_lazy_cpu = seq.out_cache_loc[
                    :offset], seq.out_cache_loc[offset:]
                # placeholder for the final accepted token loc
                seq.out_cache_loc.extend([-1] * (num_steps))#TODO +1 or not?
                seq.marked = True

        if not async_overlap:
            for idx, sequence in enumerate(scheduled_batch):
                if sequence.status.is_active():
                    sequence.accept_length = spec_info.accept_length[idx:idx+1]
                    sequence.accept_length_cpu = spec_info.accept_length.cpu()[idx:idx+1]
                    accept_length = accept_length_cpu[idx].item()
                    end = last_verify_id_start+accept_length+1
                    sequence.verified_id = spec_info.verified_id[last_verify_id_start:end]
                    # sequence.logits = spec_info.logits[last_verify_id_start:end] # keep batch dim
                    sequence.hidden_states = spec_info.hidden_states[last_verify_id_start:end]
                    last_verify_id_start = end
        if self.server_args.enable_debug:
            acn = accept_length_cpu.clamp(min=0).sum().item()
            self.total_accepted_tokens += acn
            logger.info(f"Speculative decoding: accepted {(acn)} tokens, total accepted {self.total_accepted_tokens}.")
        return scheduled_batch