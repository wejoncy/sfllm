from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch
import logging
import math
import triton
import triton.language as tl

import sf_kernel
from sfllm.spec_decoding.spec_common import SpecInput, SpecInputType

logger = logging.getLogger(__name__)


@dataclass
class EagleSpecInput(SpecInput):
    # The inputs for decode
    # shape: (b, topk)
    topk_p: torch.Tensor = None
    topk_index: torch.Tensor = None
    # shape: (b, hidden_size)
    hidden_states: torch.Tensor = None

    # Inputs for extend
    # shape: (b,)
    verified_id: torch.Tensor = None
    accept_length: torch.Tensor = None
    accept_length_cpu: List[int] = None

    # Inputs for the attention backends
    # shape: (b + 1,)
    kv_indptr: torch.Tensor = None
    kv_indices: torch.Tensor = None

    logits: torch.Tensor = None

    def raw_new(self):
        return EagleSpecInput()


@dataclass
class EagleVerifyOutput:
    # Draft input batch
    draft_input: EagleSpecInput
    # Logit outputs from target worker
    logits_output: torch.Tensor
    # Accepted token ids including the bonus token
    verified_id: torch.Tensor
    # Accepted token length per sequence in a batch in CPU.
    accept_length_per_req_cpu: List[int]
    # Accepted indices from logits_output.next_token_logits
    accepted_indices: torch.Tensor


def get_cuda_stream() -> int:
    return torch.cuda.current_stream().cuda_stream


@dataclass
class EagleVerifyInput(SpecInput):
    draft_token: torch.Tensor
    custom_mask: torch.Tensor
    positions: torch.Tensor
    retrive_index: torch.Tensor
    retrive_next_token: torch.Tensor
    retrive_next_sibling: torch.Tensor
    retrive_cum_len: torch.Tensor
    spec_steps: int
    topk: int
    draft_token_num: int

    def verify(
        self,
        batch,
        logits_output,
        page_size: int,
        vocab_mask: Optional[torch.Tensor] = None,  # For grammar
    ) -> torch.Tensor:
        """
        Verify and find accepted tokens based on logits output and batch
        (which contains spec decoding information).

        WARNING: This API in-place modifies the states of logits_output

        This API updates values inside logits_output based on the accepted
        tokens. I.e., logits_output.next_token_logits only contains
        accepted token logits.
        """
        

        bs = len(batch)
        candidates = self.draft_token.reshape(bs, self.draft_token_num)
        # sampling_info = batch.sampling_info

        predict_shape = list(logits_output.next_token_logits.shape)[:-1]
        predict_shape[-1] += 1
        predict = torch.empty(predict_shape, dtype=torch.int32, device="cuda")
        accept_index = torch.full(
            (bs, self.spec_steps + 1), -1, dtype=torch.int32, device="cuda"
        )
        accept_length = torch.empty((bs,), dtype=torch.int32, device="cuda")

        # Sample tokens. Force greedy sampling on AMD
        is_all_greedy = True#sampling_info.is_all_greedy
        if is_all_greedy:
            target_predict = torch.argmax(logits_output.next_token_logits, dim=-1)
            target_predict = target_predict.reshape(bs, self.draft_token_num)

            sf_kernel.verify_tree_greedy(
                predicts=predict,  # mutable
                accept_index=accept_index,  # mutable
                accept_token_num=accept_length,  # mutable
                candidates=candidates,
                retrive_index=self.retrive_index,
                retrive_next_token=self.retrive_next_token,
                retrive_next_sibling=self.retrive_next_sibling,
                target_predict=target_predict,
                cuda_stream=get_cuda_stream(),
            )
        else:
            # apply temperature and get target probs
            expanded_temperature = torch.repeat_interleave(
                sampling_info.temperatures, self.draft_token_num, dim=0
            )  # (bs * draft_token_num, 1)

            target_probs = F.softmax(
                logits_output.next_token_logits / expanded_temperature, dim=-1
            )  # (bs * draft_token_num, vocab_size)
            target_probs = top_k_renorm_prob(
                target_probs,
                torch.repeat_interleave(
                    sampling_info.top_ks, self.draft_token_num, dim=0
                ),
            )  # (bs * draft_token_num, vocab_size)
            if not torch.all(sampling_info.top_ps == 1.0):
                target_probs = top_p_renorm_prob(
                    target_probs,
                    torch.repeat_interleave(
                        sampling_info.top_ps, self.draft_token_num, dim=0
                    ),
                )
            target_probs = target_probs.reshape(bs, self.draft_token_num, -1)

            draft_probs = torch.zeros(
                target_probs.shape, dtype=torch.float32, device="cuda"
            )

            # coins for rejection sampling
            coins = torch.rand_like(candidates, dtype=torch.float32, device="cuda")
            # coins for final sampling
            coins_for_final_sampling = torch.rand(
                (bs,), dtype=torch.float32, device="cuda"
            )
            tree_speculative_sampling_target_only(
                predicts=predict,  # mutable
                accept_index=accept_index,  # mutable
                accept_token_num=accept_length,  # mutable
                candidates=candidates,
                retrive_index=self.retrive_index,
                retrive_next_token=self.retrive_next_token,
                retrive_next_sibling=self.retrive_next_sibling,
                uniform_samples=coins,
                uniform_samples_for_final_sampling=coins_for_final_sampling,
                target_probs=target_probs,
                draft_probs=draft_probs,
                threshold_single=get_global_server_args().speculative_accept_threshold_single,
                threshold_acc=get_global_server_args().speculative_accept_threshold_acc,
                deterministic=True,
            )
        # if SIMULATE_ACC_LEN > 0.0:
        #     # Do simulation
        #     accept_index = generate_simulated_accept_index(
        #         accept_index=accept_index,
        #         predict=predict,  # mutable
        #         accept_length=accept_length,  # mutable
        #         bs=bs,
        #         spec_steps=self.spec_steps,
        #     )
        return accept_index, accept_length, predict

    def verify_post_process(
        self,
        batch,
        accept_index: torch.Tensor,
        accept_length: torch.Tensor,
        predict: torch.Tensor,
        logits_output: torch.Tensor,
        token_to_kv_pool_allocator,
        page_size: int = 1,
    ):
        unfinished_index = []
        unfinished_accept_index = []
        accept_index_cpu = accept_index.tolist()
        predict_cpu = predict.tolist()
        has_finished = False

        # Iterate every accepted token and check if req has finished after append the token
        # should be checked BEFORE free kv cache slots
        for i, (req, accept_index_row) in enumerate(zip(batch.sequences, accept_index_cpu)):
            req.new_tokens = []
            for j, idx in enumerate(accept_index_row):
                if idx == -1:
                    break
                id = predict_cpu[idx]
                req.new_tokens.append(id)
                # req.tokens.append(id)
                req.is_done()
                if req.is_done():
                    has_finished = True
                    # set all tokens after finished token to -1 and break
                    accept_index[i, j + 1 :] = -1
                    break

            if not req.is_done():
                unfinished_index.append(i)
                if idx == -1:
                    unfinished_accept_index.append(accept_index[i, :j])
                else:
                    unfinished_accept_index.append(accept_index[i])
            # req.spec_verify_ct += 1
            # req.spec_accepted_tokens += (
            #     sum(1 for idx in accept_index_row if idx != -1) - 1
            # )

        if has_finished:
            accept_length = (accept_index != -1).sum(dim=1) - 1

        # Free the KV cache for unaccepted tokens
        # TODO: fuse them
        accept_index = accept_index[accept_index != -1]
        verified_id = predict[accept_index]
        evict_mask = torch.full_like(self.draft_token, True, dtype=torch.bool)
        evict_mask[accept_index] = False
        accept_length_cpu = accept_length.cpu()
        # FIXME: this `tolist()` fixes the numerical calculation consistency
        # try to unify the tensor representation and list representation
        accept_length_list = accept_length_cpu.tolist()

        if page_size == 1:
            # TODO: boolean array index leads to a device sync. Remove it.
            accept_cache_loc = batch.forward_batch.out_cache_loc[~evict_mask].tolist()
            for locidx, seq_bt in enumerate(batch):
                seq_bt.out_cache_loc = seq_bt.out_cache_loc[: -self.draft_token_num]
                accept_len = accept_length_list[locidx]
                seq_bt.out_cache_loc.extend(accept_cache_loc[: accept_len + 1])
                accept_cache_loc = accept_cache_loc[accept_len + 1 :]
            token_to_kv_pool_allocator.free_block(
                batch.forward_batch.out_cache_loc[evict_mask].tolist()
            )
        

        # Construct EagleVerifyOutput
        if not has_finished:
            if page_size == 1 or self.topk == 1:
                batch.forward_batch.out_cache_loc = batch.forward_batch.out_cache_loc[
                    accept_index
                ]
            #     assign_req_to_token_pool[(bs,)](
            #         batch.req_pool_indices,
            #         batch.req_to_token_pool.req_to_token,
            #         batch.seq_lens,
            #         batch.seq_lens + accept_length + 1,
            #         batch.out_cache_loc,
            #         batch.req_to_token_pool.req_to_token.shape[1],
            #         next_power_of_2(bs),
            #     )
            # else:
            #     batch.out_cache_loc = tgt_cache_loc
            # batch.seq_lens.add_(accept_length + 1)
            # batch.seq_lens_cpu.add_(accept_length_cpu + 1)

            draft_input = EagleSpecInput(
                hidden_states=self.hidden_states[accept_index],
                verified_id=verified_id,
                accept_length=accept_length,
                accept_length_cpu=accept_length_cpu,
                # seq_lens_for_draft_extend=batch.forward_batch.seq_lens,
                # seq_lens_for_draft_extend_cpu=batch.seq_lens_cpu,
                # req_pool_indices_for_draft_extend=batch.req_pool_indices,
            )

            return EagleVerifyOutput(
                draft_input=draft_input,
                logits_output=logits_output,
                verified_id=verified_id,
                accept_length_per_req_cpu=draft_input.accept_length_cpu,
                accepted_indices=accept_index,
            )


def fast_topk(values, topk, dim):
    if topk == 1:
        # Use max along the specified dimension to get both value and index
        return torch.max(values, dim=dim, keepdim=True)
    else:
        # Use topk for efficiency with larger k values
        # TODO: implement faster cuda kernels for large vocab sizes
        return torch.topk(values, topk, dim=dim)

def select_top_k_tokens(
    i: int,
    topk_p: torch.Tensor,
    topk_index: torch.Tensor,
    hidden_states: torch.Tensor,
    scores: torch.Tensor,
    topk: int,
):
    selected_input_index = torch.arange(topk_p.shape[-1], device="cuda")
    if i == 0:
        # The first step after extend
        input_ids = topk_index.flatten()
        hidden_states = hidden_states.repeat_interleave(topk, dim=0)
        scores = topk_p  # shape: (b, topk)

        tree_info = (
            topk_p.unsqueeze(1),  # shape: (b, 1, topk)
            topk_index,  # shape: (b, topk)
            torch.arange(-1, topk, dtype=torch.long, device=hidden_states.device)
            .unsqueeze(0)
            .repeat(topk_p.shape[0], 1),  # shape: (b, topk + 1)
        )
    else:
        # The later decode steps
        expand_scores = torch.mul(
            scores.unsqueeze(2), topk_p.reshape(-1, topk, topk)
        )  # (b, topk, 1) x (b, topk ,topk) -> (b, topk, topk)
        topk_cs_p, topk_cs_index = fast_topk(
            expand_scores.flatten(start_dim=1), topk, dim=-1
        )  # (b, topk)
        scores = topk_cs_p  # shape: (b, topk)

        topk_index = topk_index.reshape(-1, topk**2)
        input_ids = torch.gather(topk_index, index=topk_cs_index, dim=1).flatten()

        if hidden_states.shape[0] > 0:
            selected_input_index = topk_cs_index.flatten() // topk + torch.arange(
                0, hidden_states.shape[0], step=topk, device="cuda"
            ).repeat_interleave(topk)
            hidden_states = hidden_states[selected_input_index, :]

        tree_info = (
            expand_scores,  # shape: (b, topk, topk)
            topk_index,  # shape: (b, topk * topk)
            topk_cs_index + (topk**2 * (i - 1) + topk),  # shape: (b, topk)
        )

    return input_ids, hidden_states, scores, tree_info, selected_input_index

def organize_draft_results(
    score_list: List[torch.Tensor],
    token_list: List[torch.Tensor],
    parents_list: List[torch.Tensor],
    num_draft_token: int,
):
    score_list = torch.cat(score_list, dim=1).flatten(1)
    ss_token_list = torch.cat(token_list, dim=1)
    top_scores = torch.topk(score_list, num_draft_token - 1, dim=-1)
    top_scores_index = top_scores.indices
    top_scores_index = torch.sort(top_scores_index).values
    draft_tokens = torch.gather(ss_token_list, index=top_scores_index, dim=1)

    if len(parents_list) > 1:
        parent_list = torch.cat(parents_list[:-1], dim=1)
    else:
        batch_size = parents_list[0].shape[0]
        parent_list = torch.empty(batch_size, 0, device=parents_list[0].device)

    return parent_list, top_scores_index, draft_tokens

def build_tree_efficient_native(
    parent_list: torch.Tensor,
    selected_index: torch.Tensor,
    verified_seq_len: torch.Tensor,
    tree_mask: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    topk: int,
    draft_token_num: int,
    tree_mask_mode: int,
    bs: int,
):
    # Generate batch and token index ranges
    bs_range = torch.arange(bs, device=tree_mask.device).view(-1, 1)
    draft_token_num_range = torch.arange(draft_token_num, device=tree_mask.device)

    # Optimized common case for performance.
    if draft_token_num == 2 and topk == 1 and tree_mask_mode == 0:
        positions = verified_seq_len.repeat_interleave(draft_token_num)
        positions = (positions.view(bs, -1) + draft_token_num_range).view(-1)

        retrive_index[:] = bs_range * draft_token_num + draft_token_num_range
        retrive_next_token[:, 0] = 1
        retrive_next_token[:, 1] = -1
        return (
            positions,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            tree_mask,
        )

    # Precompute sequence tree indices
    draft_token_num_range1 = torch.arange(draft_token_num - 1, device=tree_mask.device)
    cum_seq_len = torch.cumsum(verified_seq_len * draft_token_num, dim=0)
    cum_seq_len = torch.cat((torch.tensor([0], device=tree_mask.device), cum_seq_len))
    cum_seq_len = cum_seq_len[:-1]
    seq_tree_idx = (
        draft_token_num * draft_token_num * torch.arange(bs, device=tree_mask.device)
        + cum_seq_len
    )

    # Batch processing for tree mask
    if tree_mask_mode == 0:
        token_tree_base = (
            seq_tree_idx.view(-1, 1)
            + (verified_seq_len.view(-1, 1) + draft_token_num) * draft_token_num_range
        )
        token_tree_indices = token_tree_base + verified_seq_len.view(-1, 1) + 1
    else:
        token_tree_indices = (
            bs_range * draft_token_num**2 + draft_token_num_range * draft_token_num + 1
        )

    tree_mask[token_tree_indices.flatten() - 1] = True
    indices = token_tree_indices.unsqueeze(-1) + draft_token_num_range1.view(1, 1, -1)
    tree_mask[indices.view(-1)] = False

    positions = verified_seq_len.repeat_interleave(draft_token_num)
    parent_tb_indices = selected_index // topk
    retrive_index[:] = bs_range * draft_token_num + draft_token_num_range
    tree_mask[token_tree_indices.view(-1, 1) + draft_token_num_range1] = True

    for bid in range(bs):
        for tid in range(draft_token_num):
            position = 0
            if tid == 0:
                # Process root node
                for i in range(draft_token_num - 1, 0, -1):
                    parent_position = 0
                    parent_tb_idx = parent_tb_indices[bid][i - 1]
                    if parent_tb_idx > 0:
                        parent_token_idx = parent_list[bid][parent_tb_idx]
                        loop_num = draft_token_num - parent_position
                        for _ in range(loop_num):
                            if selected_index[bid][parent_position] == parent_token_idx:
                                parent_position += 1
                                break
                            parent_position += 1
                    if parent_position == draft_token_num:
                        continue

                    if retrive_next_token[bid][parent_position] != -1:
                        retrive_next_sibling[bid][i] = retrive_next_token[bid][
                            parent_position
                        ]
                    retrive_next_token[bid][parent_position] = i
            else:
                # Process no-root nodes
                cur_position = tid - 1
                while True:
                    position += 1
                    if cur_position >= draft_token_num:
                        tree_mask[token_tree_indices + cur_position] = True
                        parent_tb_idx = selected_index[bid][cur_position] // topk
                    else:
                        parent_tb_idx = parent_tb_indices[bid][cur_position]
                    if parent_tb_idx == 0:
                        break
                    token_idx = parent_list[bid][parent_tb_idx]
                    cur_position = 0
                    for _ in range(draft_token_num):
                        if selected_index[bid][cur_position] == token_idx:
                            break
                        cur_position += 1
                positions[bid * draft_token_num + tid] += position
    return positions, retrive_index, retrive_next_token, retrive_next_sibling, tree_mask


def build_tree_kernel_efficient(
    verified_id: torch.Tensor,
    parent_list: List[torch.Tensor],
    top_scores_index: torch.Tensor,
    draft_tokens: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_lens_sum: int,
    topk: int,
    spec_steps: int,
    num_verify_tokens: int,
    tree_mask_buf: Optional[torch.Tensor] = None,
    position_buf: Optional[torch.Tensor] = None,
):
    draft_tokens = torch.cat((verified_id.unsqueeze(1), draft_tokens), dim=1).flatten()

    # seq_lens_sum == sum(seq_lens); seq_lens: sequence length without draft tokens
    bs = seq_lens.numel()
    device = seq_lens.device
    # e.g. for bs=1, tree_mask: num_draft_token, seq_lens_sum + num_draft_token (flattened)
    # where each row indicates the attending pattern of each draft token
    # if use_partial_packed_tree_mask is True, tree_mask: num_draft_token (flattened, packed)

    if tree_mask_buf is not None:
        tree_mask = tree_mask_buf
    else:
        tree_mask = torch.full(
            (
                seq_lens_sum * num_verify_tokens
                + num_verify_tokens * num_verify_tokens * bs,
            ),
            True,
            device=device,
        )
    # TODO: make them torch.empty and fuse them into `sgl_build_tree_kernel`
    retrive_buf = torch.full(
        (3, bs, num_verify_tokens), -1, device=device, dtype=torch.long
    )
    retrive_index, retrive_next_token, retrive_next_sibling = retrive_buf
    positions = torch.empty(
        (bs * num_verify_tokens,), device=device, dtype=torch.long
    )

    tree_mask_mode = 0
    sf_kernel.build_tree_kernel_efficient(
        parent_list,
        top_scores_index,
        seq_lens.long(),
        tree_mask,
        positions,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        topk,
        spec_steps,
        num_verify_tokens,
        tree_mask_mode,
    )
    return (
        tree_mask,
        positions,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        draft_tokens,
    )


@triton.jit
def generate_kv_indices_kernel(
    old_kv_indptr,          # [bs + 1], cumulative sequence lengths
    past_kv_indices,        # [seq_lens_sum], past KV cache indices
    out_cache_loc_tensor,   # [bs * topk, running_steps], new cache locations
    kv_indptr,              # Output: [running_steps * (topk * bs + 1)], cumulative indices pointers
    kv_indices,             # Output: flattened KV indices array
    running_steps: int,
    bs: int,
    kv_indices_stride_0: tl.constexpr,
    kv_indptr_stride_0: tl.constexpr,
    topk: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    num_warps: tl.constexpr,
):
    """
    Efficiently generate KV indices for multiple time steps using Triton.
    
    Optimized for smaller block sizes (128/256) with better parallelization:
    - Uses smaller BLOCK_SIZE for better occupancy
    - Supports multiple warps for long sequences
    - Each thread block processes one (step, batch, topk) combination efficiently
    
    Uses the PyTorch offset formula:
    offset_start = (seq_lens_sum*i + sum(range(1,i+1)) * bs) * topk
    """
    # Get current grid position - each thread handles one (step, batch, topk) combination
    step_id = tl.program_id(0)      # Current time step [0, running_steps)
    batch_id = tl.program_id(1)     # Current batch [0, bs)
    topk_id = tl.program_id(2)      # Current topk variant [0, topk)

    seq_lens_sum = tl.load(old_kv_indptr+bs)  # Total original sequence lengths sum
    
    # Load sequence boundaries for current batch
    seq_start = tl.load(old_kv_indptr + batch_id)
    seq_end = tl.load(old_kv_indptr + batch_id + 1)
    original_seq_len = seq_end - seq_start
    
    # Current sequence length includes original sequence + new tokens up to current step
    current_seq_len = original_seq_len + step_id + 1
    
    # === Calculate kv_indptr position and cumulative length ===
    # Each step has (topk * bs + 1) entries in indptr, first entry is always 0
    if kv_indptr_stride_0 == 0:
        indptr_base_offset = step_id * (topk * bs + 1)
    else:
        indptr_base_offset = step_id * kv_indptr_stride_0
    indptr_idx = indptr_base_offset + batch_id * topk + topk_id + 1
    
    # Calculate cumulative length up to current position using optimized formula
    # Original: old_kv_indptr[batch_id] is cumsum of original seq_lens up to batch_id
    # After adding (step_id + 1) tokens to each sequence, the new cumsum becomes:
    # new_cumsum[batch_id] = old_kv_indptr[batch_id] + batch_id * (step_id + 1)
    prev_batches_total_len = tl.load(old_kv_indptr + batch_id) + batch_id * (step_id + 1)
    cumulative_len = prev_batches_total_len * topk
    
    # Add lengths from previous topk variants in current batch
    cumulative_len += topk_id * current_seq_len
    
    # Add current sequence length to get the cumulative end position
    cumulative_len += current_seq_len
    
    # Store cumulative length in indptr
    tl.store(kv_indptr + indptr_idx, cumulative_len)
    
    # === Calculate kv_indices starting position using PyTorch formula ===
    # Base offset from PyTorch formula: offset_start = (seq_lens_sum*i + sum(range(1,i+1)) * bs) * topk
    if kv_indices_stride_0 == 0:
        indices_offset = seq_lens_sum * step_id * topk
        
        # Add sum(range(1,step_id+1)) * bs * topk = step_id*(step_id+1)/2 * bs * topk
        step_sum = step_id * (step_id + 1) // 2
        indices_offset += step_sum * bs * topk
    else:
        indices_offset = step_id * kv_indices_stride_0
    
    # Add offset within current step for previous batches
    # Use cumulative sum from old_kv_indptr: sum of (seq_len[j] + step_id + 1) for j < batch_id
    prev_batches_seq_sum = tl.load(old_kv_indptr + batch_id)  # Sum of seq lengths before current batch
    prev_batches_total_len = prev_batches_seq_sum + batch_id * (step_id + 1)
    indices_offset += prev_batches_total_len * topk
    
    # Add offset for previous topk variants in current batch
    indices_offset += topk_id * current_seq_len
    
    # === Fill KV indices array with optimized parallelization ===
    # Process original sequence indices in chunks for better parallelization
    for seq_chunk_start in range(0, original_seq_len, BLOCK_SIZE):
        seq_offsets = tl.arange(0, BLOCK_SIZE)
        seq_chunk_offsets = seq_chunk_start + seq_offsets
        seq_mask = seq_chunk_offsets < original_seq_len
        
        # Vectorized load from past_kv_indices
        past_indices = tl.load(past_kv_indices + seq_start + seq_chunk_offsets, mask=seq_mask, other=0)
        
        # Vectorized store to kv_indices
        tl.store(kv_indices + indices_offset + seq_chunk_offsets, past_indices, mask=seq_mask)
    
    # Fill new cache locations for steps 0 through current step
    # This is typically small (step_id + 1 <= 8), so one vectorized operation is sufficient
    cache_tensor_row = batch_id * topk + topk_id
    cache_offsets = tl.arange(0, BLOCK_SIZE)
    cache_mask = cache_offsets <= step_id  # 0 to step_id inclusive
    
    # Vectorized load from out_cache_loc_tensor  
    cache_locs = tl.load(out_cache_loc_tensor + cache_tensor_row * running_steps + cache_offsets,
                        mask=cache_mask, other=0)
    
    # Vectorized store to kv_indices at offset after original sequence
    tl.store(kv_indices + indices_offset + original_seq_len + cache_offsets,
            cache_locs, mask=cache_mask)


def generate_kv_indices_for_mtd_triton(kv_out_buffer: Tuple[torch.Tensor, torch.Tensor], old_kv_indptr: torch.Tensor, past_kv_indices: torch.Tensor,
                                       out_cache_loc_tensor: torch.Tensor,
                                       seq_lens_sum: int,
                                       running_steps: int, topk: int):
    """
    Triton-accelerated version of generate_kv_indices_for_mtd.
    Returns the same format as the PyTorch implementation.
    
    Uses default 1024 block size with 8 warps - simple and effective.
    """
    device = past_kv_indices.device
    bs = old_kv_indptr.numel() - 1

    # Use default configuration - simple and performs well for typical LLM workloads
    block_size = 1024
    num_warps = 8

    kv_indices_stride_0 = 0
    kv_indptr_stride_0 = 0
    if kv_out_buffer is not None:
        kv_indptr, kv_indices = kv_out_buffer
        assert kv_indices.shape[0] == running_steps and kv_indices.is_contiguous()
        assert kv_indptr.shape[0] == running_steps and kv_indptr.is_contiguous()
        kv_indices_stride_0 = kv_indices.stride(0)
        kv_indptr_stride_0 = kv_indptr.stride(0)
        assert (seq_lens_sum + (running_steps - 1 + 1) * bs) * topk < kv_indices_stride_0
    else:
        kv_indptr = torch.zeros((running_steps*(topk*bs+1),), dtype=torch.int32, device=device)
        kv_indices = torch.empty(
            ((seq_lens_sum*running_steps + sum(range(1,running_steps+1))*bs) * topk,),
            dtype=past_kv_indices.dtype,
            device=device,
        )
    if torch.cuda.is_current_stream_capturing():
        assert kv_out_buffer[0] is not None, "For cuda graph compatibility, kv_indices_out_buffer must be provided."

    grid = (running_steps, bs, topk)
    generate_kv_indices_kernel[grid](
        old_kv_indptr, past_kv_indices, out_cache_loc_tensor,
        kv_indptr, kv_indices, 
        running_steps, bs, kv_indices_stride_0, kv_indptr_stride_0,
        topk, BLOCK_SIZE=block_size, num_warps=num_warps
    )
    # for cuda graph compatibility, return the raw buffers if provided
    if kv_out_buffer is not None:
        return (kv_indptr, kv_indices)
    kv_indptr = kv_indptr.reshape(running_steps, (topk * bs + 1))
    # Convert to same format as PyTorch implementation
    kv_indices_outs = []
    for i in range(running_steps):
        # Extract indptr for current step
        # step_indptr = kv_indptr[i * (topk * bs + 1):(i + 1) * (topk * bs + 1)]        
        # Extract indices for current step using PyTorch offset formula
        offset_start = (seq_lens_sum * i + sum(range(1, i + 1)) * bs) * topk
        indices_size = (seq_lens_sum + (i + 1) * bs) * topk
        step_indices = kv_indices[offset_start:offset_start + indices_size]
        
        kv_indices_outs.append((step_indices))

    return (kv_indptr, kv_indices_outs)


def generate_kv_indices_for_mtd(kv_out_buffer: Tuple[torch.Tensor, torch.Tensor], old_kv_indptr: torch.Tensor, past_kv_indices: torch.Tensor,
            out_cache_loc_tensor:torch.Tensor, seq_lens_sum:int, bs:int, topk:int, running_steps:int):
    triton_out = generate_kv_indices_for_mtd_triton(
        kv_out_buffer, old_kv_indptr, past_kv_indices, out_cache_loc_tensor, seq_lens_sum, running_steps, topk)
    return triton_out
    device = past_kv_indices.device
    kv_indptr_out = torch.zeros((running_steps*(topk*bs+1),), dtype=torch.int32, device=device)
    kv_indices_out = torch.empty(
        ((seq_lens_sum*running_steps + sum(range(1,running_steps+1)) * bs) * topk,),
        dtype=past_kv_indices.dtype,
        device=device,
    )

    seq_lens = old_kv_indptr[1:] - old_kv_indptr[:-1]
    kv_indices_outs = []
    cur_kv_seqlen = seq_lens[..., None] + 1
    for i in range(running_steps):
        # kv_indptr = torch.zeros((topk*bs+1,), dtype=torch.int32, device=device)
        kv_indptr = kv_indptr_out[i*(topk*bs+1):(i+1)*(topk*bs+1)]
        kv_indptr[1:] = cur_kv_seqlen.expand(-1, topk).flatten().cumsum(dim=-1, dtype=torch.int32)
        offset_start = (seq_lens_sum*i + sum(range(1,i+1)) * bs) * topk
        indices_size = (seq_lens_sum + (i + 1) * bs) * topk
        kv_indices = kv_indices_out[offset_start:offset_start + indices_size]
        # kv_indices = torch.empty(
        #     ((seq_lens_sum + (i + 1) * bs) * topk,),
        #     dtype=past_kv_indices.dtype,
        #     device=device,
        # )
        batch_kv_start = 0
        kv_indices_batch_index = 0
        # inside_batch_kv_start = 0
        for bs_i in range(bs):
            kv_indices_batch_i = kv_indices[batch_kv_start : batch_kv_start + ((seq_lens[bs_i]+i+1) * topk)]
            kv_indices_batch_i = kv_indices_batch_i.view(topk, seq_lens[bs_i]+1+i)
            batch_kv_start = batch_kv_start + ((seq_lens[bs_i]+i+1) * topk)
            for sb_kv_indices in kv_indices_batch_i: # topk 
                sb_kv_indices[:-i-1] = past_kv_indices[old_kv_indptr[bs_i]:old_kv_indptr[bs_i+1]]
                sb_kv_indices[-i-1:] = out_cache_loc_tensor[kv_indices_batch_index,:i+1]
                kv_indices_batch_index += 1
            # inside_batch_kv_start += seq_lens[bs_i]
        kv_indices_outs.append(kv_indices.flatten())

        cur_kv_seqlen = cur_kv_seqlen + 1 # don't do it inplace to avoid affecting next step
    assert sum([torch.allclose(tt[0], pt[0]) and torch.allclose(tt[1], pt[1]) for tt, pt in zip(triton_out, kv_indices_outs)])==3
    return (kv_indptr_out.view(running_steps, -1), kv_indices_outs)
