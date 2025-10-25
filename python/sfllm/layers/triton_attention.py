import torch
from sfllm.kernels.decode_attention import (
    decode_attention_fwd, get_num_kv_splits_triton,
)
from sfllm.kernels.extend_attention import (
    extend_attention_fwd,
)
from sfllm.utils.nutils import get_device_core_count
import triton

class RaggedAttention:
    def __init__(self, layer_idx, config):
        self.layer_idx = layer_idx
        self.config = config
        self.num_head = config.num_attention_heads
        self.num_kv_head = config.num_attention_heads // 2
        self.device_core_count = get_device_core_count()
        self.max_kv_splits = 16
        self.static_kv_splits = False

    def get_num_kv_splits(
        self,
        num_kv_splits: torch.Tensor,
        seq_lens: torch.Tensor,
    ):
        num_token, num_seq = num_kv_splits.shape[0], seq_lens.shape[0]
        # NOTE(alcanderian): Considering speculative_decodeing,
        # num_kv_splits.shape[0] will be topk * real_num_token.
        # And the real_num_token is num_seq in decoding phase.
        num_group = num_token // num_seq

        assert num_group * num_seq == num_token, (
            f"num_seq({num_seq}), num_token({num_token}), something goes wrong!"
        )

        if self.static_kv_splits or self.device_core_count <= 0:
            num_kv_splits.fill_(self.max_kv_splits)
            return

        if num_seq < 256:
            SCHEDULE_SEQ = 256
        else:
            SCHEDULE_SEQ = triton.next_power_of_2(num_seq)

        get_num_kv_splits_triton[(1,)](
            num_kv_splits,
            seq_lens,
            num_seq,
            num_group,
            self.num_head,
            self.num_kv_head,
            self.max_kv_splits,
            self.device_core_count,
            MAX_NUM_SEQ=SCHEDULE_SEQ,
        )

    def forward_extend(self,
        q_extend,
        k_extend,
        v_extend,
        forward_metadata,
        custom_mask=None,
        is_causal=True,
        mask_indptr=None,
        sm_scale=None,
        logit_cap=0.0,
        skip_prefix_custom_mask=True,
        sliding_window_size=-1,
        sinks=None,
        window_kv_offsets=None,
        xai_temperature_len=-1):
        k_buffer,v_buffer,qo_indptr,kv_indptr,kv_indices,max_len_extend = (
            forward_metadata.past_key_values[self.layer_idx][0],
            forward_metadata.past_key_values[self.layer_idx][1],
            forward_metadata.qo_indptr,
            forward_metadata.kv_indptr,
            forward_metadata.kv_indices,
            forward_metadata.max_extend_len,
        )
        o_extend = torch.empty_like(q_extend)
        extend_attention_fwd(
            q_extend,
            k_extend,
            v_extend,
            o_extend,
            k_buffer,
            v_buffer,
            qo_indptr,
            kv_indptr,
            kv_indices,
            custom_mask,
            is_causal,
            mask_indptr,
            max_len_extend,
            sm_scale,
            logit_cap,
            skip_prefix_custom_mask,
            sliding_window_size,
            sinks,
            window_kv_offsets,
            xai_temperature_len,
        )
        return o_extend


    def forward_decode(self,q,
        forward_metadata,
        sm_scale):
        o = torch.empty_like(q)
        decode_attention_fwd(
            q,
            forward_metadata.past_key_values[self.layer_idx][0],
            forward_metadata.past_key_values[self.layer_idx][1],
            o,
            forward_metadata.kv_indptr,
            forward_metadata.kv_indices,
            forward_metadata.attn_logits,
            forward_metadata.attn_lse,
            forward_metadata.num_kv_splits,
            forward_metadata.max_kv_splits,
            sm_scale,
        )
        return o