import torch
from sfllm.kernels.decode_attention import (
    decode_attention_fwd,
)
from sfllm.kernels.extend_attention import (
    extend_attention_fwd,
)

class RaggedAttention:
    def __init__(self, layer_idx, config):
        self.layer_idx = layer_idx
        self.config = config

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