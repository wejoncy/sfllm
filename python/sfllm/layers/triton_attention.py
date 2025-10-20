import torch
from sfllm.kernels.decode_attention import (
    decode_attention_fwd,
)
from sfllm.kernels.extend_attention import (
    extend_attention_fwd,
)


def extend_attention_fwd_interface(q_extend,
    k_extend,
    v_extend,
    k_buffer,
    v_buffer,
    qo_indptr,
    kv_indptr,
    kv_indices,
    max_len_extend,
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

def decode_attention_fwd_interface(q,
                                    k_buffer,
                                    v_buffer,
                                    kv_indptr,
                                    kv_indices,
                                    attn_logits,
                                    attn_lse,
                                    num_kv_splits,
                                    max_kv_splits,
                                    sm_scale):
    o = torch.empty_like(q)
    decode_attention_fwd(
        q,
        k_buffer,
        v_buffer,
        o,
        kv_indptr,
        kv_indices,
        attn_logits,
        attn_lse,
        num_kv_splits,
        max_kv_splits,
        sm_scale,
    )