import torch
from sfllm.engine.forward_params import ForwardBatch,ForwardMode

from sfllm.kernels.decode_attention import (
    decode_attention_fwd, get_num_kv_splits_triton,
)
from sfllm.kernels.extend_attention import (
    extend_attention_fwd,
)
from sfllm.utils.nutils import get_device_core_count
import triton

class RaggedAttention:
    def __init__(self, layer_idx, **kwargs):
        self.layer_idx = layer_idx
        self.num_head = kwargs["num_heads"]
        self.num_kv_head = kwargs["num_kv_heads"]
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
        q_extend: torch.Tensor,
        k_extend: torch.Tensor,
        v_extend: torch.Tensor,
        layer,
        forward_batch: ForwardBatch,
        save_kv_cache=True,):
        if save_kv_cache:
            forward_batch.update(k_extend, v_extend, layer.layer_id)

        custom_mask=None
        is_causal=True
        mask_indptr=None
        sm_scale=None
        logit_cap=0.0
        skip_prefix_custom_mask=True
        sliding_window_size=-1
        sinks=None
        window_kv_offsets=None
        xai_temperature_len=-1
        if forward_batch.past_key_values is None:
            return torch.zeros_like(q_extend)

        k_buffer,v_buffer,qo_indptr,kv_indptr,kv_indices,max_len_extend = (
            forward_batch.past_key_values[self.layer_idx][0],
            forward_batch.past_key_values[self.layer_idx][1],
            forward_batch.qo_indptr,
            forward_batch.kv_indptr,
            forward_batch.kv_indices,
            forward_batch.max_extend_len,
        )
        o_extend = torch.empty_like(q_extend)
        extend_attention_fwd(
            q_extend.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            k_extend.contiguous(),
            v_extend.contiguous(),
            o_extend.view(-1, layer.tp_q_head_num, layer.v_head_dim),
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


    def forward_decode(self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        forward_batch: ForwardBatch,
        save_kv_cache=True,):
        if save_kv_cache:
            forward_batch.update(k, v, layer.layer_id)
        o = torch.empty_like(q)
        decode_attention_fwd(
            q,
            forward_batch.past_key_values[self.layer_idx][0],
            forward_batch.past_key_values[self.layer_idx][1],
            o,
            forward_batch.kv_indptr,
            forward_batch.kv_indices,
            forward_batch.attn_logits,
            forward_batch.attn_lse,
            forward_batch.num_kv_splits,
            forward_batch.max_kv_splits,
            layer.scaling,
        )
        return o

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        """Run forward on an attention layer."""
        if forward_batch.forward_mode == ForwardMode.DECODE:
            return self.forward_decode(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=save_kv_cache,
                **kwargs,
            )
        else:
            return self.forward_extend(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=save_kv_cache,
                **kwargs,
            )