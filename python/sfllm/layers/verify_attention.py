"""EAGLE tree verification with FA3 prefix/suffix attention, as in SGLang."""

import torch

from sfllm.server_args import get_global_server_args

try:
    from sgl_kernel import merge_state_v2
    from sgl_kernel.flash_attn import get_scheduler_metadata
    from sfllm.kernels.fa3_attention import fa3_attention_fwd
except ImportError:
    merge_state_v2 = None


FA3_VERIFY_AVAILABLE = (
    merge_state_v2 is not None
    and torch.version.cuda is not None
    and torch.cuda.is_available()
    and torch.cuda.get_device_capability()[0] == 9
    and tuple(map(int, torch.version.cuda.split(".")[:2])) >= (12, 3)
)


def prepare_verify_attention(workspace, forward_batch, q, layer):
    prefix = workspace.prepare(forward_batch, q, layer)
    batch_size = prefix.cache_seqlens.shape[0]
    width = prefix.max_query_len
    if q.shape[0] != batch_size * width:
        raise ValueError("Tree verification requires equal query counts per request.")

    # Every query sees the common prefix and its own visible tree nodes.
    offsets = torch.arange(width, device=q.device)
    prefix_lens = prefix.cache_seqlens[:, None, None]
    mask_indices = (
        forward_batch.mask_indptr[:-1, None, None]
        + offsets[None, :, None] * (prefix_lens + width)
        + prefix_lens + offsets[None, None, :]
    )
    mask = forward_batch.custom_mask[mask_indices]
    order = torch.where(mask, offsets, offsets + width).argsort(dim=-1)
    page_table = (
        forward_batch.out_cache_loc.view(batch_size, 1, width)
        .expand(-1, width, -1).gather(2, order).reshape(-1, width).int()
    )
    cache_seqlens = mask.sum(dim=-1, dtype=torch.int32).flatten()
    qo_indptr = torch.arange(q.shape[0] + 1, dtype=torch.int32, device=q.device)
    scheduler = get_scheduler_metadata(
        batch_size=q.shape[0],
        max_seqlen_q=1,
        max_seqlen_k=width,
        num_heads=layer.tp_q_head_num,
        num_heads_k=layer.tp_k_head_num,
        headdim=layer.qk_head_dim,
        cache_seqlens=cache_seqlens,
        qkv_dtype=q.dtype,
        cu_seqlens_q=qo_indptr,
        page_size=1,
        causal=False,
        num_splits=0,
    )
    suffix = type(prefix)(page_table, cache_seqlens, scheduler, qo_indptr, 1)
    return prefix, suffix


def verify_attention(q, k, v, layer, forward_batch, save_kv_cache=True, *, workspace):
    if (
        not FA3_VERIFY_AVAILABLE
        or get_global_server_args().speculative_algorithm != "eagle3"
        or forward_batch.custom_mask is None
        or q.dtype not in (torch.float16, torch.bfloat16)
        or layer.qk_head_dim != layer.v_head_dim
        or layer.qk_head_dim > 256
        or layer.qk_head_dim % 8
        or layer.sliding_window_size > 0
        or layer.logit_cap
        or layer.is_cross_attention
    ):
        return None
    if forward_batch.past_key_values is None:
        return torch.zeros_like(q)
    if save_kv_cache:
        forward_batch.update(k, v, layer.layer_id)
    if workspace is not None:
        forward_batch._verify_attention_metadata = prepare_verify_attention(
            workspace, forward_batch, q, layer
        )
    prefix, suffix = forward_batch._verify_attention_metadata
    query = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
    k_buffer, v_buffer = forward_batch.past_key_values[layer.layer_id]
    prefix_out, prefix_lse, *_ = fa3_attention_fwd(
        query, torch.empty_like(query), k_buffer, v_buffer, prefix, layer.scaling,
        False, return_softmax_lse=True,
    )
    suffix_out, suffix_lse, *_ = fa3_attention_fwd(
        query, torch.empty_like(query), k_buffer, v_buffer, suffix, layer.scaling,
        False, return_softmax_lse=True,
    )
    output, _ = merge_state_v2(
        prefix_out, prefix_lse.T.contiguous(),
        suffix_out, suffix_lse.T.contiguous(), v_merged=prefix_out,
    )
    return output.view(q.shape)
