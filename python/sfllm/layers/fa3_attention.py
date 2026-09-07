from dataclasses import dataclass

import torch
from sgl_kernel.flash_attn import get_scheduler_metadata

from sfllm.engine.forward_params import ForwardBatch, ForwardMode
from sfllm.kernels.fa3_attention import build_page_table, fa3_attention_fwd
from sfllm.server_args import get_global_server_args


@dataclass(frozen=True)
class FA3AttentionMetadata:
    page_table: torch.Tensor
    cache_seqlens: torch.Tensor
    scheduler_metadata: torch.Tensor
    qo_indptr: torch.Tensor
    max_query_len: int


class FA3AttentionWorkspace:
    def __init__(self, max_batch_size: int, max_context_length: int, device) -> None:
        self.page_table = torch.empty(
            (max_batch_size, max_context_length),
            dtype=torch.int32,
            device=device,
        )
        self.cache_seqlens = torch.empty(
            max_batch_size, dtype=torch.int32, device=device
        )
        self.decode_qo_indptr = torch.arange(
            max_batch_size + 1, dtype=torch.int32, device=device
        )

    def prepare(
        self,
        forward_batch: ForwardBatch,
        q: torch.Tensor,
        layer,
    ) -> FA3AttentionMetadata:
        if forward_batch.custom_mask is not None:
            raise NotImplementedError("FA3 does not support custom attention masks.")

        batch_size = forward_batch.kv_indptr.shape[0] - 1
        if batch_size > self.page_table.shape[0]:
            raise ValueError(
                f"FA3 batch size {batch_size} exceeds the configured maximum "
                f"{self.page_table.shape[0]}."
            )

        is_decode = forward_batch.forward_mode == ForwardMode.DECODE
        if is_decode:
            if q.shape[0] != batch_size:
                raise ValueError("FA3 decode expects one query token per sequence.")
            qo_indptr = self.decode_qo_indptr[: batch_size + 1]
            max_query_len = 1
        else:
            if forward_batch.qo_indptr is None:
                raise ValueError("FA3 extend attention requires qo_indptr metadata.")
            qo_indptr = forward_batch.qo_indptr
            max_query_len = int(forward_batch.max_extend_len)
            if max_query_len <= 0:
                raise ValueError("FA3 extend attention requires a positive query length.")

        page_table = self.page_table[:batch_size]
        cache_seqlens = self.cache_seqlens[:batch_size]
        build_page_table(
            forward_batch.kv_indices,
            forward_batch.kv_indptr,
            forward_batch.out_cache_loc,
            qo_indptr,
            page_table,
            cache_seqlens,
            append_query=not is_decode,
        )
        scheduler_metadata = get_scheduler_metadata(
            batch_size=batch_size,
            max_seqlen_q=max_query_len,
            max_seqlen_k=page_table.shape[1],
            num_heads=layer.tp_q_head_num,
            num_heads_k=layer.tp_k_head_num,
            headdim=layer.qk_head_dim,
            cache_seqlens=cache_seqlens,
            qkv_dtype=q.dtype,
            cu_seqlens_q=qo_indptr,
            page_size=1,
            causal=layer.is_causal,
            num_splits=0,
        )
        return FA3AttentionMetadata(
            page_table,
            cache_seqlens,
            scheduler_metadata,
            qo_indptr,
            max_query_len,
        )


class FA3AttentionBackend:
    def __init__(
        self,
        layer_idx: int,
        *,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> None:
        self.layer_idx = layer_idx
        self.workspace = None
        if layer_idx == 0:
            server_args = get_global_server_args()
            self.workspace = FA3AttentionWorkspace(
                int(server_args.cuda_graph_max_bs),
                server_args.max_context_length
                + server_args.speculative_num_draft_tokens,
                torch.device("cuda"),
            )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        if kwargs:
            raise NotImplementedError(
                f"FA3 does not support attention options: {sorted(kwargs)}."
            )
        if forward_batch.past_key_values is None:
            return torch.zeros_like(q)
        if save_kv_cache:
            forward_batch.update(k, v, layer.layer_id)

        if self.workspace is not None:
            metadata = self.workspace.prepare(forward_batch, q, layer)
            forward_batch._fa3_attention_metadata = metadata
        else:
            metadata = getattr(forward_batch, "_fa3_attention_metadata", None)
            if metadata is None:
                raise RuntimeError("FA3 metadata must be prepared by the first layer.")

        output = torch.empty_like(q)
        k_buffer, v_buffer = forward_batch.past_key_values[self.layer_idx]
        return fa3_attention_fwd(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            output.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            k_buffer,
            v_buffer,
            metadata,
            layer.scaling,
            layer.is_causal,
        )
