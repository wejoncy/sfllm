from dataclasses import dataclass

import torch
from sgl_kernel.flash_attn import get_scheduler_metadata

from sfllm.engine.forward_params import ForwardBatch, ForwardMode
from sfllm.kernels.fa3_attention import build_page_table, fa3_attention_fwd
from sfllm.layers.verify_attention import prepare_verify_attention, verify_attention


@dataclass(frozen=True)
class FA3AttentionMetadata:
    page_table: torch.Tensor
    cache_seqlens: torch.Tensor
    scheduler_metadata: tuple[torch.Tensor, ...]
    qo_indptr: torch.Tensor
    max_query_len: int
    scheduler_params: tuple[dict, ...]
    layer_index_mapping: dict[int, int]

    @classmethod
    def allocate(cls, max_batch_size: int, max_context_length: int, device, layer_metadata):
        page_table = torch.empty(
            (max_batch_size, max_context_length),
            dtype=torch.int32,
            device=device,
        )
        cache_seqlens = torch.empty(
            max_batch_size, dtype=torch.int32, device=device
        )
        qo_indptr = torch.arange(
            max_batch_size + 1, dtype=torch.int32, device=device
        )
        scheduler_params = []
        layer_index_mapping = {}
        for layer_id, m in layer_metadata.items():
            params = dict(
                num_heads=m["num_heads"], num_heads_k=m["num_kv_heads"],
                headdim=m["head_dim"], headdim_v=m["v_head_dim"],
                qkv_dtype=m["dtype"], causal=m["is_causal"],
                window_size=m["window_size"],
            )
            if params not in scheduler_params:
                scheduler_params.append(params)
            layer_index_mapping[layer_id] = scheduler_params.index(params)
        return cls(
            page_table, cache_seqlens, (), qo_indptr, 1,
            tuple(scheduler_params), layer_index_mapping,
        )

    def prepare_metadata(
        self, page_table, cache_seqlens, qo_indptr, max_query_len, *, is_tree_verify=False,
    ):
        scheduler_metadata = tuple(
            get_scheduler_metadata(
                batch_size=cache_seqlens.shape[0],
                max_seqlen_q=max_query_len,
                max_seqlen_k=page_table.shape[1],
                cache_seqlens=cache_seqlens,
                cu_seqlens_q=qo_indptr,
                page_size=1,
                num_splits=0,
                **(params | {"causal": False} if is_tree_verify else params),
            )
            for params in self.scheduler_params
        )
        return FA3AttentionMetadata(
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            scheduler_metadata=scheduler_metadata,
            qo_indptr=qo_indptr,
            max_query_len=max_query_len,
            scheduler_params=self.scheduler_params,
            layer_index_mapping=self.layer_index_mapping,
        )

    def prepare(
        self,
        forward_batch: ForwardBatch,
        num_tokens: int,
    ) -> tuple:
        if forward_batch.past_key_values is None:
            return None, None
        is_tree_verify = (
            forward_batch.forward_mode == ForwardMode.TARGET_VERIFY
            and forward_batch.custom_mask is not None
        )

        if is_tree_verify and any(p["window_size"] != (-1, -1) for p in self.scheduler_params):
            raise NotImplementedError("FA3 tree verification does not support sliding attention.")

        batch_size = forward_batch.kv_indptr.shape[0] - 1
        if batch_size > self.page_table.shape[0]:
            raise ValueError(
                f"FA3 batch size {batch_size} exceeds the configured maximum "
                f"{self.page_table.shape[0]}."
            )

        is_decode = forward_batch.forward_mode == ForwardMode.DECODE
        if is_decode:
            if num_tokens != batch_size:
                raise ValueError("FA3 decode expects one query token per sequence.")
            qo_indptr = self.qo_indptr[: batch_size + 1]
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
            append_query=not is_decode and not is_tree_verify,
        )
        metadata = self.prepare_metadata(
            page_table, cache_seqlens, qo_indptr, max_query_len,
            is_tree_verify=is_tree_verify,
        )
        if is_tree_verify:
            return prepare_verify_attention(
                metadata, forward_batch, num_tokens
            )
        return metadata, None


class FA3AttentionBackend:
    def __init__(self, model_runner, layer_metadata):
        args = model_runner.server_args
        max_batch_size = int(args.max_running_requests)
        if args.speculative_algorithm == "eagle3":
            max_batch_size *= args.speculative_eagle_topk
        self.metadata = FA3AttentionMetadata.allocate(
            max_batch_size,
            args.max_context_length + args.speculative_num_draft_tokens,
            torch.device("cuda", model_runner.device_id),
            layer_metadata,
        )

    def prepare(self, forward_batch, num_tokens):
        self.forward_metadata = self.metadata.prepare(forward_batch, num_tokens)

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
        if forward_batch.forward_mode == ForwardMode.TARGET_VERIFY:
            output = verify_attention(
                q, k, v, layer, forward_batch, save_kv_cache,
                metadata=self.forward_metadata,
            )
            if output is not None:
                return output
        if forward_batch.past_key_values is None:
            return torch.zeros_like(q)
        if forward_batch.custom_mask is not None:
            raise NotImplementedError("FA3 does not support custom attention masks.")
        if save_kv_cache:
            forward_batch.update(k, v, layer.layer_id)

        metadata = self.forward_metadata[0]

        output = torch.empty_like(q)
        k_buffer, v_buffer = forward_batch.past_key_values[layer.layer_id]
        return fa3_attention_fwd(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            output.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            k_buffer,
            v_buffer,
            metadata,
            layer.scaling,
            layer.is_causal,
            layer_id=layer.layer_id,
            window_size=layer.window_size,
        )
