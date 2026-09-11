"""DFlash2 proposal logic on SFLLM's shared speculative protocol."""

import torch

from sfllm.engine.forward_params import ForwardBatch, ForwardMode
from sfllm.engine.schedule_batch import BatchResult, ScheduleBatch
from sfllm.kernels.dflash2 import (
    dflash2_selector_greedy_walk,
    prepare_dflash2_block,
)
from sfllm.models.dflash2 import DFlash2Config
from sfllm.models.interfaces import HasBatchState
from sfllm.model_loader.model_config import ModelConfig
from sfllm.server_args import ServerArgs
from sfllm.spec_decoding.spec_utils import EagleSpecInput, EagleVerifyInput
from sfllm.spec_decoding.spec_worker import SpeculativeWorker


def _validate_model_pair(
    draft_config,
    target_config,
    dflash2_config: DFlash2Config,
) -> None:
    if getattr(target_config, "model_type", None) not in ("qwen3", "qwen3_5_text"):
        raise ValueError("DFlash2 supports Qwen3 and Qwen3.5 targets.")
    for name in ("hidden_size", "vocab_size"):
        draft_value = getattr(draft_config, name, None)
        target_value = getattr(target_config, name, None)
        if draft_value != target_value:
            raise ValueError(
                "DFlash2 draft/target shape mismatch: "
                f"draft {name}={draft_value!r}, target {name}={target_value!r}."
            )

    target_num_layers = int(getattr(target_config, "num_hidden_layers", 0) or 0)
    invalid_layer_ids = [
        layer_id
        for layer_id in dflash2_config.target_layer_ids
        if layer_id >= target_num_layers
    ]
    if invalid_layer_ids:
        raise ValueError(
            "DFlash2 target_layer_ids are outside the target model: "
            f"{invalid_layer_ids}."
        )


class DFlash2Worker(SpeculativeWorker):
    """DFlash2 model logic on top of SFLLM's Eagle overlap contract."""

    def __init__(self, server_args: ServerArgs) -> None:
        if not server_args.speculative_draft_model_path:
            raise ValueError("DFlash2 requires --speculative-draft-model-path.")
        if server_args.quantization is not None:
            raise ValueError(
                "DFlash2 reads target quantization from the checkpoint; omit --quantization."
            )

        draft_config = ModelConfig(
            server_args.speculative_draft_model_path
        ).hf_config
        checkpoint_config = DFlash2Config.from_hf_config(draft_config)
        if (
            "sliding_attention" in draft_config.layer_types
            and server_args.attention_backend != "fa3"
        ):
            raise ValueError("DFlash2 sliding attention requires --attention-backend fa3.")

        # A DFlash block maps directly to the shared Eagle protocol width.
        self.block_size = server_args.speculative_num_draft_tokens
        if self.block_size is None:
            self.block_size = checkpoint_config.block_size
        if not 2 <= self.block_size <= checkpoint_config.block_size:
            raise ValueError(
                f"DFlash2 draft token count must be between 2 and {checkpoint_config.block_size}."
            )
        server_args.speculative_eagle_topk = 1
        server_args.speculative_num_steps = self.block_size - 1
        server_args.speculative_num_draft_tokens = self.block_size

        super().__init__(server_args)
        self.dflash2_config = self.draft_model_runner.model.dflash_config
        for layer in self.draft_model_runner.model.layers:
            layer.attention_conv.block_size = self.block_size
            layer.mlp_conv.block_size = self.block_size
        _validate_model_pair(
            draft_config=self.draft_model_runner.get_config(),
            target_config=self.target_model_runner.get_config(),
            dflash2_config=self.dflash2_config,
        )

        target_model = self.target_model_runner.model
        target_model.set_layers_to_capture(
            list(self.dflash2_config.target_layer_ids)
        )

        self._profile_models()
        self.init_memory_pools()

        self._allocate_decode_buffers()
        self.init_e2e_runner(self.forward_decode_e2e)

    @torch.inference_mode()
    def _profile_models(self) -> None:
        self.target_model_runner.profile_run()
        token_count = self.block_size * min(8, int(self.server_args.cuda_graph_max_bs))
        input_ids = torch.zeros(token_count, dtype=torch.int64, device="cuda")
        positions = torch.arange(token_count, dtype=torch.int64, device="cuda")
        embeddings = self.target_model_runner.model.get_input_embeddings()(input_ids)
        draft_batch = ForwardBatch(None)
        draft_batch.forward_mode = ForwardMode.DRAFT_EXTEND
        self.draft_model_runner.prepare_attention(draft_batch, token_count)
        self.draft_model_runner.model(
            input_ids=input_ids,
            positions=positions,
            forward_batch=draft_batch,
            input_embeds=embeddings,
        )

    def _allocate_decode_buffers(self) -> None:
        max_bs = int(self.server_args.cuda_graph_max_bs)
        block = self.block_size
        protocol_width = self.draft_model_runner.hidden_states_buffer.shape[-1]
        device = torch.device("cuda", int(self.target_model_runner.device_id))

        scratch_count = max_bs * block
        assert self.draft_mem_pool.can_alloc(scratch_count)
        self._draft_scratch_locs = torch.tensor(
            self.draft_mem_pool.persist_alloc_block_from_rear(scratch_count),
            dtype=torch.int64,
            device=device,
        )
        self._block_ids = torch.empty((max_bs, block), dtype=torch.int64, device=device)
        self._positions = torch.empty_like(self._block_ids)
        self._proposals = torch.empty(
            (max_bs, block - 1), dtype=torch.int64, device=device
        )
        self._candidates = torch.empty_like(self._block_ids)
        self._protocol_hidden = torch.empty(
            (max_bs * block, protocol_width), dtype=self.dtype, device=device
        )
        token_indices = torch.arange(block, dtype=torch.int64, device=device)
        self._retrieve_index = (
            token_indices[None]
            + torch.arange(max_bs, dtype=torch.int64, device=device)[:, None] * block
        )
        self._retrieve_next_token = token_indices.add(1).repeat(max_bs, 1)
        self._retrieve_next_token[:, -1] = -1
        self._retrieve_next_sibling = torch.full_like(self._retrieve_index, -1)

    @torch.inference_mode()
    def forward(self, batch: ScheduleBatch) -> BatchResult:
        model = self.target_model_runner.model
        if isinstance(model, HasBatchState):
            model.prepare_batch_state(batch)
        if batch.forward_batch.forward_mode == ForwardMode.EXTEND:
            return self._forward_prefill(batch)
        return self.forward_e2e(batch)

    def accept(self, batch, proposal, verification):
        result = super().accept(batch, proposal, verification)
        model = self.target_model_runner.model
        if isinstance(model, HasBatchState):
            # The linear chain commits its anchor plus the accepted drafts.
            model.commit_speculative_state(result[3])
        return result

    @torch.inference_mode()
    def _forward_prefill(self, batch: ScheduleBatch) -> BatchResult:
        if not all(sequence.sampling_params.is_greedy for sequence in batch):
            raise ValueError("DFlash2 currently supports greedy requests only.")

        output = self.target_model_runner.forward(batch)
        if output.aux_hidden_states is None:
            raise RuntimeError("DFlash2 target prefill returned no captured states.")
        self.draft_model_runner.model.materialize_target_kv(
            target_hidden=torch.cat(output.aux_hidden_states, dim=-1),
            positions=batch.position_ids,
            cache_locs=batch.forward_batch_spec.out_cache_loc,
            kv_buffers=self.draft_mem_pool.kv_buffers,
        )

        spec_info = EagleSpecInput(
            accept_length=torch.full(
                (len(batch),),
                -1,
                dtype=torch.int32,
                device=output.next_token_ids.device,
            ),
            verified_id=output.next_token_ids,
            hidden_states=torch.zeros(
                (
                    len(batch) * self.block_size,
                    self.draft_model_runner.hidden_states_buffer.shape[1],
                ),
                dtype=self.dtype,
                device=output.next_token_ids.device,
            ),
        )
        spec_info.hash = batch.get_seq_groups_hash()
        batch.spec_info = spec_info
        batch.forward_batch_spec.spec_info = spec_info
        output.spec_info = spec_info
        output.aux_hidden_states = None
        return output

    def _draft_block_batch(self, batch: ScheduleBatch) -> ForwardBatch:
        batch_size = len(batch)
        token_count = batch_size * self.block_size
        forward_batch = ForwardBatch(self.draft_mem_pool)
        forward_batch.forward_mode = ForwardMode.DRAFT_EXTEND
        forward_batch.qo_indptr = batch.forward_batch.qo_indptr
        forward_batch.kv_indptr = batch.forward_batch.kv_indptr
        forward_batch.kv_indices = batch.forward_batch_spec.kv_indices_mtd
        forward_batch.out_cache_loc = self._draft_scratch_locs[:token_count]
        forward_batch.seq_lens = batch.forward_batch.seq_lens
        forward_batch.max_extend_len = self.block_size
        return forward_batch

    def commit_previous(self, batch: ScheduleBatch) -> None:
        """Advance DFlash2's draft KV state with the previous accepted context."""
        spec_info = batch.spec_info
        cache_locs = batch.forward_batch_spec.out_cache_loc
        count = cache_locs.shape[0]
        valid_count = torch.clamp(spec_info.accept_length + 1, min=0).sum()
        valid = torch.arange(count, device=cache_locs.device) < valid_count
        safe_cache_locs = torch.where(
            valid,
            cache_locs,
            self._draft_scratch_locs[:count],
        )
        hidden_size = int(self.config.hidden_size)
        self.draft_model_runner.model.materialize_context_kv(
            context=spec_info.hidden_states[:count, :hidden_size],
            positions=batch.forward_batch_spec.position_ids_extend[:count],
            cache_locs=safe_cache_locs,
            kv_buffers=self.draft_mem_pool.kv_buffers,
        )

    def proposal(self, batch: ScheduleBatch) -> EagleVerifyInput:
        """Build one linear DFlash2 proposal block."""
        batch_size = len(batch)
        block = self.block_size
        spec_info = batch.spec_info

        anchor_indices = batch.forward_batch_spec.qo_indptr[1:] - 1
        prepare_dflash2_block(
            anchor_tokens=spec_info.verified_id[anchor_indices],
            anchor_positions=batch.position_ids,
            block_ids_out=self._block_ids[:batch_size],
            positions_out=self._positions[:batch_size],
            mask_token_id=self.dflash2_config.mask_token_id,
        )

        flat_ids = self._block_ids[:batch_size].reshape(-1)
        flat_positions = self._positions[:batch_size].reshape(-1)
        embeddings = self.target_model_runner.model.get_input_embeddings()(flat_ids)
        forward_batch = self._draft_block_batch(batch)
        self.draft_model_runner.prepare_attention(forward_batch, flat_ids.shape[0])
        draft_hidden = self.draft_model_runner.model(
            input_ids=flat_ids,
            positions=flat_positions,
            forward_batch=forward_batch,
            input_embeds=embeddings,
        ).view(batch_size, block, -1)

        prediction_hidden = draft_hidden[:, 1:]
        candidate_ids, unary_logits = self.draft_model_runner.model.compute_candidates(
            prediction_hidden.reshape(-1, prediction_hidden.shape[-1]),
            self.target_model_runner.model.lm_head.weight,
        )
        top_k = self.dflash2_config.selector_top_k
        candidate_ids = candidate_ids.view(batch_size, block - 1, top_k)
        selector_scores = (
            self.draft_model_runner.model.candidate_selector.build_lattice(
                candidate_ids=candidate_ids,
                unary_logits=unary_logits.view(batch_size, block - 1, top_k),
                hidden_states=prediction_hidden,
                anchor_token_ids=self._block_ids[:batch_size, 0],
            )
        )
        dflash2_selector_greedy_walk(
            candidate_ids, selector_scores, self._proposals[:batch_size]
        )

        candidates = self._candidates[:batch_size]
        candidates[:, 0].copy_(self._block_ids[:batch_size, 0])
        candidates[:, 1:].copy_(self._proposals[:batch_size])
        return EagleVerifyInput(
            draft_token=candidates.reshape(-1),
            custom_mask=None,
            positions=flat_positions,
            retrive_index=self._retrieve_index[:batch_size],
            retrive_next_token=self._retrieve_next_token[:batch_size],
            retrive_next_sibling=self._retrieve_next_sibling[:batch_size],
            retrive_cum_len=None,
            spec_steps=block - 1,
            topk=1,
            draft_token_num=block,
        )

    def _adapt_verified_hidden_states(
        self, batch: ScheduleBatch, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Project target states into DFlash2's next-step context."""
        token_count = len(batch) * self.block_size
        target_context = self.draft_model_runner.model.project_target_hidden(
            hidden_states
        )
        protocol_hidden = self._protocol_hidden[:token_count]
        protocol_hidden.zero_()
        protocol_hidden[:, : target_context.shape[1]].copy_(target_context)
        return protocol_hidden


__all__ = ["DFlash2Worker"]
