import logging
from array import array
from typing import Callable

import torch

from sfllm.engine.forward_params import ForwardMode
from sfllm.engine.model_runner import ModelRunner
from sfllm.engine.schedule_batch import BatchResult, ScheduleBatch
from sfllm.server_args import ServerArgs
from sfllm.spec_decoding.spec_e2e_cuda_graph_runner import (
    SpeculativeE2ECudaGraphRunner,
)
from sfllm.spec_decoding.spec_utils import EagleVerifyInput

logger = logging.getLogger(__name__)


class SpeculativeWorker:
    """Shared worker-side implementation of SFLLM's speculative protocol."""

    def __init__(self, server_args: ServerArgs) -> None:
        self.server_args = server_args
        self.draft_model_runner = ModelRunner(server_args, is_draft=True)
        self.target_model_runner = ModelRunner(server_args)

        target_model = self.target_model_runner.model
        server_args.model_config = target_model.config
        self.tokenizer = self.target_model_runner.tokenizer
        self.detokenize = self.target_model_runner.detokenize
        self.compute_stream = self.target_model_runner.compute_stream
        self.dtype = self.target_model_runner.dtype
        self.config = target_model.config
        self.draft_model_runner.wrap_target_model(self.target_model_runner)

    @property
    def main_mem_pool(self):
        return self.target_model_runner.block_memory_manager

    @property
    def draft_mem_pool(self):
        return self.draft_model_runner.block_memory_manager

    def init_memory_pools(self) -> None:
        self.target_model_runner.init_memory_pool()
        self.draft_model_runner.init_memory_pool(
            num_blocks=self.target_model_runner.block_memory_manager.num_blocks
        )

    def init_e2e_runner(self, model_func: Callable) -> None:
        self.e2e_runner = SpeculativeE2ECudaGraphRunner(
            self.draft_model_runner,
            self.target_model_runner,
            model_func,
        )

    def init_capture_cudagraph(self) -> None:
        if not self.server_args.disable_cuda_graph:
            self.e2e_runner.init_cuda_graph()

    def forward_e2e(self, scheduled_batch: ScheduleBatch) -> BatchResult:
        return self.forward_decode_e2e_post_process(
            scheduled_batch,
            *self.e2e_runner.forward(scheduled_batch),
        )

    def commit_previous(self, scheduled_batch: ScheduleBatch) -> None:
        """Advance draft state with the tokens accepted in the previous round."""
        raise NotImplementedError

    def proposal(self, scheduled_batch: ScheduleBatch) -> EagleVerifyInput:
        """Build algorithm-specific candidates in the shared verify format."""
        raise NotImplementedError

    def _adapt_verified_hidden_states(
        self, scheduled_batch: ScheduleBatch, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Adapt target states to the representation kept for the next step."""
        return hidden_states

    def verify(
        self, scheduled_batch: ScheduleBatch, proposal: EagleVerifyInput
    ) -> BatchResult:
        """Run the target model for a speculative proposal."""
        scheduled_batch.position_ids = proposal.positions
        scheduled_batch.input_ids = proposal.draft_token
        forward_batch = scheduled_batch.forward_batch
        forward_batch.forward_mode = ForwardMode.TARGET_VERIFY
        forward_batch.custom_mask = proposal.custom_mask

        verification = self.target_model_runner.forward(scheduled_batch)
        forward_batch.forward_mode = ForwardMode.DECODE
        hidden_states = torch.cat(verification.aux_hidden_states, dim=-1)
        proposal.hidden_states = self._adapt_verified_hidden_states(
            scheduled_batch, hidden_states
        )
        return verification

    def accept(
        self,
        scheduled_batch: ScheduleBatch,
        proposal: EagleVerifyInput,
        verification: BatchResult,
    ):
        """Select the path accepted by the target model."""
        accept_index, accept_length, predict = proposal.verify(
            scheduled_batch, verification, 1
        )
        return (
            proposal,
            verification.next_token_logits,
            accept_index,
            accept_length,
            predict,
        )

    def verify_propose(
        self, scheduled_batch: ScheduleBatch, proposal: EagleVerifyInput
    ):
        """Compatibility wrapper for Eagle's combined verify/accept call."""
        verification = self.verify(scheduled_batch, proposal)
        return self.accept(scheduled_batch, proposal, verification)

    def forward_decode_e2e(self, scheduled_batch: ScheduleBatch):
        self.commit_previous(scheduled_batch)
        proposal = self.proposal(scheduled_batch)
        verification = self.verify(scheduled_batch, proposal)
        return self.accept(scheduled_batch, proposal, verification)

    def forward_decode_e2e_post_process(
        self,
        scheduled_batch: ScheduleBatch,
        verify_input: EagleVerifyInput,
        next_token_logits: torch.Tensor,
        accept_index: torch.Tensor,
        accept_length: torch.Tensor,
        predict: torch.Tensor,
    ) -> BatchResult:
        """Publish verified tokens, hidden states, and KV locations."""
        result = verify_input.verify_post_process(
            scheduled_batch, accept_index, accept_length, predict
        )
        spec_info = scheduled_batch.spec_info
        spec_info.verified_id = result.verified_id
        spec_info.hidden_states = verify_input.hidden_states[result.accepted_indices]
        spec_info.accept_length = result.accept_length
        if not self.server_args.disable_overlap:
            width = self.server_args.speculative_num_draft_tokens
            for sequence in scheduled_batch:
                # This result owns the proposals; the request keeps its root.
                root = len(sequence.out_cache_loc) - width
                sequence.out_cache_loc[root + 1:] = (
                    array("q", [-1]) * self.server_args.speculative_num_steps
                )
        return BatchResult(
            next_token_ids=result.verified_id,
            next_token_logits=next_token_logits,
            aux_hidden_states=None,
            spec_info=spec_info,
        )

    def spec_postprocess(
        self,
        scheduled_batch: ScheduleBatch,
        batch_output: BatchResult,
        async_overlap: bool = False,
    ) -> ScheduleBatch:
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
            accepted_count = (1 + accept_length_cpu).sum()
            batch_output.next_token_ids = batch_output.next_token_ids[:accepted_count]
            spec_info.verified_id = spec_info.verified_id[:accepted_count]
            spec_info.hidden_states = spec_info.hidden_states[:accepted_count]

        if spec_info.out_cache_loc is not None:
            accepted_cache_locs = spec_info.out_cache_loc.cpu()
            accepted_cache_locs = [
                loc for loc in memoryview(accepted_cache_locs.numpy()) if loc != -1
            ]
            rejected_cache_locs = list(
                set(memoryview(out_cache_loc_cpu.numpy())) - set(accepted_cache_locs)
            )
            assert len(rejected_cache_locs) + len(accepted_cache_locs) == len(
                out_cache_loc_cpu
            )
            self.main_mem_pool.free_block(rejected_cache_locs)

            if not async_overlap:
                for idx, sequence in enumerate(scheduled_batch):
                    sequence.out_cache_loc = sequence.out_cache_loc[:-draft_token_num]
                    accept_len = accept_length_cpu[idx].item() + 1
                    sequence.out_cache_loc.extend(accepted_cache_locs[:accept_len])
                    accepted_cache_locs = accepted_cache_locs[accept_len:]
            else:
                width = self.server_args.speculative_num_steps + 1
                for idx, sequence in enumerate(scheduled_batch):
                    accept_len = accept_length_cpu[idx].item()
                    sequence_cache_locs = accepted_cache_locs[: 1 + accept_len]
                    accepted_cache_locs = accepted_cache_locs[1 + accept_len :]
                    if not sequence.status.is_active():
                        # The sequence owns the root; this in-flight result owns
                        # the accepted proposal slots that follow it.
                        self.main_mem_pool.free_block(sequence_cache_locs[1:])
                        continue
                    has_next = len(sequence.tokens) - sequence.last_generated_token_pos > width
                    start = sequence.last_generated_token_pos - 1
                    end = len(sequence.out_cache_loc) - (width if has_next else 0)
                    sequence.out_cache_loc[start:end] = array("q", sequence_cache_locs)

        if async_overlap:
            num_steps = self.server_args.speculative_num_steps
            for idx, sequence in enumerate(scheduled_batch):
                if not sequence.status.is_active():
                    continue
                accept_len = batch_output.spec_info.accept_length_cpu[idx].item()
                sequence.accept_length_cpu = accept_length_cpu[idx:idx + 1]
                if len(sequence.tokens) - sequence.last_generated_token_pos <= num_steps + 1:
                    continue
                offset = len(sequence.out_cache_loc_spec) - (num_steps - accept_len)
                sequence.out_cache_loc_spec, extra_locs = (
                    sequence.out_cache_loc_spec[:offset],
                    sequence.out_cache_loc_spec[offset:],
                )
                self.draft_mem_pool.free_block(extra_locs)

        if not async_overlap:
            for idx, sequence in enumerate(scheduled_batch):
                if not sequence.status.is_active():
                    continue
                sequence.accept_length = spec_info.accept_length[idx : idx + 1]
                sequence.accept_length_cpu = spec_info.accept_length.cpu()[
                    idx : idx + 1
                ]
                accept_length = accept_length_cpu[idx].item()
                end = last_verify_id_start + accept_length + 1
                sequence.verified_id = spec_info.verified_id[last_verify_id_start:end]
                sequence.hidden_states = spec_info.hidden_states[
                    last_verify_id_start:end
                ]
                last_verify_id_start = end

        if self.server_args.enable_debug:
            accepted = accept_length_cpu.clamp(min=0).sum().item()
            logger.info(
                "Speculative decoding: accepted %s tokens.",
                accepted,
            )
        return scheduled_batch


__all__ = ["SpeculativeWorker"]
