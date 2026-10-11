from enum import IntEnum, auto
import torch
import logging

from sfllm.spec_decoding.spec_common import SpecInput

logger = logging.getLogger(__name__)


class ForwardMode(IntEnum):
    # Extend a sequence. The KV cache of the beginning part of the sequence is already computed (e.g., system prompt).
    # It is also called "prefill" in common terminology.
    EXTEND = auto()
    TARGET_VERIFY = auto()
    DRAFT_EXTEND = auto()
    # Decode one token.
    DECODE = auto()
    # Contains both EXTEND and DECODE when doing chunked prefill.
    MIXED = auto()
    # No sequence to forward. For data parallel attention, some workers will be IDLE if no sequence are allocated.
    IDLE = auto()

class PrefillOutputPlan:
    """Hidden-row selection and checkpoint writes for one prefill.

    Hidden-row indices address computed rows followed by cached_hidden rows.
    output_indices=None selects request ends; checkpoint_indices=None saves
    those same rows. output_mapping restores request order after the LM head.
    Buffers are borrowed from their owners, including during graph capture.
    """

    def __init__(self, *, output_indices=None, output_mapping=None,
                 cached_hidden=None, checkpoint_indices=None, checkpoint_buffer=None):
        self.output_indices = output_indices
        self.output_mapping = output_mapping
        self.cached_hidden = cached_hidden
        self.checkpoint_indices = checkpoint_indices
        self.checkpoint_buffer = checkpoint_buffer


class ForwardBatch:
    def __init__(self, mem_pool):
        # need to inilialize during prepare inputs
        self.max_extend_len = 0
        self.num_kv_splits = None
        self.seq_lens = None
        self.kv_indptr = None
        self.kv_indices = None
        self.kv_indices_extend = None # draft model need both extend and decode kv indices in decode step
        self.qo_indptr = None
        self.out_cache_loc = None
        self.custom_mask = None
        self.mask_indptr = None
        # Optional caller-owned storage shared by mutually exclusive CUDA graphs.
        self.logits_output_buffer = None
        self.prefill_output: PrefillOutputPlan | None = None
        self.max_kv_splits = 16
        self.sampling_batch_info = None
        self.padded_token = 0

        self.position_ids_extend = None

        self.spec_info:SpecInput = None
        self.cache_groups = None

        self.past_key_values = mem_pool.kv_buffers if mem_pool is not None else None
        self.forward_mode:ForwardMode = ForwardMode.EXTEND

    def is_decode(self):
        return self.forward_mode == ForwardMode.DECODE


    # for compatibility only, not used in current implementation
    def get_seq_length(self):
        return 0

    def update(self, key_states, value_states, layer_idx):
        if self.past_key_values is None:
            return key_states, value_states
        past_key, past_value = self.past_key_values[layer_idx]

        past_key[self.out_cache_loc, ...] = key_states
        past_value[self.out_cache_loc, ...] = value_states
        return key_states, value_states

    def cache_locations(self, layer_id):
        if self.cache_groups is None:
            return self.out_cache_loc
        return self.cache_groups[layer_id].out_cache_loc
