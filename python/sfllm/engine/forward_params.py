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

MAX_PROCESSED_TOKENS = 1024*200

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
        self.max_kv_splits = 16
        self.sampling_batch_info = None
        self.padded_token = 0
        self.padded_token_extend = 0

        self.position_ids_extend = None

        self.spec_info:SpecInput = None

        self.past_key_values = mem_pool.kv_buffers if mem_pool is not None else None
        self.forward_mode = ForwardMode.EXTEND

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
