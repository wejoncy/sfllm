from enum import IntEnum, auto
import torch
import logging

logger = logging.getLogger(__name__)


class ForwardMode(IntEnum):
    # Extend a sequence. The KV cache of the beginning part of the sequence is already computed (e.g., system prompt).
    # It is also called "prefill" in common terminology.
    EXTEND = auto()
    # Decode one token.
    DECODE = auto()
    # Contains both EXTEND and DECODE when doing chunked prefill.
    MIXED = auto()
    # No sequence to forward. For data parallel attention, some workers will be IDLE if no sequence are allocated.
    IDLE = auto()

MAX_PROCESSED_TOKENS = 1024*200

class ForwardBatch:
    def __init__(self, config, dtype="auto"):
        # need to inilialize during prepare inputs
        self.max_extend_len = 0
        self.num_kv_splits_buffer = torch.zeros((MAX_PROCESSED_TOKENS,), dtype=torch.int32, device="cuda")+2
        self.num_kv_splits = None
        self.kv_indptr_buffer = torch.zeros((MAX_PROCESSED_TOKENS,), dtype=torch.int32, device="cuda")
        self.kv_indptr = None
        self.kv_indices_buffer = torch.zeros((MAX_PROCESSED_TOKENS,), dtype=torch.int64, device="cuda")
        self.kv_indices = None
        self.qo_indptr_buffer = torch.zeros((MAX_PROCESSED_TOKENS,), dtype=torch.int32, device="cuda")
        self.qo_indptr = None
        self.out_cache_loc = None
        self.custom_mask = None
        self.mask_indptr = None
        self.max_kv_splits = 16
        self.sampling_batch_info = None
        self.dtype = config.dtype if dtype == "auto" else getattr(torch, dtype)
        self.padded_token = 0

        self.attn_logits = torch.empty(
            (128, config.num_attention_heads, self.max_kv_splits, config.head_dim),
            dtype=torch.float32,
            device="cuda",
        )
        self.attn_lse = torch.empty(
            (128, config.num_attention_heads, self.max_kv_splits),
            dtype=torch.float32,
            device="cuda",
        )
        self.past_key_values = None
        self.forward_mode = ForwardMode.EXTEND


    # for compatibility only, not used in current implementation
    def get_seq_length(self):
        return 0

    def update(self, key_states, value_states, layer_idx):
        past_key, past_value = self.past_key_values[layer_idx]

        past_key[self.out_cache_loc, ...] = key_states[0]
        past_value[self.out_cache_loc, ...] = value_states[0]
        return key_states, value_states
