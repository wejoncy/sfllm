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

class ForwardMetaData:
    def __init__(self, config):
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

        self.past_key_values = self.create_past_kv(config)
        self.forward_mode = ForwardMode.EXTEND

    def create_past_kv(self, config, max_length=1024000):
        past_key_values = []
        dim = (
            getattr(config, "head_dim", None)
            or config.hidden_size // config.num_attention_heads
        )
        n_heads = config.num_key_value_heads
        free, total = torch.cuda.mem_get_info("cuda:0")
        one_token_size = n_heads * dim * 2 * 2  # key + value, float16
        max_length = min(max_length, int(free*0.85) // one_token_size // config.num_hidden_layers)
        logger.info(
            f"GPU memory free: {free / (1024**3):.2f} GB, total: {total / (1024**3):.2f} GB"
            f", max kv length per layer: {max_length}"
        )
        for _ in range(config.num_hidden_layers):
            past_key_values.append(
                (
                    torch.zeros(max_length, n_heads, dim, dtype=config.dtype).cuda(),
                    torch.zeros(max_length, n_heads, dim, dtype=config.dtype).cuda(),
                )
            )
        return past_key_values

    # for compatibility only, not used in current implementation
    def get_seq_length(self):
        return 0

    def update(self, key_states, value_states, layer_idx):
        past_key, past_value = self.past_key_values[layer_idx]

        past_key[self.out_cache_loc, ...] = key_states[0]
        past_value[self.out_cache_loc, ...] = value_states[0]
        return key_states, value_states
