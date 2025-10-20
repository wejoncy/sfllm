from decode_attention import decode_attention_fwd, get_num_kv_splits_triton
import torch

def get_device_core_count(device_id: int = 0) -> int:
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        return torch.cuda.get_device_properties(device_id).multi_processor_count

    return 0

def _test_decode_attention_once(B, H_Q, H_KV, D):
    dtype = torch.bfloat16
    seq_len = 10  # This represents the number of tokens already in the sequence
    total_tokens = B * seq_len
    sm_scale = 1.0 / (D**0.5)
    max_kv_splits = 8
    num_kv_splits = torch.full((B,), 4, dtype=torch.int32, device="cuda")
    device_core_count = get_device_core_count(device_id=0)

    # q represents the new token being generated, one per batch
    q = torch.randn(B, H_Q, D, dtype=dtype, device="cuda")

    # k_buffer and v_buffer represent all previous tokens
    k_buffer = torch.randn(total_tokens, H_KV, D, dtype=dtype, device="cuda")
    v_buffer = torch.randn(total_tokens, H_KV, D, dtype=dtype, device="cuda")

    # o will have the same shape as q
    o = torch.zeros(B, H_Q, D, dtype=dtype, device="cuda")

    b_seq_len = torch.full((B,), seq_len, device="cuda")

    kv_indptr = torch.zeros((B + 1,), dtype=torch.int32, device="cuda")
    kv_indptr[1 : B + 1] = torch.cumsum(b_seq_len[:B], dim=0)
    kv_indices = torch.arange(total_tokens, device="cuda")

    attn_logits = torch.empty(
        (B, H_Q, max_kv_splits, D),
        dtype=torch.float32,
        device="cuda",
    )
    attn_lse = torch.empty(
        (B, H_Q, max_kv_splits),
        dtype=torch.float32,
        device="cuda",
    )
    # get_num_kv_splits_triton[(1,)](
    #         num_kv_splits,
    #         b_seq_len,
    #         B,
    #         H_Q // H_KV,
    #         H_Q,
    #         H_KV,
    #         max_kv_splits,
    #         device_core_count,
    #         MAX_NUM_SEQ=256,
    #     )
    decode_attention_fwd(
        q,
        k_buffer,
        v_buffer,
        o,
        kv_indptr,
        kv_indices,
        attn_logits,
        attn_lse,
        num_kv_splits,
        max_kv_splits,
        sm_scale,
    )
    return o

if __name__ == "__main__":
    # Example test case
    B = 4       # Batch size
    H_Q = 8     # Number of query heads
    H_KV = 4    # Number of key/value heads
    D = 64      # Head dimension

    print(_test_decode_attention_once(B, H_Q, H_KV, D))