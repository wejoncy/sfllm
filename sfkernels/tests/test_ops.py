import torch
import pytest
from sfllm.layers.activations import SiluAndMul, GeluAndMul
from sfllm.layers.rotary_embedding import RotaryEmbedding

# Skip all tests in this module if CUDA is not available
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")

import sf_kernel
ops = torch.ops.sfkernels


@pytest.mark.skipif(ops is None, reason="sf_kernel ops not available")
@pytest.mark.parametrize("batch_size", [1, 16])
@pytest.mark.parametrize("seq_len", [128])
@pytest.mark.parametrize("hidden_size", [4096])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_silu_and_mul(batch_size, seq_len, hidden_size, dtype):
    device = "cuda"
    torch.manual_seed(0)
    
    x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)
    out_ref = SiluAndMul().forward_native(x)
    
    out_cuda = torch.empty_like(out_ref)
    ops.silu_and_mul(out_cuda, x)
    
    max_diff = (out_cuda - out_ref).abs().max().item()
    print(f"Max diff: {max_diff}")

    torch.testing.assert_close(out_cuda, out_ref, rtol=1e-3, atol=1e-3)

@pytest.mark.skipif(ops is None, reason="sf_kernel ops not available")
@pytest.mark.parametrize("batch_size", [1, 16])
@pytest.mark.parametrize("seq_len", [128])
@pytest.mark.parametrize("hidden_size", [4096])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_gelu_and_mul(batch_size, seq_len, hidden_size, dtype):
    device = "cuda"
    torch.manual_seed(0)
    
    x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)
    out_ref = GeluAndMul("none").forward_native(x)
    
    out_cuda = torch.empty_like(out_ref)
    ops.gelu_and_mul(out_cuda, x)
    
    max_diff = (out_cuda - out_ref).abs().max().item()
    print(f"Max diff: {max_diff}")

    torch.testing.assert_close(out_cuda, out_ref, rtol=1e-3, atol=1e-3)

@pytest.mark.skipif(ops is None, reason="sf_kernel ops not available")
@pytest.mark.parametrize("batch_size", [1, 16])
@pytest.mark.parametrize("seq_len", [128])
@pytest.mark.parametrize("hidden_size", [4096])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_gelu_tanh_and_mul(batch_size, seq_len, hidden_size, dtype):
    device = "cuda"
    torch.manual_seed(0)
    
    x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)
    out_ref = GeluAndMul().forward_native(x)
    
    out_cuda = torch.empty_like(out_ref)
    ops.gelu_tanh_and_mul(out_cuda, x)
    
    max_diff = (out_cuda - out_ref).abs().max().item()
    print(f"Max diff: {max_diff}")

    torch.testing.assert_close(out_cuda, out_ref, rtol=1e-3, atol=1e-3)

@pytest.mark.skipif(ops is None, reason="sf_kernel ops not available")
@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("seq_len", [128])
@pytest.mark.parametrize("num_heads", [4])
@pytest.mark.parametrize("head_dim", [64])
@pytest.mark.parametrize("rotary_dim", [64])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_apply_rope_pos_ids_cos_sin_cache(batch_size, seq_len, num_heads, head_dim, rotary_dim, dtype):
    device = "cuda"
    torch.manual_seed(0)
    max_position_embeddings = 4096
    
    # Initialize RotaryEmbedding
    # is_neox_style=True matches interleave=False in the kernel
    rope_ops = RotaryEmbedding(head_dim, rotary_dim, max_position_embeddings,
                               base=10000, is_neox_style=True, dtype=dtype)
    rope_ops.to(device)
    
    # Create inputs
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype, device=device)
    
    # pos_ids must be int64 because the C++ wrapper casts to int64_t*
    pos_ids = torch.arange(seq_len, dtype=torch.int64, device=device).unsqueeze(0).expand(batch_size, seq_len).contiguous()
    
    # Get cos_sin_cache from rope_ops
    cos_sin_cache = rope_ops.cos_sin_cache
    
    # Prepare for kernel (flatten to 3D: [nnz, heads, dim])
    q_flat = q.view(batch_size * seq_len, num_heads, head_dim)
    k_flat = k.view(batch_size * seq_len, num_heads, head_dim)
    q_rope_flat = torch.empty_like(q_flat)
    k_rope_flat = torch.empty_like(k_flat)
    pos_ids_flat = pos_ids.view(-1)

    # Run kernel
    # apply_rope_pos_ids_cos_sin_cache(q, k, q_rope, k_rope, cos_sin_cache, pos_ids, interleave, enable_pdl, ...)
    ops.apply_rope_pos_ids_cos_sin_cache(
        q_flat, k_flat, q_rope_flat, k_rope_flat, cos_sin_cache, pos_ids_flat, 
        False, # interleave=False (matches is_neox_style=True)
        False, # enable_pdl
        None, None, None, None
    )
    
    q_rope = q_rope_flat.view(batch_size, seq_len, num_heads, head_dim)
    k_rope = k_rope_flat.view(batch_size, seq_len, num_heads, head_dim)
    
    # Run reference using rope_ops
    q_ref, k_ref = rope_ops.forward_native(pos_ids, q, k)
   
    # Compare
    max_diff_q = (q_rope - q_ref).abs().max().item()
    max_diff_k = (k_rope - k_ref).abs().max().item()
    print(f"Max diff Q: {max_diff_q}")
    print(f"Max diff K: {max_diff_k}")

    # Allow slightly higher tolerance for bfloat16 and trigonometric ops
    torch.testing.assert_close(q_rope, q_ref, atol=5e-2, rtol=1e-2)
    torch.testing.assert_close(k_rope, k_ref, atol=5e-2, rtol=1e-2)
