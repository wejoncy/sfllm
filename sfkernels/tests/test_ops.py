import torch
import unittest
import math
import torch.nn.functional as F

# Try to import the custom ops
try:
    import sf_kernel
    # Ensure ops are loaded
    ops = torch.ops.sfkernels
except ImportError:
    print("sf_kernel not installed or failed to load")
    ops = None

class TestOps(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        if ops is None:
            self.skipTest("sf_kernel ops not available")

    def test_silu_and_mul(self):
        from sfllm.layers.activations import SiluAndMul

        batch_size = 16
        seq_len = 128
        hidden_size = 4096 # Must be even
        
        x = torch.randn(batch_size, seq_len, hidden_size, device=self.device, dtype=torch.float16)
        out_ref = SiluAndMul().forward_native(x)
        
        out_cuda = torch.empty_like(out_ref)
        ops.silu_and_mul(out_cuda, x)
        
        torch.testing.assert_close(out_cuda, out_ref, rtol=1e-3, atol=1e-3)

    def test_gelu_and_mul(self):
        from sfllm.layers.activations import GeluAndMul

        batch_size = 16
        seq_len = 128
        hidden_size = 4096
        
        x = torch.randn(batch_size, seq_len, hidden_size, device=self.device, dtype=torch.float16)
        out_ref = GeluAndMul("none").forward_native(x)
        
        out_cuda = torch.empty_like(out_ref)
        ops.gelu_and_mul(out_cuda, x)
        
        torch.testing.assert_close(out_cuda, out_ref, rtol=1e-3, atol=1e-3)

    def test_gelu_tanh_and_mul(self):
        from sfllm.layers.activations import GeluAndMul

        batch_size = 16
        seq_len = 128
        hidden_size = 4096
        
        x = torch.randn(batch_size, seq_len, hidden_size, device=self.device, dtype=torch.float16)
        out_ref = GeluAndMul().forward_native(x)
        
        out_cuda = torch.empty_like(out_ref)
        ops.gelu_tanh_and_mul(out_cuda, x)
        
        torch.testing.assert_close(out_cuda, out_ref, rtol=1e-3, atol=1e-3)

    def test_apply_rope_pos_ids_cos_sin_cache(self):
        from sfllm.layers.rotary_embedding import RotaryEmbedding
        device = torch.device("cuda")
        dtype = torch.bfloat16
        
        batch_size = 2
        seq_len = 128
        num_heads = 4
        head_dim = 64  # Must be 64, 128, 256, 512
        rotary_dim = 64 # Must be 16, 32, 64, 128, 256
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
        # Allow slightly higher tolerance for bfloat16 and trigonometric ops
        # 0.0625 is typical for BF16 differences
        self.assertTrue(torch.allclose(q_rope, q_ref, atol=5e-2, rtol=1e-2), 
                        f"Q mismatch. Max diff: {(q_rope - q_ref).abs().max().item()}")
        self.assertTrue(torch.allclose(k_rope, k_ref, atol=5e-2, rtol=1e-2),
                        f"K mismatch. Max diff: {(k_rope - k_ref).abs().max().item()}")

if __name__ == '__main__':
    unittest.main()
