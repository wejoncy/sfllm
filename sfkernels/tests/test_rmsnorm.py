import torch
import sf_kernel
from sfllm.layers.layernorm import RMSNorm

def test_rmsnorm():
    if not torch.cuda.is_available():
        print("Skipping test: CUDA not available")
        return

    torch.manual_seed(0)
    hidden_size = 4096
    num_tokens = 128
    epsilon = 1e-6
    dtype = torch.float16
    device = "cuda"

    print("Testing standard RMSNorm...")
    input = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    weight = torch.randn(hidden_size, dtype=dtype, device=device)
    
    # Reference implementation
    ref_rmsnorm = RMSNorm(hidden_size, eps=epsilon).to(device, dtype=dtype)
    ref_rmsnorm.weight.data.copy_(weight)
    ref_out = ref_rmsnorm.forward_native(input)

    # Custom implementation
    out = torch.empty_like(input)
    sf_kernel.rmsnorm(out, input, weight, epsilon)

    max_diff = (out - ref_out).abs().max().item()
    print(f"Max diff: {max_diff}")
    assert torch.allclose(out, ref_out, atol=1e-2, rtol=1e-2)
    print("Standard RMSNorm passed!")

    print("\nTesting Fused Add-RMSNorm...")
    input_2 = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    
    # Reference: RMSNorm(input + input_2)
    ref_out_fused, residual_out = ref_rmsnorm.forward_native(input, input_2)

    # Custom implementation
    out_fused = torch.empty_like(input)
    sf_kernel.rmsnorm(out_fused, input, weight, epsilon, input_2)

    max_diff_fused = (out_fused - ref_out_fused).abs().max().item()
    print(f"Max diff fused: {max_diff_fused}")
    assert torch.allclose(out_fused, ref_out_fused, atol=1e-2, rtol=1e-2)
    assert torch.allclose(input_2, residual_out, atol=1e-4, rtol=1e-4)
    print("Fused Add-RMSNorm passed!")

if __name__ == "__main__":
    test_rmsnorm()
