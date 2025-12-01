import torch
import pytest
import sf_kernel
from sfllm.layers.layernorm import RMSNorm

# Skip all tests in this module if CUDA is not available
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")

@pytest.mark.parametrize("hidden_size", [1024, 4096])
@pytest.mark.parametrize("num_tokens", [1, 128])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_rmsnorm_standard(hidden_size, num_tokens, dtype):
    torch.manual_seed(0)
    epsilon = 1e-6
    device = "cuda"

    input_tensor = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    weight = torch.randn(hidden_size, dtype=dtype, device=device)
    
    # Reference implementation
    ref_rmsnorm = RMSNorm(hidden_size, eps=epsilon).to(device, dtype=dtype)
    ref_rmsnorm.weight.data.copy_(weight)
    ref_out = ref_rmsnorm.forward_native(input_tensor)

    # Custom implementation
    out = torch.empty_like(input_tensor)
    sf_kernel.rmsnorm(out, input_tensor, weight, epsilon)

    max_diff = (out - ref_out).abs().max().item()
    print(f"Max diff: {max_diff}")

    # Use torch.testing.assert_close for robust comparison
    torch.testing.assert_close(out, ref_out, atol=1e-2, rtol=1e-2)

@pytest.mark.parametrize("hidden_size", [1024, 4096])
@pytest.mark.parametrize("num_tokens", [1, 128])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_rmsnorm_fused(hidden_size, num_tokens, dtype):
    torch.manual_seed(0)
    epsilon = 1e-6
    device = "cuda"

    input_tensor = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    residual = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    weight = torch.randn(hidden_size, dtype=dtype, device=device)
    
    # Reference implementation
    ref_rmsnorm = RMSNorm(hidden_size, eps=epsilon).to(device, dtype=dtype)
    ref_rmsnorm.weight.data.copy_(weight)
    
    # Clone residual for reference to ensure clean state
    residual_ref = residual.clone()
    ref_out_fused, residual_out_ref = ref_rmsnorm.forward_native(input_tensor, residual_ref)

    # Custom implementation
    out_fused = torch.empty_like(input_tensor)
    # Clone residual for custom kernel to avoid modifying the original tensor used for setup
    residual_custom = residual.clone()
    
    sf_kernel.rmsnorm(out_fused, input_tensor, weight, epsilon, residual_custom)

    max_diff = (out_fused - ref_out_fused).abs().max().item()
    print(f"Max diff fused: {max_diff}")
    
    max_diff_residual = (residual_custom - residual_out_ref).abs().max().item()
    print(f"Max diff residual: {max_diff_residual}")

    torch.testing.assert_close(out_fused, ref_out_fused, atol=1e-2, rtol=1e-2)
    # Verify residual update
    torch.testing.assert_close(residual_custom, residual_out_ref, atol=1e-2, rtol=1e-2)

