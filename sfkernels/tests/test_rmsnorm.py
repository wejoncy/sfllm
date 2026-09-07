import torch
import pytest
import sf_kernel
from sfllm.layers.layernorm import GemmaRMSNorm, RMSNorm
import sfllm.layers.layernorm as layernorm

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


@pytest.mark.parametrize("gemma_style", [False, True])
@pytest.mark.parametrize("with_residual", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rmsnorm_strided_qkv_view(with_residual, dtype, gemma_style):
    torch.manual_seed(0)
    num_tokens, num_heads, head_dim = 7, 32, 128
    q_size = num_heads * head_dim
    qkv = torch.randn(
        num_tokens, q_size + 2048, dtype=dtype, device="cuda"
    )
    input_tensor = qkv[:, :q_size].view(num_tokens, num_heads, head_dim)
    assert not input_tensor.is_contiguous()
    weight = torch.randn(head_dim, dtype=dtype, device="cuda")
    output = torch.empty_like(input_tensor)
    residual = None
    residual_ref = None
    if with_residual:
        residual = torch.randn_like(input_tensor)
        residual_ref = residual.clone()

    values = input_tensor.float()
    if residual_ref is not None:
        values = values + residual_ref.float()
    expected = (
        values
        * torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + 1e-6)
        * (weight.float() + float(gemma_style))
    ).to(dtype)

    sf_kernel.rmsnorm(output, input_tensor, weight, 1e-6, residual, gemma_style=gemma_style)

    torch.testing.assert_close(output, expected, atol=1e-2, rtol=1e-2)
    if residual is not None:
        torch.testing.assert_close(
            residual, values.to(dtype), atol=1e-2, rtol=1e-2
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [
    (24, 32, 128), (1, 1, 1), (3, 7, 127), (3, 3, 129),
    (3, 3, 1024), (3, 3, 1025), (3, 2, 4096),
    (0, 3, 128), (3, 0, 128),
])
def test_gated_rmsnorm_strided(shape, dtype):
    torch.manual_seed(0)
    storage = torch.randn(*shape[:-1], 2 * shape[-1] + 1, device="cuda", dtype=dtype)
    values = storage[..., 1:shape[-1] + 1]
    gate = torch.randn_like(storage)[..., 1:shape[-1] + 1]
    weight = torch.randn(shape[-1], device="cuda", dtype=torch.float32)
    output = torch.empty(shape, device="cuda", dtype=dtype)
    x, z = values.float(), gate.float()
    expected = (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
                * weight * z * z.sigmoid()).to(dtype)

    sf_kernel.gated_rmsnorm(output, values, gate, weight, 1e-6)

    torch.testing.assert_close(output, expected, atol=1e-3, rtol=1e-2)
