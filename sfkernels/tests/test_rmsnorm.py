import torch
import pytest
import sf_kernel
from sfllm.layers.layernorm import GemmaRMSNorm, RMSNorm
import sfllm.layers.layernorm as layernorm
from sfllm.layers.quantization.fp8_kernel import gemma_rmsnorm_quant_fp8

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
    (24, 32, 128), (3, 48, 128), (1, 1, 1), (3, 7, 127), (3, 3, 129),
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


@pytest.mark.parametrize("rows", [1, 64, 192, 193])
@pytest.mark.parametrize("input_scale", [None, 0.02])
def test_fp8_norm_residual_capture_cuda_graph(rows, input_scale):
    # Capture the sum before norm updates the residual, then replay with new data.
    x = torch.randn(rows, 2560, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    weight = torch.randn(2560, device="cuda", dtype=x.dtype)
    aux = torch.empty_like(x)
    scale = None if input_scale is None else torch.tensor(input_scale, device="cuda")

    def run():
        return gemma_rmsnorm_quant_fp8(
            x, weight, residual, output_dtypes=(torch.float8_e4m3fn, None),
            input_scales=(scale, None), residual_output=aux,
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run()
    for _ in range(3):
        x.normal_()
        residual.normal_()
        expected_sum = x + residual
        expected = gemma_rmsnorm_quant_fp8(
            x, weight, residual.clone(), output_dtypes=(torch.float8_e4m3fn, None),
            input_scales=(scale, None),
        )
        graph.replay()
        torch.testing.assert_close(aux, expected_sum, rtol=0, atol=0)
        torch.testing.assert_close(residual, expected_sum, rtol=0, atol=0)
        torch.testing.assert_close(actual[0][0].float(), expected[0][0].float(), rtol=0, atol=0)
        torch.testing.assert_close(actual[0][1], expected[0][1], rtol=0, atol=0)
        torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)


def test_fp8_norm_waits_for_dependent_residual_and_gate():
    if torch.version.hip is not None or torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("PDL requires SM90 or newer")
    import triton
    import triton.language as tl
    from sfllm.layers.quantization.fp8_kernel import _norm_activation_fp8_kernel

    @triton.jit
    def producer(S, X, R, G, N: tl.constexpr, B: tl.constexpr):
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
        idx = tl.program_id(0) * B + tl.arange(0, B)
        value = tl.load(S + idx, idx < N, 0)
        tl.store(X + idx, value * 2, idx < N)
        tl.store(R + idx, value * 3, idx < N)
        tl.store(G + idx, value * 4, idx < N)

    rows, width = 32, 256
    src = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(width, device="cuda", dtype=src.dtype)
    buffers = [(torch.empty_like(src), torch.empty_like(src), torch.empty_like(src))
               for _ in range(2)]
    outputs = [torch.empty_like(src, dtype=torch.float8_e4m3fn) for _ in range(2)]
    scales = [torch.empty(rows, 1, device="cuda") for _ in range(2)]
    strides = (width, width, 1)

    def run(index):
        x, residual, gate = buffers[index]
        pdl = bool(index)
        producer[(triton.cdiv(src.numel(), 256),)](
            src, x, residual, gate, src.numel(), 256, launch_pdl=pdl,
        )
        _norm_activation_fp8_kernel[(rows,)](
            x, weight, residual, gate, (outputs[index],), (scales[index],),
            strides, strides, strides, 1, 1, width, 1e-6, False, False, 1, width,
            (False,), 448., ENABLE_PDL=pdl, num_warps=4, launch_pdl=pdl,
            enable_fp_fusion=False,
        )

    for index in (0, 1):
        run(index)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for index in (0, 1):
            run(index)
    for _ in range(3):
        src.normal_()
        graph.replay()
        torch.testing.assert_close(outputs[0].float(), outputs[1].float(), rtol=0, atol=0)
        torch.testing.assert_close(scales[0], scales[1], rtol=0, atol=0)
        torch.testing.assert_close(buffers[0][1], buffers[1][1], rtol=0, atol=0)
