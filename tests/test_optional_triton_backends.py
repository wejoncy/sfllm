"""Migration equivalence and execution without optional CUDA backend packages."""

import subprocess
import sys
import textwrap

import pytest
import sf_kernel  # Register the existing SFLLM activation operator.
import torch



cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")




# Use a fresh interpreter: sys.modules in the pytest process may already contain
# optional packages. Hide discovery too, matching an actual missing installation.
BLOCK_OPTIONAL_IMPORTS = """
import importlib.abc
import importlib.util
import sys
blocked = {'flashinfer', 'sgl_kernel', 'sglang'}
# Block every top-level module provided by the optional kernel distribution.
import importlib.metadata
for package, distributions in importlib.metadata.packages_distributions().items():
    if 'sglang-kernel' in distributions:
        blocked.add(package)
original_find_spec = importlib.util.find_spec
def find_spec(name, package=None):
    if name.split('.')[0] in blocked:
        return None
    return original_find_spec(name, package)
importlib.util.find_spec = find_spec
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split('.')[0]
        if root in blocked:
            raise ModuleNotFoundError(f'Blocked optional dependency: {fullname}', name=root)
sys.meta_path.insert(0, BlockOptional())
assert not any(name.split('.')[0] in blocked for name in sys.modules)
"""


@cuda
def test_moe_and_gdn_execute_without_optional_packages():
    code = BLOCK_OPTIONAL_IMPORTS + textwrap.dedent("""
        import torch
        from sfllm.kernels import gdn
        from sfllm.layers.moe import FusedMoE
        from sfllm.models.qwen3_5_moe import _cutlass_available

        assert not _cutlass_available()
        assert gdn.gdn_decode is None
        backend = gdn.GatedDeltaNetBackend('triton', 'triton')
        try:
            gdn.GatedDeltaNetBackend('flashinfer', 'triton')
        except ImportError as exc:
            assert '--linear-attn-backend triton' in str(exc)
        else:
            raise AssertionError('Missing FlashInfer was accepted')

        torch.manual_seed(62)
        layer = FusedMoE(128, 64, 8, 2).to(device='cuda', dtype=torch.bfloat16)
        with torch.no_grad():
            for p in layer.parameters():
                p.normal_(0, 0.03)
        x = torch.randn(17, 128, device='cuda', dtype=torch.bfloat16)
        result = layer(x)
        assert torch.isfinite(result).all() and result.abs().max() > 0

        # Packed unequal sequence lengths, checkpoint head dimensions, FP32 state.
        # Exercise the backend's public prefill and decode methods, not mocks.
        nk, nv, dim, slots = 2, 4, 128, 3
        conv_dim = (2 * nk + nv) * dim
        rand = lambda *s: torch.randn(*s, device='cuda', dtype=torch.bfloat16) * 0.1
        conv = torch.zeros(slots, conv_dim, 3, device='cuda', dtype=torch.bfloat16)
        states = torch.zeros(slots, nv, dim, dim, device='cuda', dtype=torch.float32)
        indices = torch.tensor([2, 0], device='cuda', dtype=torch.int32)
        starts = torch.tensor([0, 5, 17], device='cuda', dtype=torch.int32)
        options = dict(conv_weight=rand(conv_dim, 4), conv_states=conv,
                       ssm_states=states, state_indices=indices,
                       a_log=torch.zeros(nv, device='cuda'), dt_bias=rand(nv),
                       num_k_heads=nk, num_v_heads=nv, head_k_dim=dim, head_v_dim=dim)
        output = backend.prefill(rand(17, conv_dim), rand(17, nv), rand(17, nv),
                                 query_start_loc=starts, query_start_loc_i64=starts.long(), **options)
        assert output.shape == (17, nv, dim) and torch.isfinite(output).all()
        assert states.abs().max() > 0 and states[1].count_nonzero() == 0
        projected = rand(2, conv_dim + nv * dim)
        ba = rand(2, 2 * nv)
        backend.decode(projected, ba, **options)
        initial = states.clone()
        initial_conv = conv.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual, _ = backend.decode(projected, ba, **options)
        for _ in range(2):
            projected.normal_(0, 0.1)
            states.copy_(initial)
            conv.copy_(initial_conv)
            graph.replay()
            captured_state = states.clone()
            states.copy_(initial)
            conv.copy_(initial_conv)
            expected, _ = backend.decode(projected, ba, **options)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            torch.testing.assert_close(states, captured_state, atol=0, rtol=0)
        assert not any(name.split('.')[0] in blocked for name in sys.modules)
        print('MoE, GDN prefill/decode and CUDA Graph passed without optional packages')
    """)
    result = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
