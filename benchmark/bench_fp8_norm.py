"""Compare FP8 norm kernels with an exported baseline module under CUDA Graphs.

  git show HEAD^:python/sfllm/layers/quantization/fp8_kernel.py > /tmp/fp8-baseline.py
  python benchmark/bench_fp8_norm.py --baseline /tmp/fp8-baseline.py

Times eight independent calls per graph, not a model forward or serving QPS.
The auxiliary case includes the standalone residual addition used when capturing auxiliary hidden states.
"""

import argparse
import importlib.util
import json
import statistics

import torch

from sfllm.layers.quantization import fp8_kernel as candidate


def capture(module, case, ops, optimized):
    def run(x, weight, residual, gate):
        if case == "gate":
            return module.rmsnorm_silu_gate_quant_fp8(x, weight, gate)
        aux = None
        kwargs = {}
        if case == "aux":
            if optimized:
                aux = torch.empty_like(x)
                kwargs["residual_output"] = aux
            else:
                aux = x + residual
        output = module.gemma_rmsnorm_quant_fp8(x, weight, residual, **kwargs)
        return (*output, residual, aux) if aux is not None else (*output, residual)

    for _ in range(8):
        for args in ops:
            run(*args)
    # Residuals mutate in place. Restore them before capturing either variant.
    for _, _, residual, _ in ops:
        residual.zero_()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = [run(*args) for args in ops]
    return graph, outputs


def measure(graph):
    for _ in range(8):
        graph.replay()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(100):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / (100 * 8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="Exported baseline fp8_kernel.py")
    parser.add_argument("--rows", nargs="+", type=int, default=[1, 32, 64, 128, 192])
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("fp8_norm_baseline", args.baseline)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    torch.manual_seed(173)
    print(json.dumps({"gpu": torch.cuda.get_device_name(), "scope": "norm kernels only"}))
    for rows in args.rows:
        for case in ("gate", "residual", "aux"):
            shape = (rows, 32, 128) if case == "gate" else (rows, 2560)
            ops = []
            for _ in range(8):
                x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
                weight = torch.randn(shape[-1], device="cuda", dtype=x.dtype)
                ops.append((x, weight, torch.zeros_like(x), torch.randn_like(x)))
            old_ops = [(x, w, r.clone(), g) for x, w, r, g in ops]
            graphs = [capture(baseline, case, old_ops, False),
                      capture(candidate, case, ops, True)]
            for actual, expected in zip(graphs[1][1], graphs[0][1]):
                for a, b in zip(actual, expected):
                    torch.testing.assert_close(a.float(), b.float(), rtol=1e-5, atol=1e-6)
            samples = [[], []]
            for rep in range(5):
                for index in (rep % 2, 1 - rep % 2):
                    # Keep repeated residual updates finite and identical.
                    for _, _, residual, _ in (old_ops, ops)[index]:
                        residual.zero_()
                    samples[index].append(measure(graphs[index][0]))
            old, new = (statistics.median(s) for s in samples)
            print(json.dumps({"rows": rows, "case": case, "baseline_us": old,
                              "candidate_us": new, "reduction_pct": 100 * (1 - new / old)}),
                  flush=True)


if __name__ == "__main__":
    main()
