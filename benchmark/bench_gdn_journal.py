"""GDN verify + commit with per-request acceptance changing every round.

Run from an installed SFLLM checkout, for example:
  python benchmark/bench_gdn_journal.py --batch 20 --layers 27 --steps 4 6 8 16

Uses FP32 states, synthetic projected BF16 inputs and CUDA graphs. Includes SSM slot
preparation, verification, a common GPU acceptance selector, and commit.
Excludes projection, convolution, sampling, and other model layers. This is
a kernel benchmark, not a serving throughput measurement.
"""

import argparse
import gc
import hashlib
import json
import random
import statistics
from pathlib import Path

import torch
import triton
import triton.language as tl
import flashinfer
from flashinfer.gdn_decode import gated_delta_rule_mtp

from sfllm.kernels.gdn import packed_gdn_decode, update_recurrent_state_indices
from sfllm.kernels.gdn_journal import packed_gdn_journal_verify, replay_gdn_journal


@triton.jit
def _select_accepts(table, cursor, accepted, batch: tl.constexpr,
                    rounds: tl.constexpr, block: tl.constexpr):
    row = tl.arange(0, block)
    current = tl.load(cursor)
    value = tl.load(table + current * batch + row, row < batch, other=-1)
    tl.store(accepted + row, value, row < batch)
    tl.store(cursor, (current + 1) % rounds)


class Case:
    def __init__(self, batch, layers, steps, rounds):
        dtype = torch.float32
        self.batch, self.layers, self.steps, self.rounds = batch, layers, steps, rounds
        h, hv, k, v = 16, 32, 128, 128
        self.h, self.hv, self.k, self.v = h, hv, k, v
        torch.manual_seed(16092026)
        self.x = torch.randn(layers, batch * steps, 2 * h * k + hv * v,
                             device="cuda", dtype=torch.bfloat16)
        self.a = torch.randn(layers, batch * steps, hv, device="cuda", dtype=torch.bfloat16)
        self.b = torch.randn_like(self.a)
        self.a_log = torch.randn(layers, hv, device="cuda") * .2
        self.bias = torch.randn(layers, hv, device="cuda", dtype=torch.bfloat16)
        self.requests = torch.arange(1, batch + 1, device="cuda", dtype=torch.int32)
        self.base = (self.requests - 1) * (steps + 1) + 1
        self.current = torch.zeros(batch + 1, device="cuda", dtype=torch.int32)
        self.read = torch.empty_like(self.requests)
        self.write = torch.empty(batch, steps, device="cuda", dtype=torch.int32)
        self.state = torch.zeros(layers, batch * (steps + 1) + 1, hv, v, k,
                                 device="cuda", dtype=dtype)
        self.jstate = torch.zeros(layers, batch + 1, hv, v, k, device="cuda", dtype=dtype)
        self.initial = (torch.randn_like(self.jstate[:, 1:]) * .1).contiguous()
        self.journal = (
            torch.empty(layers, batch, hv, steps, v, device="cuda"),
            torch.empty(layers, batch, hv, steps, k, device="cuda"),
            torch.empty(layers, batch, hv, steps, device="cuda"),
        )
        self.layer_journals = [tuple(t[layer] for t in self.journal) for layer in range(layers)]
        self.cursor = torch.zeros((), device="cuda", dtype=torch.int32)
        self.accepted = torch.empty_like(self.requests)
        self.table = torch.empty(rounds, batch, device="cuda", dtype=torch.int32)
        self.outputs = {}
        self.reset()

    def reset(self):
        self.state[:, self.base.long()] = self.initial
        self.jstate[:, 1:].copy_(self.initial)
        self.current[1:].copy_(self.base)
        self.cursor.zero_()

    def prepare(self):
        update_recurrent_state_indices(self.current, self.requests, self.read, self.write)

    def verify(self, backend):
        outputs = []
        for layer in range(self.layers):
            if backend == "journal":
                out = packed_gdn_journal_verify(
                    self.x[layer], self.a[layer], self.b[layer], self.a_log[layer],
                    self.bias[layer], self.jstate[layer], self.requests,
                    self.h, self.layer_journals[layer],
                )
            elif backend == "triton":
                out = packed_gdn_decode(
                    self.x[layer], self.a[layer], self.b[layer], self.a_log[layer],
                    self.bias[layer], self.state[layer], self.read, self.h, self.write,
                )
            else:
                q, k, v = self.x[layer].split((self.h * self.k, self.h * self.k,
                                               self.hv * self.v), dim=-1)
                out, _ = gated_delta_rule_mtp(
                    q=q.view(self.batch, self.steps, self.h, self.k),
                    k=k.view(self.batch, self.steps, self.h, self.k),
                    v=v.view(self.batch, self.steps, self.hv, self.v),
                    output=v.new_empty((self.batch, self.steps, self.hv, self.v)),
                    A_log=self.a_log[layer], a=self.a[layer].view(self.batch, self.steps, self.hv),
                    dt_bias=self.bias[layer], b=self.b[layer].view(self.batch, self.steps, self.hv),
                    use_qk_l2norm=True, initial_state=self.state[layer],
                    initial_state_indices=self.read, ssm_state_indices=self.write,
                    disable_state_update=False,
                )
                out = out.view(-1, self.hv, self.v)
            outputs.append(out)
        self.outputs[backend] = outputs

    def select(self):
        _select_accepts[(1,)](self.table, self.cursor, self.accepted, self.batch,
                              self.rounds, triton.next_power_of_2(self.batch), num_warps=1)

    def commit(self, backend):
        if backend == "journal":
            replay_gdn_journal(self.jstate, self.journal, self.requests, self.accepted)
        else:
            update_recurrent_state_indices(self.current, self.requests, self.read,
                                            self.write, accepted_steps=self.accepted)

    def round(self, backend):
        if backend != "journal":
            self.prepare()
        self.verify(backend)
        self.select()
        self.commit(backend)

    def check(self):
        self.reset()
        errors = {"output_max_abs": 0.0, "state_max_abs": 0.0}
        for round_id in range(self.rounds):
            self.prepare()
            self.verify("triton")
            self.verify("journal")
            self.select()
            self.commit("triton")
            self.commit("journal")
            for expected, actual in zip(self.outputs["triton"], self.outputs["journal"]):
                torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-6,
                                           msg=f"Output differs at round {round_id}")
                errors["output_max_abs"] = max(
                    errors["output_max_abs"], (actual.float() - expected.float()).abs().max().item(),
                )
            expected = self.state[:, self.current[self.requests.long()].long()]
            actual = self.jstate[:, 1:]
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-7,
                                       msg=f"Committed state differs at round {round_id}")
            errors["state_max_abs"] = max(
                errors["state_max_abs"], (actual - expected).abs().max().item(),
            )
        self.reset()
        return errors


def acceptance_trace(profile, rounds, batch, steps):
    weights = torch.ones(steps)
    if profile == "short":
        weights = torch.exp(-torch.arange(steps).float() * .6)
    elif profile == "long":
        weights = torch.exp(-torch.arange(steps - 1, -1, -1).float() * .6)
    elif profile == "bimodal":
        weights.zero_()
        weights[0] = weights[-1] = 1
    rng = torch.Generator().manual_seed(16092026)
    return torch.multinomial(weights, rounds * batch, replacement=True,
                             generator=rng).view(rounds, batch).to(torch.int32)


def measure(case, trials):
    backends = ["triton", "journal"]
    # FlashInfer's snapshot-index MTP API requires at least two steps.
    if case.steps >= 2:
        backends.insert(1, "flashinfer")
    graphs = {}
    for backend in backends:
        case.reset()
        for _ in range(2):
            case.round(backend)
        torch.cuda.synchronize()
        case.reset()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(case.rounds):
                case.round(backend)
        graphs[backend] = graph
        graph.replay()
    torch.cuda.synchronize()
    samples = {name: [] for name in backends}
    rng = random.Random(29)
    for _ in range(trials):
        rng.shuffle(backends)
        for backend in backends:
            case.reset()
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            graphs[backend].replay()
            end.record()
            end.synchronize()
            samples[backend].append(start.elapsed_time(end) * 1000 / case.rounds)
    return {name: {"median_us": statistics.median(values), "samples_us": values}
            for name, values in samples.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=20)
    parser.add_argument("--layers", type=int, default=27)
    parser.add_argument("--steps", type=int, nargs="+", default=[4, 6, 8, 16])
    parser.add_argument("--profiles", nargs="+", choices=["short", "uniform", "long", "bimodal"],
                        default=["short", "uniform", "long", "bimodal"])
    parser.add_argument("--rounds", type=int, default=32)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("gdn_journal.json"))
    args = parser.parse_args()
    if min(args.batch, args.layers, args.rounds, args.trials, *args.steps) <= 0:
        parser.error("batch, layers, rounds, trials and block lengths must be positive")
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    result = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  triton=triton.__version__, flashinfer=flashinfer.__version__,
                  batch=args.batch, layers=args.layers, rounds=args.rounds, trials=args.trials,
                  input_dtype="bfloat16", state_dtype="float32", journal_dtype="float32",
                  key_heads=16, value_heads=32, key_dim=128, value_dim=128,
                  acceptance_seed=16092026, cases=[])
    root = Path(__file__).resolve().parents[1]
    result["sources"] = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ("benchmark/bench_gdn_journal.py", "python/sfllm/kernels/gdn.py",
                     "python/sfllm/kernels/gdn_journal.py")
    }
    for steps in args.steps:
        case = Case(args.batch, args.layers, steps, args.rounds)
        for profile in args.profiles:
            trace = acceptance_trace(profile, args.rounds, args.batch, steps)
            case.table.copy_(trace)
            errors = case.check()
            record = dict(steps=steps, profile=profile,
                          numerical_errors=errors,
                          replay_counts=(trace + 1).tolist(), timings=measure(case, args.trials))
            result["cases"].append(record)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps({k: v for k, v in record.items() if k != "replay_counts"}), flush=True)
        del case
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
