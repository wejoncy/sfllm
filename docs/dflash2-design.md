# DFlash2 design

## Boundary

DFlash2 is a draft proposer plugged into SFLLM's existing Eagle3 speculative
pipeline. Algorithm selection happens once in `InferenceEngine`. From that
point onward the scheduler, overlap loop, CUDA Graph runner, verify,
postprocess, and KV lifetime logic are shared.

| Shared with Eagle3 | DFlash2-specific |
| --- | --- |
| `ScheduleBatch` speculative metadata | DFlash2 checkpoint/model definition |
| `EagleSpecInput` and `EagleVerifyInput` | target-hidden to draft-KV projection |
| target verify and greedy acceptance | fixed masked-block draft forward |
| accepted-token/KV packing | block-local convolution and candidate selector |
| overlap placeholders and KV ownership | linear Eagle verify topology adapter |
| `SpeculativeE2ECudaGraphRunner` | |

There is no DFlash2 request state, scheduler branch, delayed-release protocol,
accept kernel, pack kernel, or graph runner.

## Decode

For block size `B`, one DFlash2 decode graph performs:

1. Materialize the previous accepted target contexts into the draft KV slots
   prepared by the shared speculative scheduler.
2. Build `[anchor, MASK, ..., MASK]` and its consecutive positions.
3. Run the non-causal DFlash2 draft block against the committed draft prefix.
4. Apply DFlash2's block-local dynamic convolutions around each attention/MLP
   sublayer, then use the target LM head to form top-k unary candidates, score
   adjacent candidate transitions, and choose one linear path of `B - 1`
   proposals.
5. Represent `[anchor, proposals...]` as a linear `EagleVerifyInput` and call
   the shared target verify/accept path.
6. Project the captured target hidden states into the shared Eagle hidden-state
   buffer for the next overlapped step.

The entire steady-state sequence above is the callback captured by
`SpeculativeE2ECudaGraphRunner`. Host-side overlap ordering and cache ownership are
therefore exactly the Eagle3 ordering.

Prefill is the only different entry path: the target performs normal prefill,
then its captured hidden states are projected directly into draft KV. The
result is published as `EagleSpecInput`, so all following steps use the shared
pipeline.

## Supported checkpoint contract

The current model backend supports Qwen3-family target and draft configs with
dense weights and greedy decoding. Dimensions are read from the checkpoint;
Qwen3-4B is the initial validated model, not a hard-coded shape.

DFlash2 fields live under `dflash_config`:

- `block_size`
- `mask_token_id`
- `selector_rank`
- `selector_top_k`
- `target_layer_ids`

The validated checkpoint preserves the trained DFlash tensors, initializes
each new convolution from an identity base plus random dynamic projection, and
randomly initializes the selector tensors.

## Validation

Validation is offline only. It checks that each request produces exactly its
requested token count, all token IDs are in the target vocabulary, every
decode replay has a captured batch-size graph, and target/draft KV usage
returns to its pre-request baseline.

Node-level CUDA Graph coverage is checked separately with `nsys profile` and
`--cuda-graph-trace=node`. A valid steady-state profile has one
`cudaGraphLaunch` per decode step, with the DFlash2 block preparation, selector,
draft forward, target verify, acceptance, and KV-materialization kernels all
reported as graph nodes rather than standalone launches.

The helper does not claim bitwise target-only or eager parity. Target verify
uses a batched extend kernel while target-only decoding uses a decode kernel;
with BF16, near-tied logits can therefore choose different argmax tokens even
when speculative acceptance is correct.
