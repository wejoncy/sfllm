"""Run batched offline inference with a base, Eagle3, or DFlash2 model."""

import argparse

import tqdm

from sfllm.engine.inference_engine import InferenceEngine
from sfllm.engine.sampling_params import SamplingParams
from sfllm.server_args import ServerArgs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    ServerArgs.add_cli_args(parser)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    engine = InferenceEngine(server_args)
    prompts = [
        "Hello, my name is",
        # "Hello, my name is",
        # "Hello, my name is",
        "The president of the United States is",
        # "The president of the United States is",
        "The capital of France is",
        # "The capital of France is",
        "The future of AI is",
        # "The future of AI is",
        # "The future of AI is",
    ]
    # engine.add_request("Hello, world!", SamplingParams())
    outputs = engine.generate(
        prompts,
        SamplingParams(max_new_tokens=args.max_new_tokens, top_k=1),
        stream=False,
    )
    for output in tqdm.tqdm(outputs):
        for _, output_d in output.items():
            v = f"Prompt: {output_d['prompt']}\nGenerated text: {output_d['text']}"
            print(v)
    if server_args.speculative_algorithm is not None:
        extra_accepted = (
            engine.scheduler.metrics.cum_spec_accept_tokens
            - engine.scheduler.metrics.cum_forward_ct
        )
        print(f"Speculative extra accepted tokens: {extra_accepted}")
    print("Inference step completed.")
