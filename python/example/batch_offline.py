import argparse
from sfllm.engine.inference_engine import InferenceEngine
from sfllm.engine.sampling_params import SamplingParams
from sfllm.server_args import ServerArgs

if __name__ == "__main__":
    # Example usage
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    # server_args.disable_cuda_graph = True
    engine = InferenceEngine(server_args)
    prompts = [
        "Hello, my name is",
        "Hello, my name is",
        "Hello, my name is",
        "The president of the United States is",
        "The president of the United States is",
        "The capital of France is",
        "The capital of France is",
        "The future of AI is",
        "The future of AI is",
        "The future of AI is",
    ]
    # engine.add_request("Hello, world!", SamplingParams())
    outputs = engine.generate(
        prompts, SamplingParams(max_new_tokens=2000, top_k=1), stream=False
    )
    for output in outputs:
        for _, output_d in output.items():
            v = f"Prompt: {output_d['prompt']}\nGenerated text: {output_d['text']}"
    print("Inference step completed.")
