import argparse
import asyncio
import json
import time
import random
import aiohttp
import statistics
import tqdm
from typing import List, Optional
from dataclasses import dataclass

SAMPLE_PROMPTS = [
    "a " * 512,
    "a " * 1024,
]

@dataclass
class DatasetRow:
    prompt: str
    prompt_len: int
    output_len: int
    text_prompt_len: Optional[int] = None
    vision_prompt_len: Optional[int] = None
    image_data: Optional[List[str]] = None
    timestamp: Optional[float] = None

    def __post_init__(self):
        if self.text_prompt_len is None:
            self.text_prompt_len = self.prompt_len
        if self.vision_prompt_len is None:
            self.vision_prompt_len = 0

def sample_sharegpt_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer,
    fixed_output_len: Optional[int] = None,
    context_len: Optional[int] = None,
    prompt_suffix: Optional[str] = "",
    apply_chat_template=False,
) -> List[DatasetRow]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")

    # Download sharegpt if necessary
    if dataset_path == "":
        dataset_path = download_and_cache_file(SHAREGPT_URL)

    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)

    # Filter out the conversations with less than 2 turns.
    dataset = [
        data
        for data in dataset
        if len(data.get("conversations", data.get("conversation", []))) >= 2
    ]
    # Only keep the first two turns of each conversation.
    dataset = [
        (
            data.get("conversations", data.get("conversation", []))[0]["value"],
            data.get("conversations", data.get("conversation", []))[1]["value"],
        )
        for data in dataset
    ]

    # Shuffle the dataset.
    random.shuffle(dataset)

    # Filter out sequences that are too long or too short
    filtered_dataset: List[DatasetRow] = []
    for i in range(len(dataset)):
        if len(filtered_dataset) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt = dataset[i][0]
        if prompt_suffix:
            prompt = (
                remove_suffix(prompt, ASSISTANT_SUFFIX)
                + prompt_suffix
                + ASSISTANT_SUFFIX
            )

        if apply_chat_template:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
            if tokenizer.bos_token:
                prompt = prompt.replace(tokenizer.bos_token, "")

        prompt_token_ids = tokenizer.encode(prompt)
        completion = dataset[i][1]
        completion_token_ids = tokenizer.encode(completion)
        prompt_len = len(prompt_token_ids)
        output_len = (
            len(completion_token_ids) if fixed_output_len is None else fixed_output_len
        )

        if prompt_len < 2 or output_len < 2:
            # Prune too short sequences.
            continue

        if context_len and prompt_len + output_len > context_len:
            # Prune too long sequences.
            continue

        filtered_dataset.append(
            DatasetRow(
                prompt=prompt,
                prompt_len=prompt_len,
                output_len=output_len,
            )
        )

    print(f"#Input tokens: {np.sum([x.prompt_len for x in filtered_dataset])}")
    print(f"#Output tokens: {np.sum([x.output_len for x in filtered_dataset])}")
    return filtered_dataset

async def send_request(session, url, prompt, max_new_tokens=64):
    """Send a single request and return timing information."""
    data = {
        "prompt": prompt,
        "temperature": 0.7,
        "max_new_tokens": max_new_tokens,
        "top_p": 0.95,
        "model": "gemma-7b"
    }

    start_time = time.time()
    try:
        async with session.post(url, json=data, timeout=150) as response:
            result = await response.json()
            latency = time.time() - start_time
            return {
                "success": True,
                "latency": latency,
                "status": response.status
            }
    except Exception as e:
        return {
            "success": False,
            "latency": time.time() - start_time,
            "error": str(e)
        }


async def warm_up(session, url, prompts):
    """Warm up the server before the benchmark."""
    print("Warming up the server...")
    for _ in tqdm.tqdm(range(5)):
        prompt = random.choice(prompts)
        await send_request(session, f"{url}/v1/completions", prompt)
    print("Warm-up complete.")


async def run_benchmark(url, concurrency, num_requests, prompts=None):
    """Run benchmark with a semaphore to control concurrency."""
    if not prompts:
        prompts = SAMPLE_PROMPTS

    print(f"Running benchmark with concurrency={concurrency}, requests={num_requests}")

    timeout = aiohttp.ClientTimeout(total=30)
    results = []
    sem = asyncio.Semaphore(concurrency)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Warm up first
        await warm_up(session, url, prompts)

        print("Starting main benchmark...")
        start_time = time.time()

        async def worker(prompt):
            async with sem:
                r = await send_request(session, f"{url}/v1/completions", prompt)
                results.append(r)

        tasks = [
            asyncio.create_task(worker(random.choice(prompts)))
            for _ in range(num_requests)
        ]

        for f in tqdm.tqdm(asyncio.as_completed(tasks), total=num_requests):
            try:
                await f
            except Exception as e:
                results.append({"success": False, "latency": 0, "error": str(e)})

        total_time = time.time() - start_time

    return results, total_time


def print_results(results_tuple):
    """Print benchmark results in a nice format."""
    results, total_time = results_tuple
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print("\n--- Benchmark Results ---")
    print(f"Total Requests: {len(results)}")
    print(f"Successful: {len(successful)} ({len(successful) / len(results) * 100:.1f}%)")
    print(f"Failed: {len(failed)} ({len(failed) / len(results) * 100:.1f}%)")

    if successful:
        latencies = [r["latency"] for r in successful]
        print("\nLatency (seconds):")
        print(f"  Min: {min(latencies):.4f}")
        print(f"  Max: {max(latencies):.4f}")
        print(f"  Avg: {statistics.mean(latencies):.4f}")
        if len(latencies) > 1:
            print(f"  Median: {statistics.median(latencies):.4f}")
            print(f"  Std Dev: {statistics.stdev(latencies):.4f}")

        print(f"\nThroughput: {len(successful) / total_time:.2f} requests/second")

    if failed:
        print("\nError summary:")
        errors = {}
        for r in failed:
            error = r.get("error", "Unknown")
            errors[error] = errors.get(error, 0) + 1
        for error, count in errors.items():
            print(f"  {error}: {count} occurrences")


async def main():
    parser = argparse.ArgumentParser(description="Benchmark the LLM serving system")
    parser.add_argument("--url", default="http://localhost:8080", help="Server URL")
    parser.add_argument("--concurrency", type=int, default=1, help="Concurrent requests")
    parser.add_argument("--requests", type=int, default=3, help="Total number of requests")
    args = parser.parse_args()

    # Check server health before starting
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{args.url}/health", timeout=5) as response:
                health = await response.json()
                print(f"Server health check: {health}")
    except Exception as e:
        print(f"Error connecting to server: {e}")
        return

    # Run benchmark
    try:
        # input_requests = sample_sharegpt_requests("D:\\work\\gemma_serving\\ShareGPT_V3_unfiltered_cleaned_split.json", 10, args.model)
        results = await run_benchmark(args.url, args.concurrency, args.requests)
        print_results(results)
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user.")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
