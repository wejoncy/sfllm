import argparse
import asyncio
import time
import random
import aiohttp
import statistics
import tqdm
from dataclasses import dataclass
from typing import Optional, List
import json

SAMPLE_PROMPTS = [
    "a " * 64,
    "a " * 128,
]


@dataclass
class RequestMetrics:
    success: bool
    total_latency: float
    ttft: Optional[float] = None
    tokens: Optional[int] = None
    error: Optional[str] = None


async def send_request(session, url, prompt, stream=False, max_new_tokens=64):
    data = {
        "prompt": prompt,
        "temperature": 0.7,
        "max_new_tokens": max_new_tokens,
        "top_p": 0.95,
        "model": "gemma-7b",
        "stream": stream,
    }

    start_time = time.time()
    ttft = None
    token_count = 0

    try:
        async with session.post(url, json=data, timeout=150) as response:
            if not stream:
                await response.text()
            else:
                async for line in response.content:
                    if not line.strip():
                        continue
                    line = line.decode()[6:].strip()
                    if line == '[DONE]':
                        continue
                    now = time.time()
                    token_count += len(json.loads(line).get("output_ids", []))
                    if ttft is None:
                        ttft = now - start_time

        total_latency = time.time() - start_time
        return RequestMetrics(True, total_latency, ttft, token_count)

    except Exception as e:
        print(e)
        return RequestMetrics(False, time.time() - start_time, None, None, str(e))


async def warm_up(session, url, prompts, stream=False):
    print("Warming up the server...")
    prompt = random.choice(prompts)
    await send_request(session, f"{url}/v1/completions", prompt, stream=stream)
    print("Warm-up complete.\n")


async def run_benchmark(url, concurrency, num_requests, stream=False, prompts=None, max_new_tokens=64):
    if not prompts:
        prompts = SAMPLE_PROMPTS

    print(
        f"Running benchmark: concurrency={concurrency}, requests={num_requests}, stream={stream}, max_new_tokens={max_new_tokens}"
    )

    async with aiohttp.ClientSession() as session:
        await warm_up(session, url, prompts, stream=stream)
    timeout = aiohttp.ClientTimeout(total=150)
    results = []
    progress = tqdm.tqdm(total=num_requests, desc="Benchmark Progress")

    connector = aiohttp.TCPConnector(limit=0, limit_per_host=concurrency)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        start_time = time.time()
        queue = asyncio.Queue()

        for _ in range(num_requests):
            queue.put_nowait(random.choice(prompts))

        async def worker(worker_id: int):
            while True:
                try:
                    prompt = await queue.get()
                except asyncio.CancelledError:
                    break
                result = await send_request(
                    session, f"{url}/v1/completions", prompt, stream=stream, max_new_tokens=max_new_tokens
                )
                results.append(result)
                progress.update(1)
                queue.task_done()

        workers = [asyncio.create_task(worker(i)) for i in range(concurrency)]
        await queue.join()

        for w in workers:
            w.cancel()

        total_time = time.time() - start_time
        progress.close()

    return results, total_time


def print_results(results_tuple):
    results, total_time = results_tuple
    success = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    print("\n--- Benchmark Results ---")
    print(f"Total Requests: {len(results)}")
    print(f"Successful: {len(success)} ({len(success) / len(results) * 100:.1f}%)")
    print(f"Failed: {len(failed)} ({len(failed) / len(results) * 100:.1f}%)")

    if success:
        total_lat = [r.total_latency*1000 for r in success]
        ttfts = [r.ttft*1000 for r in success if r.ttft is not None]
        tpot = [
            (r.total_latency - (r.ttft or 0)) / r.tokens
            for r in success
            if r.tokens and r.tokens > 1
        ]
        total_tokens = sum(r.tokens or 0 for r in success)

        print("\nLatency (milliseconds):")
        print(f"  Min: {min(total_lat):.4f}")
        print(f"  Max: {max(total_lat):.4f}")
        print(f"  Avg: {statistics.mean(total_lat):.4f}")
        if len(total_lat) > 1:
            print(f"  P50: {statistics.median(total_lat):.4f}")
            print(f"  P95: {sorted(total_lat)[int(0.95 * len(total_lat)) - 1]:.4f}")

        if ttfts:
            print("\nTime to First Token (TTFT):")
            print(f"  Avg: {statistics.mean(ttfts):.4f}")
            print(f"  P50: {statistics.median(ttfts):.4f}")

        if tpot:
            print("\nAvg Time per Output Token (TPOT):")
            print(f"  Avg: {statistics.mean(tpot):.6f}")
            print(f"  P50: {statistics.median(tpot):.6f}")

        print(f"\nThroughput:")
        print(f"  Requests/sec: {len(success) / total_time:.2f}")
        if total_tokens > 0:
            print(f"  Tokens/sec: {total_tokens / total_time:.2f}")

    if failed:
        print("\nError summary:")
        errors = {}
        for r in failed:
            err = r.error or "Unknown"
            errors[err] = errors.get(err, 0) + 1
        for err, count in errors.items():
            print(f"  {err}: {count} occurrences")


async def main():
    parser = argparse.ArgumentParser(description="Benchmark the LLM serving system")
    parser.add_argument("--url", default="http://localhost:8081", help="Server URL")
    parser.add_argument("--concurrency", type=int, default=32, help="Concurrent requests")
    parser.add_argument("--requests", type=int, default=100, help="Total number of requests")
    parser.add_argument("--max_new_tokens", type=int, default=640, help="Max new tokens per request")
    parser.add_argument(
        "--stream", action="store_true", default=True, help="Enable streaming mode"
    )
    args = parser.parse_args()

    try:
        connector = aiohttp.TCPConnector(limit=0, limit_per_host=args.concurrency)
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.get(f"{args.url}/health", timeout=5) as resp:
                print("Server health:", await resp.text())
    except Exception as e:
        print(f"Error connecting to server: {e}")
        return

    try:
        results = await run_benchmark(
            args.url, args.concurrency, args.requests, stream=args.stream, max_new_tokens=args.max_new_tokens
        )
        print_results(results)
    except KeyboardInterrupt:
        print("\nBenchmark interrupted.")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
