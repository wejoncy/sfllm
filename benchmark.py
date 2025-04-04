import argparse
import asyncio
import time
import requests
import random
import aiohttp
import statistics
import tqdm
from typing import List, Dict

# Sample prompts for benchmarking
SAMPLE_PROMPTS = [
    "a " *512,
    "a " *1024,
]

async def send_request(session, url, prompt, max_tokens=64):
    """Send a single request and return timing information."""
    data = {
        "prompt": prompt,
        "temperature": 0.7,
        "max_tokens": max_tokens,
        "top_p": 0.95,
        "model": "gemma-7b"
    }
    
    start_time = time.time()
    try:
        async with session.post(url, json=data) as response:
            result = await response.json()
            end_time = time.time()
            return {
                "success": True,
                "latency": end_time - start_time,
                "status": response.status
            }
    except Exception as e:
        end_time = time.time()
        return {
            "success": False,
            "latency": end_time - start_time,
            "error": str(e)
        }

async def run_benchmark(url, concurrency, num_requests, prompts=None):
    """Run benchmark with the specified concurrency level."""
    if prompts is None or not prompts:
        prompts = SAMPLE_PROMPTS
        
    print(f"Running benchmark with concurrency={concurrency}, requests={num_requests}")
    results = []
    async with aiohttp.ClientSession() as session:
        tasks = []
        for _ in tqdm.tqdm(range(num_requests)):
            prompt = random.choice(prompts)
            tasks.append(asyncio.create_task(
                send_request(session, f"{url}/v1/completions", prompt)
            ))
            
            # Control concurrency
            if len(tasks) >= concurrency:
                # Wait for some tasks to complete
                results.extend(await asyncio.gather(*tasks[:concurrency]))
                tasks = tasks[concurrency:]
                
        # Wait for remaining tasks
        results .extend(await asyncio.gather(*tasks))
    
    return results

def print_results(results):
    """Print benchmark results in a nice format."""
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    
    print("\n--- Benchmark Results ---")
    print(f"Total Requests: {len(results)}")
    print(f"Successful: {len(successful)} ({len(successful)/len(results)*100:.1f}%)")
    print(f"Failed: {len(failed)} ({len(failed)/len(results)*100:.1f}%)")
    
    if successful:
        latencies = [r["latency"] for r in successful]
        print("\nLatency (seconds):")
        print(f"  Min: {min(latencies):.4f}")
        print(f"  Max: {max(latencies):.4f}")
        print(f"  Avg: {sum(latencies)/len(latencies):.4f}")
        if len(latencies) > 1:
            print(f"  Median: {statistics.median(latencies):.4f}")
            print(f"  Std Dev: {statistics.stdev(latencies):.4f}")
        
        print(f"\nThroughput: {len(successful) / sum(latencies):.2f} requests/second")
    
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
    parser.add_argument("--concurrency", type=int, default=10, help="Concurrent requests")
    parser.add_argument("--requests", type=int, default=10, help="Total number of requests")
    args = parser.parse_args()
    
    # Check server health before starting
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{args.url}/health") as response:
                health = await response.json()
                print(f"Server health check: {health}")
    except Exception as e:
        print(f"Error connecting to server: {e}")
        return
    
    # Run the benchmark
    results = await run_benchmark(args.url, args.concurrency, args.requests)
    
    # Print results
    print_results(results)

if __name__ == "__main__":
    asyncio.run(main())
