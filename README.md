# Multimodal LLM Serving Framework for Gemma 3-4B-IT

This project implements a serving framework for Google's Gemma 3-4B-IT model that follows the OpenAI API protocol with support for both text and image inputs.

```bash
curl -X POST "http://localhost:6006/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-3-4b-it",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "temperature": 0.7,
    "max_new_tokens": 1024
  }'
```

# Cost improvement and analysis:

## OUTPUT
| before      | after |
| ----------- | ----------- |
|0.4 requests/second | 1.41 requests/second|


`python benchmark.py --concurrency 1 --requests 2` this is used to simulate the base line, we process it one by one
For the first version with running request one by one, we got **Throughput: 0.4 requests/second.** Given that rental price
is 7.68RMB per hour. we have nearly 7.68/(3600/3.02)=0.00644266666 RMB per request.

```
root@autodl-container-4c3247ac55-044103ae:~/work/gemma_serving# python benchmark.py --concurrency 1 --requests 40 --url http://104.208.77.11:8080
Server health check: {'status': 'healthy', 'queue_size': 0, 'workers': 1}
Running benchmark with concurrency=1, requests=40
Warming up the server...
100%|██████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:14<00:00,  2.81s/it]
Warm-up complete. starting benchmark...
100%|████████████████████████████████████████████████████████████████████████████████████████| 40/40 [01:51<00:00,  2.79s/it]
Waiting for remaining 0 tasks to complete...

--- Benchmark Results ---
Total Requests: 40
Successful: 40 (100.0%)
Failed: 0 (0.0%)

Latency (seconds):
  Min: 2.5529
  Max: 3.1295
  Avg: 2.7865
  Median: 2.6659
  Std Dev: 0.1082

Throughput: 0.36 requests/second
```
Hence, we can improve it from some aspect:
1. improve the model inference performance for a single forward process
  - cuda graph
  - hihgly optimized kernel for RMSnorm
  - fusion qkv/up_down matmul together
  - optimize attention kernel for decode
2. improve the model forward with ragged batching support
3. improve the model forward with  mixed PD fusion support
  - this involves scheduler support
4. support paged-kv-cache managerment
5. improve scheduler.
  with the support of paged-kv-cache, we can easily split p/D to different stage and fusion more batches together
6. more a base change, to implement a customized GEMMA3 definition, we need to customize input/output/attention and others
  - failed to get it through quickly
7. over-lapped scheduling/ Overlap CPU/GPU 
  - this is definitely a good direction to improve the system performance for small models like GEMMA3

Even I planned many optmizations which we can apply to the LLM server, It's super hard to implement it in a short time, even only for GEMMA3 without image input.

Then I start to seek more reasonable improvement which I can do in near 2 hours and we can have a improvement versus baseline.
Actually, I spend about 1.5 hours to debug the customized Model definition, which works but make me think about if it's doable to implement the others tech.

What we know is that, almost the time is consumed on *decoding time*. so we can try to batch *multi-request* together

So I focus on what I can do for the inference service on the acspect of performance.
1. batching
  - even it's almost impossible to run a ragged batching, and it's very unefficent to padding num of request to feed into one batch.
  - I used a scheduler to group those batch with similar input token length then we can easiyly batch running the model and improve gpu utilization.
  - given that batch-1 and batch 2 or 3 are almost the same time on decoding time. 
2. isolate the server and inference engine.
  - so we can uninterrupted accept new request from user and put it into a queue for scheduling.
3. schedulding
  - sort and group request according to their input length. we are maxmaize the throughput instead of the best TTPT/TPOT. 
  so it's doable to prior executing max batch with similar requests.


`python benchmark.py --concurrency 20 --requests 20`
after the optimiation，we achieved **Throughput: 1.41 requests/second.**. It's 2 times faster than before.
A very abvious improvement is the GPU utilization, the baseline can't leverage GPU all the time and it's very unefficent with only one batchsize.


Some benchmark results
```
Throughput: 1.79 requests/second
root@autodl-container-4c3247ac55-044103ae:~/work/gemma_serving# python benchmark.py --concurrency 20 --requests 40
Server health check: {'status': 'healthy', 'queue_size': 0, 'workers': 1}
Running benchmark with concurrency=20, requests=40
Warming up the server...
100%|██████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:12<00:00,  2.41s/it]
Warm-up complete. starting benchmark...
100%|████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:04<00:00,  9.88it/s]
Waiting for remaining 40 tasks to complete...

--- Benchmark Results ---
Total Requests: 40
Successful: 40 (100.0%)
Failed: 0 (0.0%)

Latency (seconds):
  Min: 2.3248
  Max: 19.9545
  Avg: 0.5596
  Median: 13.0541
  Std Dev: 4.9845

Throughput: 1.79 requests/second
root@autodl-container-4c3247ac55-044103ae:~/work/gemma_serving# python benchmark.py --concurrency 20 --requests 40 --url http://104.208.77.11:8080
Server health check: {'status': 'healthy', 'queue_size': 0, 'workers': 1}
Running benchmark with concurrency=20, requests=40
Warming up the server...
100%|██████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:14<00:00,  2.82s/it]
Warm-up complete. starting benchmark...
100%|████████████████████████████████████████████████████████████████████████████████████████| 40/40 [00:04<00:00,  9.90it/s]
Waiting for remaining 40 tasks to complete...

--- Benchmark Results ---
Total Requests: 40
Successful: 40 (100.0%)
Failed: 0 (0.0%)

Latency (seconds):
  Min: 3.4448
  Max: 19.2793
  Avg: 0.5603
  Median: 13.4163
  Std Dev: 4.5976

Throughput: 1.78 requests/second
```