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
    "max_tokens": 1024
  }'
```

# Cost improvement and analysis:

`python benchmark.py --concurrency 1 --requests 2` this is used to simulate the base line, we process it one by one
For the first version with running request one by one, we got **Throughput: 0.42 requests/second.** Given that rental price
is 7.68RMB per hour. we have nearly 7.68/(3600/3.02)=0.00644266666 RMB per request.

Hence, we can improve it from some aspect:
1. improve the model inference performance for a single forward process
  1. cuda graph
  2. hihgly optimized kernel for RMSnorm
  3. fusion qkv/up_down matmul together
  4. optimize attention kernel for decode
2. improve the model forward with ragged batching support
3. improve the model forward with  mixed PD fusion support
  1. this involves scheduler support
4. support paged-kv-cache managerment
5. improve scheduler.
  with the support of paged-kv-cache, we can easily split p/D to different stage and fusion more batches together
6. more a base change, to implement a customized GEMMA3 definition, we need to customize input/output/attention and others
  1. failed to get it through quickly
7. over-lapped scheduling/ Overlap CPU/GPU 
  1. this is definitely a good direction to improve the system performance for small models like GEMMA3

Even I planned many optmizations which we can apply to the LLM server, It's super hard to implement it in a short time, even only for GEMMA3 without image input.

Then I start to seek more reasonable improvement which I can do in near 2 hours and we can have a improvement versus baseline.
Actually, I spend about 1.5 hours to debug the customized Model definition, which works but make me think about if it's doable to implement the others tech.

What we know is that, almost the time is consumed on *decoding time*. so we can try to batch *multi-request* together

So I focus on what I can do for the inference service on the acspect of performance.
1. batching
  1. even it's almost impossible to run a ragged batching, and it's very unefficent to padding num of request to feed into one batch.
  2. I used a scheduler to group those batch with similar input token length then we can easiyly batch running the model and improve gpu utilization.
  3. given that batch-1 and batch 2 or 3 are almost the same time on decoding time. 
2. isolate the server and inference engine.
  1. so we can uninterrupted accept new request from user and put it into a queue for scheduling.
3. schedulding
  1. sort and group request according to their input length. we are maxmaize the throughput instead of the best TTPT/TPOT. 
  so it's doable to prior executing max batch with similar requests.


`python benchmark.py --concurrency 20 --requests 2`
after the optimiation，we achieved **Throughput: 0.2 requests/second.**. It's 2 times faster than before.
A very abvious improvement is the GPU utilization, the baseline can't leverage GPU all the time and it's very unefficent with only one batchsize.