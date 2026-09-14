# Qwen3.8-27B

SFLLM supports text generation from the official
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) checkpoint using
the dense Qwen3.5 implementation. The official config declares
`Qwen3_5ForConditionalGeneration`, so SFLLM loads that registered class.
[`qwen3_8.py`](../python/sfllm/models/qwen3_8.py) also provides the separate
`Qwen3_8ForConditionalGeneration` entry, inheriting the same implementation,
for configs explicitly declaring that architecture. Model selection follows
the config's `architectures` field.
The 64 decoder layers contain 48 Gated
DeltaNet layers and 16 full-attention layers. The checkpoint's GDN `swish`
gate is equivalent to SiLU; it does not change full-attention sigmoid gating.

## Start the server

```bash
hf download Qwen/Qwen3.8-27B --local-dir /mnt/data/work/Qwen3.8-27B
python python/sfllm/serving/app.py \
  --model /mnt/data/work/Qwen3.8-27B \
  --dtype bfloat16 --attention-backend fa3 \
  --max-running-requests 8 --cuda-graph-max-bs 8 --port 8081
```

This configuration has been exercised on an H100 96 GB, with default FP32
recurrent states, FlashInfer GDN and decode CUDA Graphs. The model download
is about 55.6 GB. Adjust concurrency and context length to available memory;
the command above retains SFLLM's 8192-token context limit.

## Chat template controls

Qwen3.8 defaults to thinking enabled, `reasoning_effort="xhigh"`, and preserved
historical thinking. For a direct answer:

```bash
curl http://localhost:8081/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3.8-27B",
    "messages": [{"role": "user", "content": "请用一句话解释什么是大语言模型。"}],
    "max_tokens": 128,
    "temperature": 0.7,
    "top_p": 0.8,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

For thinking mode, set `"reasoning_effort": "low"`, `"medium"`, or `"xhigh"`
at the top level of the chat request and leave `enable_thinking` enabled.
The top-level value takes precedence over `chat_template_kwargs.reasoning_effort`.
SFLLM passes these options to the checkpoint's own template; invalid effort
values return HTTP 400 for non-streaming requests, or an error in the event
stream, without stopping the tokenizer worker.

For multiple turns, include the previous assistant answer as `content` and
its thinking text as `reasoning_content` (`reasoning` is accepted as an alias).
Set `"chat_template_kwargs": {"preserve_thinking": false}` to disable retention
of historical thinking. `/generate` also accepts `messages` together with
`chat_template_kwargs`; raw text and input token IDs bypass chat templating.

## Scope

- Text prefill and decode use the existing dense Qwen3.5 implementation.
- The vision encoder, image/video inputs, and native MTP prediction are not
  supported by this text path.
- Generated thinking remains in the existing text response; the server does
  not split out a separate reasoning response field or parse tool calls.
- The BF16 checkpoint is the validated configuration. Quantized exports,
  speculative drafts, and extended context lengths need separate validation.

## Regression checks

With SFLLM and its development dependencies installed:

```bash
python -m pytest tests/test_qwen3_8.py sfkernels/tests/test_ops.py sfkernels/tests/test_rmsnorm.py
```

The chat tests use a pinned copy of the official template and configuration,
without downloading weights. CUDA tests cover the model's 48-head GDN gate
and 24-head full-attention gate shapes.

The initial H100 BF16 validation passed 93 regression checks. Two short greedy
generation samples (arithmetic and Chinese text) matched Transformers 5.12.1
token for token, including EOS. Server checks also covered eight concurrent
requests, all three reasoning efforts, historical-thinking retention, streaming,
and successful requests after invalid template parameters. These smoke checks
do not establish model quality or long-context accuracy.
