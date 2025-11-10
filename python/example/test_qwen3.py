def generate_greedy(model, tokenizer, prompt, max_new_tokens=50, device="cuda"):
    model.eval()
    model.to(device)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt = "Give me a short introduction to large language model."
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,  # Switches between thinking and non-thinking modes. Default is True.
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    input_ids = model_inputs["input_ids"]

    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(input_ids=input_ids[:, -1:])
            logits = outputs
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True) - _

        input_ids = torch.cat([input_ids, next_token], dim=-1)

        if (
            tokenizer.eos_token_id is not None
            and next_token.item() == tokenizer.eos_token_id
        ):
            break

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


if __name__ == "__main__":
    import transformers

    model_path = r"D:\\work\\Qwen3-0.6B"

    config = transformers.AutoConfig.from_pretrained(model_path)
    forward_batch = ForwardBatch(config)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, use_fast=False)
    with TorchDefaultDtype(config.dtype):
        model = Qwen3ForCausalLM(config).cuda()
        _load_check_point(model, model_path)
    model.eval()
    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

    attention_mask = inputs["attention_mask"].cuda()
    input_ids = inputs["input_ids"].cuda()

    out = generate_greedy(model, tokenizer, "Hello, my dog is cute", max_new_tokens=20)
    print(out)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        forward_batch=forward_batch,
    )
    print(outputs[0].shape)  # logits shape
