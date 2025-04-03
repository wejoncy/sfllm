import torch
import multiprocessing
import asyncio
import torch.cuda.graphs as cuda_graphs
from transformers.cache_utils import (
    DynamicCache,
    StaticCache,
)
from serving.model_loader import USE_SGLANG
# Global cache for CUDA graphs by model type
_cuda_graph_cache = {}
_kv_cache_pool = {}


def worker(model_obj,prompts,sampling_params,q):
    outputs = model_obj.generate(prompts, sampling_params)
    q.put(outputs)

async def generate_text(model, prompt, temperature=0.7, top_p=1.0, 
                        max_tokens=1024, image_embeddings=None, 
                        is_gemma=True, use_cuda_graph=True):
    """
    Generate text based on the prompt and optional image embeddings
    
    Args:
        model: The loaded model and tokenizer
        prompt: The input prompt
        temperature: Controls randomness in boltzmann distribution
        top_p: Controls diversity via nucleus sampling
        max_tokens: Maximum number of tokens to generate
        image_embeddings: Optional list of image embeddings for multimodal input
        is_gemma: Flag indicating if this is a Gemma model
        use_cuda_graph: Whether to use CUDA graphs for generation (only for token generation, not prefill)
        
    Returns:
        The generated text
    """
    use_cuda_graph = False
    tokenizer = model["tokenizer"]
    model_obj = model["model"]
    processor = model.get("processor")
    sampling_params = {"temperature": temperature, "top_p": top_p,"max_new_tokens": max_tokens}

    if USE_SGLANG:
        q = multiprocessing.Queue()
        prompts = [prompt]
        p = multiprocessing.Process(target=worker, args=(model_obj,prompts,sampling_params, q))
        p.start()
        p.join()
        outputs = q.get()
        # outputs = model_obj.generate(prompts, sampling_params)
        # Print the outputs.
        return outputs[0]['text']
    # Use a thread pool to run the model inference
    loop = asyncio.get_event_loop()
    
    def _generate():
        # Set model to evaluation mode
        model_obj.eval()
        
        # Check if model is on CUDA
        is_cuda = next(model_obj.parameters()).is_cuda
        model_name = model_obj.__class__.__name__
        max_seq_length = 2048
        
        # Tokenize the prompt
        inputs = tokenizer(prompt, return_tensors="pt")
        if is_cuda:
            inputs = {k: v.to(model_obj.device) for k, v in inputs.items()}
            
        input_length = inputs['input_ids'].shape[1]
        
        # Initialize cache manager for this model if not already done
        if use_cuda_graph and is_cuda and model_name not in _kv_cache_pool:
            _kv_cache_pool[model_name] = create_kv_cache_pool(model_obj, max_seq_length)
            
        # Handle multimodal input for the prefill stage
        if image_embeddings:
            if hasattr(model_obj, "process_images"):
                # For models with built-in image processing
                with torch.no_grad():
                    # Prefill stage - we don't use CUDA graph here
                    outputs = model_obj.generate(
                        **inputs,
                        images=image_embeddings,
                        max_length=input_length + 1,  # Just get the first token
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=temperature > 0,
                        pad_token_id=tokenizer.eos_token_id,
                        use_cache=True,
                        return_dict_in_generate=True,
                        output_scores=False
                    )
            elif processor:
                # For models requiring separate image processing
                image_features = processor(image_embeddings)
                if is_cuda:
                    image_features = image_features.to(model_obj.device)
                
                with torch.no_grad():
                    # Prefill stage
                    outputs = model_obj.generate(
                        **inputs,
                        image_features=image_features,
                        max_length=input_length + 1,  # Just get the first token
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=temperature > 0,
                        pad_token_id=tokenizer.eos_token_id,
                        use_cache=True,
                        return_dict_in_generate=True,
                        output_scores=False
                    )
            else:
                # Regular text generation
                with torch.no_grad():
                    # Prefill stage
                    outputs = model_obj.generate(
                        **inputs,
                        max_length=input_length + 1,  # Just get the first token
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=temperature > 0,
                        pad_token_id=tokenizer.eos_token_id,
                        use_cache=True,
                        return_dict_in_generate=True,
                        output_scores=False
                    )
        else:
            # Text-only generation
            with torch.no_grad():
                # Prefill stage
                outputs = model_obj.generate(
                    **inputs,
                    max_length=max_tokens,  # Just get the first token
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                    return_dict_in_generate=True,
                    output_scores=False,
                    do_sample=False
                )
                generated_ids = outputs
        
        # Initialize generated tokens with the first token from prefill
        generated_ids = outputs.sequences
        current_length = generated_ids.shape[1]
        
        # Token generation stage - this is where we can use CUDA graphs
        if use_cuda_graph and is_cuda:
            # Get or create CUDA graph for this model
            if model_name not in _cuda_graph_cache:
                _cuda_graph_cache[model_name] = {}
            
            # Generate the remaining tokens using CUDA Graph for the decoding loop
            remaining_tokens = min(max_tokens - 1, max_seq_length - current_length)
            
            # Prepare KV cache from the prefill stage
            past_key_values = outputs.past_key_values
            
            # Get cached kv_cache pool for this model
            kv_cache_pool = _kv_cache_pool.get(model_name)
            
            # Auto-regressive generation with CUDA Graph
            for i in range(max_seq_length):
                # Use the last generated token as input
                input_ids = generated_ids[:, -1].unsqueeze(-1)
                
                # Get relevant parameters for the generation step
                temperature_key = f"temp_{temperature:.1f}"
                if temperature_key not in _cuda_graph_cache[model_name]:
                    # Create and capture CUDA graph for this temperature
                    _cuda_graph_cache[model_name][temperature_key] = create_cuda_graph(
                        model_obj, input_ids, past_key_values, temperature, top_p
                    )
                
                # Run the cached CUDA graph
                graph = _cuda_graph_cache[model_name][temperature_key]
                next_token_logits, next_token, updated_past = run_cuda_graph(
                    graph, input_ids, past_key_values, kv_cache_pool, i
                )
                
                # Update KV cache with the new values
                past_key_values = updated_past
                
                # Add token to the generated sequence
                generated_ids = torch.cat([generated_ids, next_token.unsqueeze(-1)], dim=-1)
                
                # Check for EOS token
                if next_token.item() == tokenizer.eos_token_id:
                    break
        else:
            # Standard autoregressive generation without CUDA graph
            with torch.no_grad():
                remaining_tokens = min(max_tokens - 1, max_seq_length - current_length)
                # if remaining_tokens > 0:
                #     outputs = model_obj.generate(
                #         **inputs,
                #         max_length=input_length + max_tokens,
                #         temperature=temperature,
                #         top_p=top_p,
                #         do_sample=temperature > 0,
                #         pad_token_id=tokenizer.eos_token_id
                #     )
                #     generated_ids = outputs
        
        # Decode the generated tokens
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # Handle Gemma-specific formatting in the response
        if is_gemma:
            # Find where the model output starts and ends
            if "<end_of_turn>" in generated_text[len(prompt):]:
                # Extract just the model's response without the end tag
                response = generated_text[len(prompt):].split("<end_of_turn>")[0].strip()
            else:
                response = generated_text[len(prompt):].strip()
            return response
        else:
            # Return only the newly generated text (remove the prompt)
            return generated_text[len(prompt):]
    
    # Run the generation in a separate thread to avoid blocking the event loop
    return await loop.run_in_executor(None, _generate)

def create_kv_cache_pool(model, max_seq_length):
    """
    Create a pre-allocated pool of key-value caches for the model.
    This avoids memory fragmentation during generation.
    
    Args:
        model: The model to create caches for
        max_seq_length: Maximum sequence length
        
    Returns:
        A pre-allocated KV cache
    """
    # Detect the model architecture and create appropriate KV cache
    if hasattr(model, "config"):
        config = model.config
        num_layers = getattr(config, "num_hidden_layers", 
                      getattr(config, "n_layer", 
                      getattr(config, "num_layers", 12)))
        num_heads = getattr(config, "num_attention_heads", 
                    getattr(config, "n_head", 
                    getattr(config, "num_heads", 12)))
        hidden_size = getattr(config, "hidden_size", 768)
        head_dim = hidden_size // num_heads
        
        # Create a placeholder for KV cache
        # The shape depends on the model architecture
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        
        # Common format: [batch_size, num_heads, seq_len, head_dim]
        kv_cache = []
        for _ in range(num_layers):
            # For each layer, create key and value cache
            layer_cache = (
                torch.zeros((1, num_heads, max_seq_length, head_dim), device=device, dtype=dtype),
                torch.zeros((1, num_heads, max_seq_length, head_dim), device=device, dtype=dtype)
            )
            kv_cache.append(layer_cache)
        
        return kv_cache
    
    return None

def create_cuda_graph(model, input_ids, past_key_values, temperature, top_p):
    """
    Create and capture a CUDA graph for a generation step
    
    Args:
        model: The model to use
        input_ids: Input token IDs
        past_key_values: Past key-values from previous step
        temperature: Temperature for sampling
        top_p: Top-p for nucleus sampling
        
    Returns:
        A CUDA graph for the generation step
    """
    # Clone inputs to avoid modifying the originals during capture
    static_input_ids = input_ids.clone()
    static_past = [(k.clone(), v.clone()) for k, v in past_key_values]
    
    # Warmup before capture to ensure all kernels are compiled
    for _ in range(3):
        with torch.no_grad():
            outputs = model(
                input_ids=static_input_ids,
                past_key_values=static_past,
                use_cache=True
            )
            logits = outputs.logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            if temperature > 0:
                probs = torch.nn.functional.softmax(logits, dim=-1)
                # Apply top-p filtering
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                mask = cumulative_probs > top_p
                mask[..., 0] = 0  # Keep at least the top token
                sorted_probs[mask] = 0.0
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                next_token = torch.multinomial(sorted_probs, num_samples=1).squeeze(-1)
                next_token = sorted_indices.gather(-1, next_token.unsqueeze(-1)).squeeze(-1)
            else:
                next_token = torch.argmax(logits, dim=-1)
    
    # Capture the CUDA graph
    g = cuda_graphs.CUDAGraph()
    
    with torch.cuda.graph(g):
        with torch.no_grad():
            outputs = model(
                input_ids=static_input_ids,
                past_key_values=static_past,
                use_cache=True
            )
            logits = outputs.logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            
            if temperature > 0:
                probs = torch.nn.functional.softmax(logits, dim=-1)
                # Apply top-p filtering
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                mask = cumulative_probs > top_p
                mask[..., 0] = 0  # Keep at least the top token
                sorted_probs[mask] = 0.0
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                next_token = torch.multinomial(sorted_probs, num_samples=1).squeeze(-1)
                next_token = sorted_indices.gather(-1, next_token.unsqueeze(-1)).squeeze(-1)
            else:
                next_token = torch.argmax(logits, dim=-1)
    
    # Return graph and static tensors for reuse
    return {
        'graph': g,
        'static_input_ids': static_input_ids,
        'static_past': static_past,
        'logits': logits,
        'next_token': next_token,
        'outputs': outputs
    }

def run_cuda_graph(graph_data, input_ids, past_key_values, kv_cache_pool, step_idx):
    """
    Run a captured CUDA graph for token generation
    
    Args:
        graph_data: The captured graph data
        input_ids: New input token IDs
        past_key_values: Current past_key_values
        kv_cache_pool: Pre-allocated KV cache pool
        step_idx: Current generation step
        
    Returns:
        next_token_logits, next_token, updated_past
    """
    # Copy new inputs to static tensors
    graph_data['static_input_ids'].copy_(input_ids)
    
    # Copy past_key_values to static tensors
    for i, (k, v) in enumerate(past_key_values):
        graph_data['static_past'][i][0].copy_(k)
        graph_data['static_past'][i][1].copy_(v)
    
    # Run the graph
    graph_data['graph'].replay()
    
    # Copy the new past_key_values to our pre-allocated cache
    if kv_cache_pool:
        updated_past = []
        for layer_idx, (k, v) in enumerate(graph_data['outputs'].past_key_values):
            # Use pre-allocated cache
            k_cache, v_cache = kv_cache_pool[layer_idx]
            
            # Copy the new values to the pre-allocated cache
            # Only update the position we're currently at
            k_cache[:, :, step_idx:step_idx+1, :] = k
            v_cache[:, :, step_idx:step_idx+1, :] = v
            
            # Create tuple of views to the cache
            updated_past.append((
                k_cache[:, :, :step_idx+1, :],
                v_cache[:, :, :step_idx+1, :]
            ))
    else:
        # If no cache pool, just use the output directly
        updated_past = graph_data['outputs'].past_key_values
    
    return graph_data['logits'], graph_data['next_token'], updated_past
