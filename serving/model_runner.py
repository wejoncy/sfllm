import torch
import multiprocessing
import asyncio
import torch.cuda.graphs as cuda_graphs
from transformers.cache_utils import (
    DynamicCache,
    StaticCache,
)
import time
from typing import Dict, List, Any, Optional


async def generate_text(model, inputs, temperature=0.7, top_p=1.0, 
                        max_tokens=64):
    """
    Generate text based on the prompt and optional image embeddings
    
    Args:
        model: The loaded model and tokenizer
        temperature: Controls randomness in boltzmann distribution
        top_p: Controls diversity via nucleus sampling
        max_tokens: Maximum number of tokens to generate        
    Returns:
        The generated text
    """
    model_obj = model.model
    tokenizer = model.tokenizer.tokenizer
    # Use a thread pool to run the model inference
    loop = asyncio.get_event_loop()
    
    def _generate():
        input_length = inputs['input_ids'].shape[1]
            
        # Regular text generation
        with torch.no_grad():
            outputs = model_obj.generate(
                **inputs,
                max_new_tokens = max_tokens,
                # temperature=temperature,
                # top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
                # use_cache=True,
                return_dict_in_generate=True,
                output_scores=False,
                do_sample=False,
            )

        # Initialize generated tokens with the first token from prefill
        generated_ids = outputs.sequences
        return generated_ids

    
    # Run the generation in a separate thread to avoid blocking the event loop
    return await loop.run_in_executor(None, _generate)


async def batch_generate_text(model, batch_inputs):
    """
    Generate text for a batch of requests.
    
    Args:
        model: The model to use for generation
        batch_inputs: List of tokenized inputs
        
    Returns:
        List of generated text outputs
    """
    if not batch_inputs:
        return []
    
    results = []
    
    # Process each input in the batch
    inputs_ids =[i[-1].input_ids for i in batch_inputs]
    attentionmask =[i[-1].attention_mask for i in batch_inputs]
    (_, temperature, top_p, max_tokens, inputs) = batch_inputs[0]
    inputs_ids = torch.cat(inputs_ids, dim=0)
    attentionmask = torch.cat(attentionmask, dim=0)
    inputs = {
        'input_ids': inputs_ids,
        'attention_mask': attentionmask
    }
    generated_ids = await generate_text(
        model,
        inputs,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )
    results = []
    input_length = inputs['input_ids'].shape[1]
    generated_texts = model.tokenizer.tokenizer.batch_decode(generated_ids[:,input_length:], skip_special_tokens=True)
    for generated_text in generated_texts:
        response = generated_text.strip()
        total_token = generated_ids[0].shape[0]
        results.append({"text": response,
            "usage": {
                "prefill_token": input_length,
                "completion_token": total_token - input_length,
                "total_token": generated_ids[0].shape[0],
            }
        })
    return results
