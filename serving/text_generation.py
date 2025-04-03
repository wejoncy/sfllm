import torch
import multiprocessing
import asyncio
import torch.cuda.graphs as cuda_graphs
from transformers.cache_utils import (
    DynamicCache,
    StaticCache,
)
# Global cache for CUDA graphs by model type
_cuda_graph_cache = {}
_kv_cache_pool = {}


async def generate_text(model, prompt, messages, temperature=0.7, top_p=1.0, 
                        max_tokens=64, image_embeddings=None, 
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
    if prompt is None:
        prompt = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False,
                return_dict=True, return_tensors="pt"
            )#.to(model_obj.device, dtype=torch.bfloat16)

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
            
        # Regular text generation
        with torch.no_grad():
            outputs = model_obj.generate(
                **inputs,
                max_new_tokens = max_tokens,  # Just get the first token
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
                return_dict_in_generate=True,
                output_scores=False,
                do_sample=False,
            )

        # Initialize generated tokens with the first token from prefill
        generated_ids = outputs.sequences
        generated_text = tokenizer.decode(generated_ids[0,input_length:], skip_special_tokens=True)
        response = generated_text.strip()
        return response

    
    # Run the generation in a separate thread to avoid blocking the event loop
    return await loop.run_in_executor(None, _generate)
