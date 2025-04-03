import os
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
import sglang as sgl

USE_SGLANG = os.getenv("USE_SGLANG", "0") == "1"
MODEL_PATH = "/root/work/gemma-3-4b-it"

def load_model(model_name=MODEL_PATH):
    """
    Load the model and tokenizer
    
    Args:
        model_name: The name or path of the model to load
        
    Returns:
        A dictionary containing model, tokenizer, and processor
    """
    print(f"Loading model: {model_name}")
    
    # use sglang
    if USE_SGLANG:
        model = sgl.Engine(model_path=MODEL_PATH)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Try loading a processor for vision models
    try:
        processor = AutoProcessor.from_pretrained(model_name)
    except:
        processor = None

        
    return {
        "model": model, 
        "tokenizer": tokenizer,
        "processor": processor
    }
