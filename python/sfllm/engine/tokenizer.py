"""
This file contains the tokenizer class for the model.
It handles the tokenization of prompts and messages for the model.
"""
import torch
from transformers import AutoTokenizer
from tokenizers.decoders import DecodeStream


class IncrementalDetokenizer:
    """Decode generated token deltas without splitting UTF-8 characters."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.stream = DecodeStream(skip_special_tokens=True)
        self.token_ids = []
        self.text_offset = 0

    def append(self, token_ids, finished=False):
        self.token_ids.extend(token_ids)
        if finished:
            # Decode once in full to flush any bytes left at the token limit.
            return self.tokenizer.decode(
                self.token_ids, skip_special_tokens=True
            )[self.text_offset :]
        delta = self.stream.step(self.tokenizer.backend_tokenizer, token_ids) or ""
        self.text_offset += len(delta)
        return delta


class Tokenizer:
    def __init__(self, model_name: str):
        """
        Initialize the Tokenizer with the model path.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)


    def tokenize(self, prompt, messages=None):
        """
        Tokenize the prompt and messages for the model.
        
        Args:
            prompt: The prompt to tokenize
            messages: The messages to tokenize
            
        Returns:
            The tokenized inputs
        """
        if prompt is None:
            prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False,
                return_dict=True, return_tensors="pt"
            )
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        return inputs

    def detokenize(self, tokens):
        """
        Detokenize the tokens to get the original text.
        
        Args:
            tokens: The tokens to detokenize
            
        Returns:
            The detokenized text
        """
        return self.tokenizer.decode(tokens, skip_special_tokens=True)