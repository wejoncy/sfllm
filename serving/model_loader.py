from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from .tokenizer import Tokenizer


MODEL_PATH = "/root/work/gemma-3-4b-it"

class ForwardModel:
    def __init__(self, model_name=MODEL_PATH):
        """
        Initialize the ForwardModel with the model name or path.
        
        Args:
            model_name: The name or path of the model to load
        """
        self.model = None
        self.tokenizer = None
        
        # Load the model and tokenizer
        self.load_model(model_name)

    def load_model(self, model_name=MODEL_PATH):
        """
        Load the model and tokenizer
        
        Args:
            model_name: The name or path of the model to load
            
        Returns:
            A dictionary containing model, tokenizer, and processor
        """
        print(f"Loading model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
        )
        self.model = self.model.eval()
        # Try loading a processor for vision models
        try:
            self.processor = AutoProcessor.from_pretrained(model_name)
        except:
            self.processor = None
        self.tokenizer = Tokenizer(model_name)

    def tokenize(self, prompt, messages=None):
        """
        Tokenize the prompt and messages for the model.
        
        Args:
            prompt: The prompt to tokenize
            messages: The messages to tokenize
            
        Returns:
            The tokenized inputs
        """
        return self.tokenizer.tokenize(prompt, messages)


    def detokenize(self, tokens):
        """
        Detokenize the tokens to get the original text.
        
        Args:
            tokens: The tokens to detokenize
            
        Returns:
            The detokenized text
        """
        return self.tokenizer.detokenize(tokens)