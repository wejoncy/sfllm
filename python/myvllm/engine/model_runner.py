import logging
import torch
import asyncio
from typing import Dict, List, Any, Optional

from myvllm.engine.model_loader import TorchDefaultDtype, _load_check_point
from myvllm.models.modeling_qwen3 import Qwen3ForCausalLM

logger = logging.getLogger(__name__)


class ForwardMode(IntEnum):
    # Extend a sequence. The KV cache of the beginning part of the sequence is already computed (e.g., system prompt).
    # It is also called "prefill" in common terminology.
    EXTEND = auto()
    # Decode one token.
    DECODE = auto()
    # Contains both EXTEND and DECODE when doing chunked prefill.
    MIXED = auto()
    # No sequence to forward. For data parallel attention, some workers will be IDLE if no sequence are allocated.
    IDLE = auto()

class ForwardMetaData:
    def __init__(self, config):
        self.attn_logits = torch.empty((12800, config.num_attention_heads, 
                                        8, config.head_dim), dtype=torch.float32, device="cuda")
        self.attn_lse = torch.empty((12800, config.num_attention_heads, 8),dtype=torch.float32,device="cuda")
        # need to inilialize during prepare inputs
        self.max_extend_len = 0
        self.num_kv_splits_buffer = torch.zeros((128, config.num_attention_heads), dtype=torch.int32, device="cuda")
        self.num_kv_splits = None
        self.kv_indptr_buffer = torch.zeros((128,), dtype=torch.int32, device="cuda")
        self.kv_indptr = None
        self.kv_indices_buffer = torch.zeros((128,), dtype=torch.int64, device="cuda")
        self.kv_indices = None
        self.qo_indptr_buffer = torch.zeros((128,), dtype=torch.int32, device="cuda")
        self.qo_indptr = None
        self.custom_mask = None
        self.mask_indptr = None

        self.past_key_values = self.create_past_kv(config)
        self.seq_length = 0
        self.forward_mode = ForwardMode.EXTEND

    def create_past_kv(self, config, max_length=1024):
        past_key_values = []
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        n_heads = config.num_key_value_heads
        free, total = torch.cuda.mem_get_info("cuda:0")
        logger.info(f"GPU memory free: {free / (1024**3):.2f} GB, total: {total / (1024**3):.2f} GB")
        one_token_size = n_heads * dim * 2 * 2  # key + value, float16
        max_length = min(max_length, free // one_token_size // config.num_hidden_layers)
        for _ in range(config.num_hidden_layers):
            past_key_values.append((torch.zeros(max_length, n_heads, dim).cuda(),
                                    torch.zeros(max_length, n_heads, dim).cuda()))
        return past_key_values

    def get_seq_length(self):
        return self.seq_length

    def update(self, key_states, value_states, layer_idx, cache_kwargs):
        bsz, seq_len, kv_heads, head_dim = key_states.size()
        past_key, past_value = self.past_key_values[layer_idx]
        past_key[self.seq_length:self.seq_length + seq_len, ...] = key_states[0]
        past_value[self.seq_length:self.seq_length + seq_len, ...] = value_states[0]
        self.seq_length += seq_len
        updated_key = past_key[:self.seq_length, :, :][None]
        updated_value = past_value[:self.seq_length, :, :][None]
        return updated_key, updated_value

class ModelRunner:
    def __init__(self, model, device_id: int=0):
        self.model = model
        self.device_id = device_id
        self.stream = torch.cuda.Stream(device=device_id)
        self.graph = None
        self.forward_metadata = None
        self.capture_batch_size = [1, 2, 3, 4, 8, 16]

        # cuda graph
        self.input_ids = torch.empty((max(self.capture_batch_size)), dtype=torch.long, device=self.device_id)
        self.output_logits = {}
        self.cuda_graphs = {}
        self.graph_pool = torch.cuda.graph_pool_handle()
        # self.attention_mask = torch.empty((max(self.capture_batch_size), 1), dtype=torch.long, device=self.device_id)

    def alloc_kv_cache(self, config):
        self.forward_metadata = ForwardMetaData(config)
    
    def prepare_inputs(self, input_ids, attention_mask):
        seq_len = input_ids.shape
        bsz = 1
        if self.forward_metadata.forward_mode == ForwardMode.EXTEND:
            self.forward_metadata.max_extend_len = seq_len
            self.forward_metadata.kv_indptr = self.forward_metadata.kv_indptr_buffer[:bsz + 1].mul_(0)
            self.forward_metadata.kv_indices = self.forward_metadata.kv_indices_buffer[:seq_len].mul_(0)
            self.forward_metadata.qo_indptr = self.forward_metadata.qo_indptr_buffer[:bsz + 1].mul_(0)
        else:
            self.forward_metadata.kv_indptr = self.forward_metadata.kv_indptr_buffer[:seq_len + 1]
            self.forward_metadata.kv_indices = self.forward_metadata.kv_indices_buffer[:seq_len]
            self.forward_metadata.qo_indptr = self.forward_metadata.qo_indptr_buffer[:seq_len + 1]
            self.forward_metadata.num_kv_splits = self.forward_metadata.num_kv_splits_buffer[:seq_len, :]


        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'past_key_values': self.forward_metadata
        }

    def forward(self, inputs):
        with torch.cuda.stream(self.stream):
            outputs = self.model(**inputs)
        return outputs

    def capture_graph(self):
        self.stream.synchronize()
        with torch.no_grad():
            model(self.input_ids, past_key_values=self.past_key_values)

        for batch_size in self.capture_batch_size:
            torch.cuda.synchronize()
            cudagraph = torch.cuda.CUDAGraph()
            # attention_mask = torch.empty((batch_size), dtype=torch.long, device=self.device_id)
            with torch.cuda.graph(cudagraph, stream=self.stream, pool=self.graph_pool):
                output = self.model(self.input_ids[:batch_size], past_key_values=self.past_key_values)
            torch.cuda.synchronize()
            self.output_logits[batch_size] = output
            self.cuda_graphs[batch_size] = cudagraph
            
        self.cuda_graphs[1].replay()

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


if __name__ == "__main__":
    import transformers
    model_path = r"D:\\work\\Qwen3-0.6B"

    config = transformers.AutoConfig.from_pretrained(model_path)
    forward_metadata = ForwardMetaData(config)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    with TorchDefaultDtype(config.dtype):
        model = Qwen3ForCausalLM(config).cuda()
        _load_check_point(model, model_path)
    model.eval()
    model_runner = ModelRunner(model)
    model_runner.capture_graph()