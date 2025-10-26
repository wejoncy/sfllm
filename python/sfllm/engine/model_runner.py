import logging
import torch
import bisect
import tqdm
from typing import Dict, List, Any

from sfllm.engine.model_loader import ForwardModel
from sfllm.engine.shedule_batch import ScheduleBatch
from sfllm.engine.forward_params import ForwardMode, ForwardBatch
from sfllm.layers.sampler import Sampler, SamplingBatchInfo
from sfllm.server_args import ServerArgs


logger = logging.getLogger(__name__)
DEFAULT_CUDA_GRAPH_BATCH_SIZES = [1, 2, 3, 4, 6, 8, 10, 12, 14, 16]+list(range(20, 2048+1, 4))

class ModelRunner:
    def __init__(self, server_args: ServerArgs, device_id: int = 0, mem_pool=None):
        self.model = ForwardModel(server_args.model_path, server_args.dtype)
        server_args.model_config = self.model.config
        self.device_id = device_id
        self.compute_stream = torch.cuda.Stream(device=device_id)
        self.cuda_graph_max_bs = server_args.cuda_graph_max_bs
        ind = bisect.bisect_right(DEFAULT_CUDA_GRAPH_BATCH_SIZES, self.cuda_graph_max_bs)
        self.capture_batch_size = DEFAULT_CUDA_GRAPH_BATCH_SIZES[:ind]
        self.capture_batch_size = self.capture_batch_size[:ind]
        self.sampler = Sampler(self.model.config)
        self.rank = 0
        self.server_args = server_args

        # cuda graph
        self.input_ids = torch.empty((max(self.capture_batch_size),), dtype=torch.long, device=self.device_id)
        self.position_ids = torch.empty((max(self.capture_batch_size),), dtype=torch.long, device=self.device_id)
        self.out_cache_loc = torch.zeros((max(self.capture_batch_size),), dtype=torch.int64, device=self.device_id)
        self.output_logits = {}
        self.cuda_graphs = {}
        self.graph_pool = torch.cuda.graph_pool_handle()
        self.forward_metadata = ForwardBatch(self.model.config, self.server_args.dtype)

    def capture_graph(self):
        if self.server_args.disable_cuda_graph is False:
            self.capture_graph()


    def set_mem_pool(self, mem_pool):
        self.forward_metadata.past_key_values = mem_pool

    def get_max_context_length(self):
        return self.model.config.max_position_embeddings
    
    def prepare_inputs(self, sequence_group: ScheduleBatch) -> Dict[str, Any]:
        cur_seq_lens_list = []
        input_ids_list = []
        position_ids_list = []
        cache_loc_ids_list = []
        kv_indices_list = []
        prefix_lens_list = []
        batch_size = len(sequence_group)
        if len(sequence_group[-1].new_tokens) == len(sequence_group[-1].tokens):
            self.forward_metadata.forward_mode = ForwardMode.EXTEND
        else:
            self.forward_metadata.forward_mode = ForwardMode.DECODE

        padded_batch_size = batch_size
        padded_token = 0
        if (
            # self.server_args.disable_cuda_graph and 
            self.forward_metadata.forward_mode == ForwardMode.DECODE
            and batch_size < self.cuda_graph_max_bs
        ):
            padded_batch_size = self.capture_batch_size[
                bisect.bisect_left(self.capture_batch_size, batch_size)
            ]
            padded_token = padded_batch_size - batch_size

        for sequence in sequence_group:
            cur_seq_lens_list.append(len(sequence.new_tokens))
            input_ids_list.extend(sequence.new_tokens)
            start_pos = len(sequence.tokens) - len(sequence.new_tokens)
            position_ids_list.extend(
                list(range(start_pos, start_pos+cur_seq_lens_list[-1]))
            )
            prefix_lens_list.append(start_pos)
            cache_loc_ids_list.extend(sequence.cache_loc_ids[-len(sequence.new_tokens):])
            kv_indices_list.extend(sequence.cache_loc_ids)

        if padded_token > 0:
            input_ids_list.extend([0]*padded_token)
            position_ids_list.extend([0]*padded_token)
            cur_seq_lens_list.extend([1]*padded_token)
            cache_loc_ids_list.extend([0]*padded_token)
            kv_indices_list.extend([0]*padded_token)
            prefix_lens_list.extend([0]*padded_token)

        input_ids = torch.tensor(
            input_ids_list, dtype=torch.long, device=self.device_id
        )
        position_ids = torch.tensor(
            position_ids_list, dtype=torch.long, device=self.device_id
        )
        cur_seq_lens = torch.tensor(cur_seq_lens_list, dtype=torch.int32)
        cache_loc_ids = torch.tensor(cache_loc_ids_list, dtype=torch.int64)
        kv_indices = torch.tensor(kv_indices_list, dtype=torch.int64)
        prefix_lens = torch.tensor(prefix_lens_list, dtype=torch.int32)

        total_seq_len = kv_indices.shape[0]
        if self.forward_metadata.forward_mode == ForwardMode.EXTEND:
            self.forward_metadata.max_extend_len = max(cur_seq_lens_list)
            self.forward_metadata.kv_indptr = self.forward_metadata.kv_indptr_buffer[
                : batch_size + 1
            ]
            self.forward_metadata.kv_indptr[1:].copy_(
                prefix_lens.cumsum(dim=-1), non_blocking=True
            )

            self.forward_metadata.kv_indices = self.forward_metadata.kv_indices_buffer[
                :total_seq_len
            ]
            self.forward_metadata.kv_indices.copy_(kv_indices, non_blocking=True)

            self.forward_metadata.qo_indptr = self.forward_metadata.qo_indptr_buffer[
                : batch_size + 1
            ]
            self.forward_metadata.qo_indptr[1:].copy_(cur_seq_lens.cumsum(dim=-1), non_blocking=True)
        else:
            self.forward_metadata.kv_indptr = self.forward_metadata.kv_indptr_buffer[
                : padded_batch_size + 1
            ]
            self.forward_metadata.kv_indptr[1:].copy_((prefix_lens+1).cumsum(dim=-1), non_blocking=True)
            
            self.forward_metadata.kv_indices = self.forward_metadata.kv_indices_buffer[
                :total_seq_len
            ]
            self.forward_metadata.kv_indices.copy_(kv_indices, non_blocking=True)
            self.forward_metadata.num_kv_splits = (
                self.forward_metadata.num_kv_splits_buffer[:padded_batch_size]
            )
        self.forward_metadata.padded_token = padded_token

        self.forward_metadata.out_cache_loc = cache_loc_ids.to(self.forward_metadata.kv_indices.device, non_blocking=True)

        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            # "attention_mask": attention_mask,
            "forward_metadata": self.forward_metadata,
        }

    def prepare_sample(self, seqs: ScheduleBatch) -> torch.Tensor:
        # if self.forward_metadata.sampling_batch_info is not None:
        #     return
        temperatures = []
        top_ps = []
        top_ks = []
        for seq in seqs:
            temperatures.append(seq.sampling_params.temperature)
            top_ps.append(seq.sampling_params.top_p)
            top_ks.append(seq.sampling_params.top_k)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        top_ks = torch.tensor(top_ks, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        top_ps = torch.tensor(top_ps, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        self.forward_metadata.sampling_batch_info = SamplingBatchInfo(
            temperatures=temperatures,
            top_ps=top_ps,
            top_ks=top_ks,
            min_ps=torch.zeros_like(top_ps),
            is_all_greedy=all(seq.sampling_params.is_greedy for seq in seqs),
        )

    def prepare_replay(self, inputs: Dict[str, Any], sequence_group: ScheduleBatch):
        bs_size = len(sequence_group)
        padded_bs_size = bs_size + self.forward_metadata.padded_token
        self.input_ids[:padded_bs_size].copy_(inputs["input_ids"], non_blocking=True)
        self.position_ids[:padded_bs_size].copy_(
            inputs["position_ids"], non_blocking=True
        )
        self.out_cache_loc[:padded_bs_size].copy_(
            self.forward_metadata.out_cache_loc, non_blocking=True
        )
        # self.attention_mask[:bs_size].copy_(inputs["attention_mask"], non_blocking=True)


    def forward(self, sequence_group: ScheduleBatch):
        inputs = self.prepare_inputs(sequence_group)
        self.prepare_sample(sequence_group) if self.rank == 0 else None
        bs_size = len(sequence_group)
        pad_bs_size = self.forward_metadata.padded_token + bs_size
        if (
            self.forward_metadata.forward_mode == ForwardMode.DECODE
            and self.cuda_graphs.get(pad_bs_size) is not None
        ):
            self.prepare_replay(inputs, sequence_group)
            self.cuda_graphs[pad_bs_size].replay()
            logits = self.output_logits[pad_bs_size][:bs_size]
        else:
            logits = self.model(**inputs)[:bs_size]

        token_ids = (
            self.sampler(logits, self.forward_metadata.sampling_batch_info) if self.rank == 0 else None
        )
        return token_ids

    def capture_graph(self):
        batch_size = 1
        input_ids = self.input_ids[:batch_size]
        self.forward_metadata.kv_indptr = self.forward_metadata.kv_indptr_buffer[: batch_size + 1]
        self.forward_metadata.kv_indices = self.forward_metadata.kv_indices_buffer[
            : input_ids.shape[0]
        ]
        self.forward_metadata.num_kv_splits = self.forward_metadata.num_kv_splits_buffer[:batch_size]
        self.forward_metadata.forward_mode = ForwardMode.DECODE
        self.forward_metadata.out_cache_loc = self.forward_metadata.kv_indices
        self.compute_stream.synchronize()

        with torch.no_grad():
            self.model(
                self.input_ids[:batch_size],
                position_ids=self.position_ids[:batch_size],
                forward_metadata=self.forward_metadata,
            )

        for batch_size in tqdm.tqdm(self.capture_batch_size, desc="Capturing CUDA Graphs"):
            self.forward_metadata.kv_indptr = self.forward_metadata.kv_indptr_buffer[: batch_size + 1]
            self.forward_metadata.out_cache_loc = self.out_cache_loc[:batch_size]
            torch.cuda.synchronize()
            cudagraph = torch.cuda.CUDAGraph()
            # attention_mask = torch.empty((batch_size), dtype=torch.long, device=self.device_id)
            with torch.cuda.graph(cudagraph, stream=self.compute_stream, pool=self.graph_pool):
                output = self.model(
                    self.input_ids[:batch_size],
                    position_ids=self.position_ids[:batch_size],
                    forward_metadata=self.forward_metadata,
                )
            torch.cuda.synchronize()
            self.output_logits[batch_size] = output
            self.cuda_graphs[batch_size] = cudagraph
            
        self.cuda_graphs[1].replay()
    
    def tokenize(self, prompt):
        return self.model.tokenizer.encode(prompt)


    def detokenize(self, tokens):
        return self.model.tokenizer.decode(
            tokens, skip_special_tokens=True, spaces_between_special_tokens=True
        )
