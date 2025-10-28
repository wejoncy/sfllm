import torch
import bisect
from typing import Dict, List, Any

from sfllm.engine.forward_params import ForwardBatch,ForwardMode
from sfllm.utils.nutils import DEFAULT_CUDA_GRAPH_BATCH_SIZES, MAX_PROCESSED_TOKENS
from sfllm.layers.sampler import Sampler, SamplingBatchInfo

class ScheduleBatch:
    def __init__(self, sequences, mem_pool):
        self.sequences = sequences
        self.device = torch.device("cuda:0")
        self.mem_pool = mem_pool
        self.forward_metadata = ForwardBatch(mem_pool)


    def empty(self):
        return len(self.sequences) == 0

    def extend(self, sequences):
        self.sequences.extend(sequences)

    def __iter__(self):
        return iter(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx]
    
    def __len__(self) -> int:
        return len(self.sequences)

    def prepare_inputs(self):
        cur_seq_lens_list = []
        input_ids_list = []
        position_ids_list = []
        cache_loc_ids_list = []
        kv_indices_list = []
        prefix_lens_list = []
        device = self.device

        batch_size = len(self.sequences)
        if len(self.sequences[-1].new_tokens) == len(self.sequences[-1].tokens):
            self.forward_metadata.forward_mode = ForwardMode.EXTEND
        else:
            self.forward_metadata.forward_mode = ForwardMode.DECODE

        padded_batch_size = batch_size
        padded_token = 0
        if (self.forward_metadata.forward_mode == ForwardMode.DECODE):
            padded_batch_size = DEFAULT_CUDA_GRAPH_BATCH_SIZES[
                bisect.bisect_left(DEFAULT_CUDA_GRAPH_BATCH_SIZES, batch_size)
            ]
            padded_token = padded_batch_size - batch_size

        for sequence in self.sequences:
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
    
        input_ids = torch.tensor(input_ids_list, dtype=torch.long, pin_memory=True).to(device, non_blocking=True)
        position_ids = torch.tensor(
            position_ids_list, dtype=torch.long, pin_memory=True).to(device, non_blocking=True)
        cur_seq_lens = torch.tensor(cur_seq_lens_list, dtype=torch.int32, pin_memory=True).to(device, non_blocking=True)
        cache_loc_ids = torch.tensor(cache_loc_ids_list, dtype=torch.int64, pin_memory=True).to(device, non_blocking=True)
        kv_indices = torch.tensor(kv_indices_list, dtype=torch.int64, pin_memory=True).to(device, non_blocking=True)

        prefix_lens = torch.tensor(prefix_lens_list, dtype=torch.int32, pin_memory=True).to(device, non_blocking=True)

        total_seq_len = kv_indices.shape[0]
        if self.forward_metadata.forward_mode == ForwardMode.EXTEND:
            self.forward_metadata.max_extend_len = max(cur_seq_lens_list)
            self.forward_metadata.kv_indptr = prefix_lens.cumsum(dim=-1)
            self.forward_metadata.kv_indices = kv_indices
            self.forward_metadata.qo_indptr = cur_seq_lens.cumsum(dim=-1)
        else:
            self.forward_metadata.kv_indptr = (prefix_lens + 1).cumsum(dim=-1)
            self.forward_metadata.kv_indices = kv_indices
        self.forward_metadata.padded_token = padded_token
        self.forward_metadata.out_cache_loc = cache_loc_ids
        self.input_ids = input_ids
        self.position_ids = position_ids

    def prepare_sample(self):
        temperatures = []
        top_ps = []
        top_ks = []
        device = self.device

        for seq in self.sequences:
            temperatures.append(seq.sampling_params.temperature)
            top_ps.append(seq.sampling_params.top_p)
            top_ks.append(seq.sampling_params.top_k)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).to(device, non_blocking=True)
        top_ks = torch.tensor(top_ks, dtype=torch.int32, pin_memory=True).to(device, non_blocking=True)
        top_ps = torch.tensor(top_ps, dtype=torch.float32, pin_memory=True).to(device, non_blocking=True)
        self.forward_metadata.sampling_batch_info = SamplingBatchInfo(
            temperatures=temperatures,
            top_ps=top_ps,
            top_ks=top_ks,
            min_ps=torch.zeros_like(top_ps),
            is_all_greedy=all(seq.sampling_params.is_greedy for seq in self.sequences),
        )