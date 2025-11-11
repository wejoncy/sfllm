import dataclasses
import torch
import itertools
import bisect
from typing import Dict, List, Any, Optional, Union
from contextlib import contextmanager

from sfllm.engine.forward_params import ForwardBatch,ForwardMode
from sfllm.utils.nutils import DEFAULT_CUDA_GRAPH_BATCH_SIZES, MAX_PROCESSED_TOKENS
from sfllm.layers.sampler import SamplingBatchInfo
from sfllm.server_args import get_global_server_args
from sfllm.spec_decoding.spec_common import SpecInput

ALIGN_EAGLE_WITH_SGLANG_ = True
class ScheduleBatch:
    def __init__(self, sequences, mem_pool, draft_mem_pool=None):
        self.sequences = sequences
        self.device = torch.device("cuda:0")
        self.mem_pool = mem_pool
        self.forward_batch = ForwardBatch(mem_pool)
        self.forward_batch_spec = ForwardBatch(draft_mem_pool) if draft_mem_pool is not None else None
        self.fake_ids = None
        self.input_ids = None
        self.position_ids = None
        self.copy_done = None
        self.next_token_ids = None
        self.spec_info: SpecInput = None


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

    def clear(self):
        self.sequences = []
    
    def merge(self, other:'ScheduleBatch'):
        self.sequences = list(
            {s.sequence_id: s for s in self.sequences + other.sequences}.values()
        )
    
    def filter(self) -> 'ScheduleBatch':
        # indices = [seq.is_done() for seq in self.sequences]
        filtered_seqs = [seq for seq in self.sequences if not seq.is_done()]
        self.sequences = filtered_seqs
        # indices_ = torch.tensor(indices, dtype=torch.bool, pin_memory=True).to(self.device, non_blocking=True)
        # output_ids = self.next_token_ids[indices_]
        # self.input_ids = output_ids

    def add_placeholder_token(self, future_limit: int):
        for seq in self.sequences:
            place_id = -(seq.sequence_id % future_limit)
            seq.new_tokens = [place_id]  # use negative id as placeholder for future token position
            seq.tokens.append(place_id)

    def fake_tokenid_indices(self, future_limit: int):
        fake_ids = [(i.sequence_id % future_limit) for i in self.sequences]
        fake_ids = torch.tensor(fake_ids, dtype=torch.int64, pin_memory=True).to(self.device, non_blocking=True)
        self.fake_ids = fake_ids
        return self.fake_ids

    @contextmanager
    def switch_spec_forward_batch(self):
        self.forward_batch, self.forward_batch_spec = self.forward_batch_spec, self.forward_batch
        yield
        self.forward_batch, self.forward_batch_spec = self.forward_batch_spec, self.forward_batch

    def get_seq_groups_hash(self):
        group_hash = 0
        for seq in self.sequences:
            x = seq.sequence_id * 0x9e3779b97f4a7c15
            group_hash ^= (x ^ (x >> 32)) & 0xFFFFFFFFFFFFFFFF
        return group_hash

    def prepare_prefill_for_draft(self):
        spec_out_cache_loc_list = []
        spec_kv_indices_list = []
        device = self.device
        for sequence in self.sequences:
            spec_out_cache_loc_list.extend(sequence.out_cache_loc_spec[-len(sequence.new_tokens):])
            spec_kv_indices_list.extend(sequence.out_cache_loc_spec)

        padded_token = self.forward_batch.padded_token
        if padded_token > 0:
            spec_out_cache_loc_list.extend([0] * padded_token)
            spec_kv_indices_list.extend([0] * padded_token)

        out_cache_loc_spec = torch.tensor(spec_out_cache_loc_list, dtype=torch.int64, pin_memory=True).to(device, non_blocking=True)
        kv_indices_spec = torch.tensor(spec_kv_indices_list, dtype=torch.int64, pin_memory=True).to(device, non_blocking=True)

        self.forward_batch_spec.max_extend_len = self.forward_batch.max_extend_len
        self.forward_batch_spec.kv_indptr = self.forward_batch.kv_indptr
        self.forward_batch_spec.kv_indices = kv_indices_spec
        self.forward_batch_spec.qo_indptr = self.forward_batch.qo_indptr
        # self.forward_batch_spec.seq_lens = prefix_lens
        self.forward_batch_spec.padded_token = padded_token
        self.forward_batch_spec.out_cache_loc = out_cache_loc_spec

        self.spec_info = SpecInput()
        self.spec_info.hash = self.get_seq_groups_hash()


    def prepare_decode_for_draft(self, position_ids_list: List[int]):
        g_hash = self.get_seq_groups_hash()
        if self.spec_info.hash != g_hash:
            self.spec_info = self.spec_info.raw_new()
            self.spec_info.hash = g_hash
            self.spec_info.verified_id = torch.cat([seq.verified_id for seq in self.sequences], dim=-1)
            self.spec_info.accept_length = torch.cat([seq.accept_length for seq in self.sequences], dim=-1)
            self.spec_info.accept_length_cpu = self.spec_info.accept_length.cpu()
            self.spec_info.hidden_states = torch.cat([seq.hidden_states for seq in self.sequences])
            self.spec_info.logits = torch.cat([seq.logits for seq in self.sequences])

        # prepare position_ids for draft model extend for last verified tokens
        positions_outs = []
        batch_size = len(self.sequences)
        server_args = get_global_server_args()
        update_accept_length_cpu = self.spec_info.accept_length_cpu + 1
        for i in range(len(position_ids_list)):
            positions_outs.append(range(
                position_ids_list[i] - update_accept_length_cpu[i],
                position_ids_list[i]
            ))
        position_ids_list = list(itertools.chain.from_iterable(positions_outs))

        spec_out_cache_loc_list = []
        spec_kv_indices_list = []
        spec_kv_indices_mtd_list = []
        device = self.device
        for sequence in self.sequences:
            total_draft_len = sequence.accept_length_cpu[0] + 1
            spec_out_cache_loc_list.extend(sequence.out_cache_loc_spec[-total_draft_len:])
            total_draft_len_past = 0 if ALIGN_EAGLE_WITH_SGLANG_ else total_draft_len
            if total_draft_len_past == 0:
                total_draft_len_past = -len(sequence.out_cache_loc_spec)
            spec_kv_indices_list.extend(sequence.out_cache_loc_spec[:-total_draft_len_past]) # actually, it's for extend. it's weird here.
            spec_kv_indices_mtd_list.extend(sequence.out_cache_loc_spec)

        padded_token = self.forward_batch.padded_token
        if padded_token > 0:
            spec_out_cache_loc_list.extend([0] * padded_token)
            spec_kv_indices_list.extend([0] * padded_token)
        out_cache_loc_spec = torch.tensor(spec_out_cache_loc_list, dtype=torch.int64, pin_memory=True).to(device, non_blocking=True)
        kv_indices_spec = torch.tensor(spec_kv_indices_list, dtype=torch.int64, pin_memory=True).to(device, non_blocking=True)
        kv_indices_mtd_spec = torch.tensor(spec_kv_indices_mtd_list, dtype=torch.int64, pin_memory=True).to(device, non_blocking=True)

        #kv_indptr would be used in two place, extend forward for the latest accepted token,, the other is multi-step draft decode path
        minux_const = torch.arange(1, len(self.sequences)+1, dtype=torch.int32, 
                        pin_memory=True).add_(
                        0 if ALIGN_EAGLE_WITH_SGLANG_ else (self.spec_info.accept_length_cpu + 1))
        # sglang dropped the first token
        # leading zero
        self.forward_batch_spec.kv_indptr = self.forward_batch.kv_indptr.clone()
        self.forward_batch_spec.kv_indptr[1:].sub_(minux_const.to(self.device, non_blocking=True))
        self.forward_batch_spec.kv_indices = kv_indices_spec
        self.forward_batch_spec.kv_indices_mtd = kv_indices_mtd_spec
        self.forward_batch_spec.padded_token = padded_token
        self.forward_batch_spec.out_cache_loc = out_cache_loc_spec
        self.forward_batch_spec.qo_indptr = self.forward_batch.kv_indptr.clone()
        self.forward_batch_spec.qo_indptr[1:batch_size + 1] = (1 + self.spec_info.accept_length).cumsum(dim=0, dtype=torch.int32)
        self.forward_batch_spec.max_extend_len = max(self.spec_info.accept_length_cpu.tolist())+1
        self.forward_batch_spec.position_ids_extend = torch.tensor(
            position_ids_list, dtype=torch.long, pin_memory=True).to(self.device, non_blocking=True)

        # for this step, target model will go the verify path
        forward_batch_target = self.forward_batch
        num_draft_tokens = server_args.speculative_num_draft_tokens
        seq_mask_len = num_draft_tokens * (forward_batch_target.seq_lens + num_draft_tokens)
        forward_batch_target.mask_indptr = self.forward_batch.kv_indptr.clone()
        forward_batch_target.mask_indptr[1:] = torch.cumsum(seq_mask_len, dim=0, dtype=torch.int32)
        forward_batch_target.max_extend_len = num_draft_tokens

        forward_batch_target.qo_indptr = torch.arange(
            0, (1 + batch_size) * num_draft_tokens, step=num_draft_tokens,
            dtype=torch.int32, device=device,
        )
        # it's for verify_extend
        minux_const = torch.arange(1, len(self.sequences)+1, dtype=torch.int32, device=device)
        forward_batch_target.kv_indptr[1:].sub_(minux_const)

    def prepare_inputs(self):
        cur_seq_lens_list = [0]
        input_ids_list = []
        position_ids_list = []
        out_cache_loc_list = []
        kv_indices_list = []
        prefix_lens_list = [0]
        device = self.device

        batch_size = len(self.sequences)
        if len(self.sequences[-1].new_tokens) == len(self.sequences[-1].tokens):
            self.forward_batch.forward_mode = ForwardMode.EXTEND
        else:
            self.forward_batch.forward_mode = ForwardMode.DECODE

        padded_batch_size = batch_size
        padded_token = 0
        if (self.forward_batch.forward_mode == ForwardMode.DECODE and self.forward_batch_spec is None):
            padded_batch_size = DEFAULT_CUDA_GRAPH_BATCH_SIZES[
                bisect.bisect_left(DEFAULT_CUDA_GRAPH_BATCH_SIZES, batch_size)
            ]
            padded_token = padded_batch_size - batch_size

        for sequence in self.sequences:
            if self.forward_batch.forward_mode == ForwardMode.DECODE:
                # it's posible to decode multiple tokens at once when doing speculative decoding
                cur_seq_lens_list.append(1)
                input_ids_list.append(sequence.new_tokens[-1])
                start_pos = len(sequence.tokens) - 1
            else:
                cur_seq_lens_list.append(len(sequence.new_tokens))
                input_ids_list.extend(sequence.new_tokens)
                start_pos = len(sequence.tokens) - len(sequence.new_tokens)
            position_ids_list.extend(list(range(start_pos, start_pos+cur_seq_lens_list[-1])))
            prefix_lens_list.append(start_pos)
            if self.forward_batch_spec is not None and self.forward_batch.forward_mode == ForwardMode.DECODE:
                # target model used for verify, speculative_num_draft_tokens cache loc
                out_cache_loc_list.extend(sequence.out_cache_loc[len(sequence.tokens)-1:])
                kv_indices_list.extend(sequence.out_cache_loc[:len(sequence.tokens)-1])
            else:
                out_cache_loc_list.extend(sequence.out_cache_loc[-len(sequence.new_tokens):])
                kv_indices_list.extend(sequence.out_cache_loc)

        if padded_token > 0:
            input_ids_list.extend([0]*padded_token)
            position_ids_list.extend([0]*padded_token)
            cur_seq_lens_list.extend([1]*padded_token)
            out_cache_loc_list.extend([0]*padded_token)
            kv_indices_list.extend([0]*padded_token)
            prefix_lens_list.extend([0]*padded_token)
    
        input_ids = torch.tensor(input_ids_list, dtype=torch.long, pin_memory=True).to(device, non_blocking=True)
        position_ids = torch.tensor(position_ids_list, dtype=torch.long, pin_memory=True).to(device, non_blocking=True)
        cur_seq_lens = torch.tensor(cur_seq_lens_list, dtype=torch.int32, pin_memory=True).to(device, non_blocking=True)
        out_cache_loc = torch.tensor(out_cache_loc_list, dtype=torch.int64, pin_memory=True).to(device, non_blocking=True)
        kv_indices = torch.tensor(kv_indices_list, dtype=torch.int64, pin_memory=True).to(device, non_blocking=True)

        prefix_lens = torch.tensor(prefix_lens_list, dtype=torch.int32, pin_memory=True).to(device, non_blocking=True)

        total_seq_len = kv_indices.shape[0] # this would be wrong if speculative decoding with draft model
        if self.forward_batch.forward_mode == ForwardMode.EXTEND:
            self.forward_batch.max_extend_len = max(cur_seq_lens_list)
            self.forward_batch.kv_indptr = prefix_lens.cumsum(dim=-1, dtype=torch.int32)
            self.forward_batch.kv_indices = kv_indices
            self.forward_batch.qo_indptr = cur_seq_lens.cumsum(dim=-1, dtype=torch.int32)
        else:
            self.forward_batch.kv_indptr = prefix_lens.clone()
            self.forward_batch.kv_indptr[1:] = (prefix_lens[1:] + 1).cumsum(dim=-1, dtype=torch.int32)
            self.forward_batch.kv_indices = kv_indices

        self.forward_batch.seq_lens = prefix_lens[1:]
        self.forward_batch.seq_lens_sum = sum(prefix_lens_list)
        self.forward_batch.extend_lens_list = cur_seq_lens_list[1:]
        self.forward_batch.padded_token = padded_token
        self.forward_batch.out_cache_loc = out_cache_loc
        self.input_ids = input_ids
        self.position_ids = position_ids

        if self.forward_batch_spec is not None:
            if self.forward_batch.forward_mode == ForwardMode.EXTEND:
                self.prepare_prefill_for_draft()
            else:
                self.prepare_decode_for_draft(position_ids_list)

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
        self.forward_batch.sampling_batch_info = SamplingBatchInfo(
            temperatures=temperatures,
            top_ps=top_ps,
            top_ks=top_ks,
            min_ps=torch.zeros_like(top_ps),
            is_all_greedy=all(seq.sampling_params.is_greedy for seq in self.sequences),
        )

@dataclasses.dataclass
class BatchResult:
    next_token_ids: torch.Tensor
    next_token_logits: torch.Tensor
    aux_hidden_states: Optional[torch.Tensor] = None
    spec_info: SpecInput = None

@dataclasses.dataclass
class LogitsProcessorOutput:
    ## Part 1: This part will be assigned in python/sglang/srt/layers/logits_processor.py::LogitsProcessor
    # The logits of the next tokens.       shape: [#seq, vocab_size]
    # Can be None for certain prefill-only requests (e.g., multi-item scoring) that don't need next token generation
    next_token_logits: Optional[torch.Tensor]
    # Used by speculative decoding (EAGLE)
    # The last hidden layers
    hidden_states: Optional[torch.Tensor] = None

    ## Part 2: This part will be assigned in python/sglang/srt/layers/sampler.py::Sampler
    # he log probs of output tokens, if SGLANG_RETURN_ORIGINAL_LOGPROB = True, 
    # will get the log probs before applying temperature. 
    # If False, will get the log probs before applying temperature.
    next_token_logprobs: Optional[torch.Tensor] = None
    # The logprobs and ids of the top-k tokens in output positions. shape: [#seq, k]
    next_token_top_logprobs_val: Optional[List] = None
    next_token_top_logprobs_idx: Optional[List] = None
    # The logprobs and ids of the requested token ids in output positions. 
    # shape: [#seq, n] (n is the number of requested token ids)
    # Can contain either lists or GPU tensors (for delayed copy optimization in prefill-only requests)
    next_token_token_ids_logprobs_val: Optional[
        List[Union[List[float], torch.Tensor]]
    ] = None
    next_token_token_ids_logprobs_idx: Optional[List] = None

    ## Part 3: Prefill-only. This part will be assigned in python/sglang/srt/layers/logits_processor.py::LogitsProcessor
    # The logprobs of input tokens.        shape: [#token]
    input_token_logprobs: Optional[torch.Tensor] = None
    # The logprobs and ids of the top-k tokens in input positions.  shape: [#seq, #token, k]
    input_top_logprobs_val: List = None
    input_top_logprobs_idx: List = None
    # The logprobs and ids of the requested token ids in input positions. 
    # shape: [#seq, n] (n is the number of requested token ids)
    # Can contain either lists or GPU tensors (for delayed GPU-to-CPU transfer optimization)
    input_token_ids_logprobs_val: Optional[List[Union[List[float], torch.Tensor]]] = (
        None
    )
    input_token_ids_logprobs_idx: Optional[List] = None
