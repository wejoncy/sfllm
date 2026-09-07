import dataclasses
import numpy as np
import torch
import itertools
import bisect
from typing import List, Optional, Union
from contextlib import contextmanager

from sfllm.engine.forward_params import ForwardBatch,ForwardMode
from sfllm.engine.sequence import RequestSequence
from sfllm.utils.nutils import DEFAULT_CUDA_GRAPH_BATCH_SIZES
from sfllm.layers.sampler import SamplingBatchInfo
from sfllm.server_args import get_global_server_args
from sfllm.spec_decoding.spec_common import SpecInput


class ScheduleBatch:
    def __init__(self, sequences, mem_pool, draft_mem_pool=None):
        self.sequences:RequestSequence = sequences
        self.device = torch.device("cuda:0")
        self.forward_batch = ForwardBatch(mem_pool)
        self.forward_batch_spec = ForwardBatch(draft_mem_pool) if draft_mem_pool is not None else None
        self.input_ids:torch.Tensor = None
        self.position_ids:torch.Tensor = None
        self.copy_done: Optional[torch.cuda.Event] = None
        self.next_token_ids:torch.Tensor = None
        self.spec_info: SpecInput = None
        ###
        self.overlap_affiliated: Optional[tuple[torch.Tensor, torch.cuda.Stream]] = None


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
        self.spec_info = other.spec_info
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

    def add_placeholder_token(self, future_token_stride: int = 1):
        for seq in self.sequences:
            place_id = -((seq.request_index + 1) * future_token_stride)
            seq.new_tokens = list(reversed(range(place_id, place_id+future_token_stride))) # use negative id as placeholder for future token position
            seq.tokens.extend(seq.new_tokens)

    def fake_tokenid_indices(self, future_token_stride: int = 1):
        return torch.tensor(
            [
                seq.request_index * future_token_stride + offset
                for seq in self.sequences
                for offset in range(1, future_token_stride + 1)
            ],
            dtype=torch.int64, pin_memory=True, device="cpu",
        ).to(self.device, non_blocking=True)

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

    def update_spec_info_if_needed(self, hidden_states_buffer:torch.Tensor):
        if self.spec_info is None:
            return
        g_hash = self.get_seq_groups_hash()
        if self.spec_info.hash != g_hash:
            from sfllm.kernels.triton_utils import copy_tensors_to_buffer,move_neg1_to_tail
            self.spec_info = self.spec_info.raw_new()
            self.spec_info.hash = g_hash
            self.spec_info.verified_id = move_neg1_to_tail(torch.cat([
                seq.verified_id[:len(seq.new_tokens)] for seq in self.sequences
            ], dim=-1))
            self.spec_info.verified_id = torch.where(self.spec_info.verified_id < 0,torch.zeros_like(self.spec_info.verified_id),
                                                  self.spec_info.verified_id)
            self.spec_info.accept_length = torch.cat([seq.accept_length for seq in self.sequences], dim=-1)
            if hidden_states_buffer is not None:
                assert isinstance(self.sequences[0].hidden_states, tuple)
                src_hd_list = [ seq.hidden_states[0] for seq in self.sequences]
                src_range_list = [ seq.hidden_states[1] for seq in self.sequences]
                src_range = torch.stack(src_range_list, dim=0)
                draft_steps = hidden_states_buffer.shape[1]
                hidden_states_buffer = hidden_states_buffer.view(-1, hidden_states_buffer.shape[-1])
                self.spec_info.hidden_states = copy_tensors_to_buffer(src_hd_list, src_range, hidden_states_buffer)
                self.spec_info.hidden_states = self.spec_info.hidden_states[:len(src_hd_list)*draft_steps]
            else:
                self.spec_info.hidden_states = torch.cat([seq.hidden_states for seq in self.sequences], dim=0)
            cache_sources = [seq.out_cache_loc_lazy for seq in self.sequences
                             if seq.out_cache_loc_lazy is not None]
            if cache_sources:
                self.spec_info.out_cache_loc = copy_tensors_to_buffer(
                    [source[0] for source in cache_sources],
                    torch.stack([source[1] for source in cache_sources]),
                    cache_sources[0][0].new_empty((self.spec_info.verified_id.numel(), 1)),
                ).view(-1)
            # self.spec_info.logits = torch.cat([seq.logits for seq in self.sequences])
    def prepare_decode_for_draft(self, position_ids_list: List[int], is_overlap:bool=False):
        # prepare position_ids for draft model extend for last verified tokens
        positions_outs = []
        batch_size = len(self.sequences)
        server_args = get_global_server_args()
        for i in range(len(position_ids_list)):
            positions_outs.append(range(
                position_ids_list[i] - len(self.sequences[i].new_tokens),
                position_ids_list[i]
            ))
        position_ids_list = list(itertools.chain.from_iterable(positions_outs))

        spec_out_cache_loc_list = []
        spec_kv_indices_list = []
        spec_kv_indptr_list = [0]
        device = self.device
        for sequence in self.sequences:
            total_draft_len = len(sequence.new_tokens)
            if (len(sequence.tokens) == sequence.last_generated_token_pos
                    and sequence.accept_length_cpu[0].item() < 0):
                spec_out_cache_loc_list.append(0)
            else:
                spec_out_cache_loc_list.extend(sequence.out_cache_loc_spec[-total_draft_len:])
            spec_kv_indptr_list.append(spec_kv_indptr_list[-1] + len(sequence.out_cache_loc_spec))
            spec_kv_indices_list.extend(sequence.out_cache_loc_spec)

        padded_token = self.forward_batch.padded_token
        if padded_token > 0:
            spec_out_cache_loc_list.extend([0] * padded_token)
            spec_kv_indices_list.extend([0] * padded_token)
        out_cache_loc_spec = torch.tensor(spec_out_cache_loc_list, dtype=torch.int64, pin_memory=True).to(device, non_blocking=True)
        kv_indices_spec = torch.tensor(spec_kv_indices_list, dtype=torch.int64, pin_memory=True).to(device, non_blocking=True)

        #kv_indptr would be used in two place, extend forward for the latest accepted token,, the other is multi-step draft decode path
        self.forward_batch_spec.kv_indptr = torch.tensor(
            spec_kv_indptr_list, dtype=torch.int32, pin_memory=True,
        ).to(device, non_blocking=True)
        self.forward_batch_spec.kv_indices = kv_indices_spec
        self.forward_batch_spec.kv_indices_mtd = kv_indices_spec[:spec_kv_indptr_list[-1]]
        self.forward_batch_spec.padded_token = padded_token
        self.forward_batch_spec.out_cache_loc = out_cache_loc_spec
        extend_lens = [len(seq.new_tokens) for seq in self.sequences]
        self.forward_batch_spec.max_extend_len = max(extend_lens)
        # Overlap resolves these reserved rows to accepted rows before forward.
        self.forward_batch_spec.qo_indptr = torch.tensor(
            [0, *itertools.accumulate(extend_lens)], dtype=torch.int32, pin_memory=True,
        ).to(device, non_blocking=True)
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
        if is_overlap:
            hidden_states_buffer,compute_stream = self.overlap_affiliated
            with torch.cuda.stream(compute_stream):
                self.update_spec_info_if_needed(hidden_states_buffer=hidden_states_buffer)
        else:
            self.update_spec_info_if_needed(None)

    def prepare_inputs(self, is_overlap:bool=False):
        cur_seq_lens_list = [0]
        input_ids_list = []
        position_ids_list = []
        out_cache_loc_list = []
        kv_indices_parts = []
        prefix_lens_list = [0]
        device = self.device

        batch_size = len(self.sequences)
        if len(self.sequences[-1].new_tokens) == len(self.sequences[-1].tokens):
            self.forward_batch.forward_mode = ForwardMode.EXTEND
        else:
            self.forward_batch.forward_mode = ForwardMode.DECODE
        padded_token = 0
        if (
            self.forward_batch.is_decode()
            and self.forward_batch_spec is None
            and batch_size <= DEFAULT_CUDA_GRAPH_BATCH_SIZES[-1]
        ):
            graph_batch_size = DEFAULT_CUDA_GRAPH_BATCH_SIZES[
                bisect.bisect_left(DEFAULT_CUDA_GRAPH_BATCH_SIZES, batch_size)
            ]
            if graph_batch_size <= get_global_server_args().cuda_graph_max_bs:
                padded_token = graph_batch_size - batch_size

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
            # Local views are copied into the batch-owned pinned buffer below.
            cache_locs = torch.from_numpy(np.frombuffer(sequence.out_cache_loc, dtype=np.int64))
            if self.forward_batch_spec is not None and self.forward_batch.forward_mode == ForwardMode.DECODE:
                true_lens = len(sequence.out_cache_loc) - get_global_server_args().speculative_num_draft_tokens
                prefix_lens_list[-1] = true_lens
                # target model used for verify, speculative_num_draft_tokens cache loc, different from normal decode
                out_cache_loc_list.extend(sequence.out_cache_loc[true_lens:])
                kv_indices_parts.append(cache_locs[:true_lens])
            else:
                out_cache_loc_list.extend(sequence.out_cache_loc[-len(sequence.new_tokens):])
                kv_indices_parts.append(cache_locs)

        if padded_token > 0:
            input_ids_list.extend([0]*padded_token)
            position_ids_list.extend([0]*padded_token)
            cur_seq_lens_list.extend([1]*padded_token)
            out_cache_loc_list.extend([0]*padded_token)
            prefix_lens_list.extend([0]*padded_token)
    
        input_ids = torch.tensor(input_ids_list, dtype=torch.long, pin_memory=True).to(device, non_blocking=True)
        position_ids = torch.tensor(position_ids_list, dtype=torch.long, pin_memory=True).to(device, non_blocking=True)
        cur_seq_lens = torch.tensor(cur_seq_lens_list, dtype=torch.int32, pin_memory=True).to(device, non_blocking=True)
        out_cache_loc = torch.tensor(out_cache_loc_list, dtype=torch.int64, pin_memory=True).to(device, non_blocking=True)
        num_kv_indices = sum(part.shape[0] for part in kv_indices_parts)
        kv_indices_cpu = torch.empty(
            num_kv_indices + padded_token,
            dtype=torch.int64,
            device="cpu",
            pin_memory=True,
        )
        torch.cat(kv_indices_parts, out=kv_indices_cpu[:num_kv_indices])
        if padded_token:
            kv_indices_cpu[num_kv_indices:].zero_()
        kv_indices = kv_indices_cpu.to(device, non_blocking=True)

        prefix_lens = torch.tensor(prefix_lens_list, dtype=torch.int32, pin_memory=True).to(device, non_blocking=True)

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
                self.prepare_decode_for_draft(position_ids_list, is_overlap=is_overlap)

    def prepare_sample(self):
        is_all_greedy = all(
            seq.sampling_params.is_greedy for seq in self.sequences
        )
        if is_all_greedy:
            self.forward_batch.sampling_batch_info = SamplingBatchInfo(
                temperatures=None,
                top_ps=None,
                top_ks=None,
                min_ps=None,
                is_all_greedy=True,
            )
            return

        temperatures = []
        top_ps = []
        top_ks = []
        device = self.device

        for seq in self.sequences:
            temperatures.append(
                1.0 if seq.sampling_params.is_greedy else seq.sampling_params.temperature
            )
            top_ps.append(seq.sampling_params.top_p)
            top_ks.append(1 if seq.sampling_params.is_greedy else seq.sampling_params.top_k)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).to(device, non_blocking=True)
        top_ks = torch.tensor(top_ks, dtype=torch.int32, pin_memory=True).to(device, non_blocking=True)
        top_ps = torch.tensor(top_ps, dtype=torch.float32, pin_memory=True).to(device, non_blocking=True)
        self.forward_batch.sampling_batch_info = SamplingBatchInfo(
            temperatures=temperatures,
            top_ps=top_ps,
            top_ks=top_ks,
            min_ps=torch.zeros_like(top_ps),
            is_all_greedy=False,
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
