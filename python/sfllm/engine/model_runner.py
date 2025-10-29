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
from sfllm.utils.nutils import DEFAULT_CUDA_GRAPH_BATCH_SIZES, MAX_PROCESSED_TOKENS

logger = logging.getLogger(__name__)

class ModelRunner:
    def __init__(self, server_args: ServerArgs, device_id: int = 0):
        self.model = ForwardModel(server_args.model_path, server_args.dtype)
        server_args.model_config = self.model.config
        self.device_id = device_id
        self.compute_stream = torch.cuda.Stream(device=device_id)
        self.copy_in_stream = torch.cuda.Stream(device=device_id)
        self.cuda_graph_max_bs = server_args.cuda_graph_max_bs
        ind = bisect.bisect_right(DEFAULT_CUDA_GRAPH_BATCH_SIZES, self.cuda_graph_max_bs)
        self.capture_batch_size = DEFAULT_CUDA_GRAPH_BATCH_SIZES[:ind]
        self.capture_batch_size = self.capture_batch_size[:ind]
        self.sampler = Sampler(self.model.config)
        self.rank = 0
        self.server_args = server_args

        # cuda graph
        max_batch_size = max(self.capture_batch_size)
        self.input_ids = torch.empty((max_batch_size,), dtype=torch.long, device=self.device_id)
        self.position_ids = torch.empty((max_batch_size,), dtype=torch.long, device=self.device_id)
        self.out_cache_loc = torch.zeros((max_batch_size,), dtype=torch.int64, device=self.device_id)

        self.num_kv_splits_buffer = torch.zeros((MAX_PROCESSED_TOKENS,), dtype=torch.int32, device="cuda")+2
        self.kv_indptr_buffer = torch.zeros((MAX_PROCESSED_TOKENS,), dtype=torch.int32, device="cuda")
        self.kv_indices_buffer = torch.zeros((MAX_PROCESSED_TOKENS,), dtype=torch.int64, device="cuda")
        self.qo_indptr_buffer = torch.zeros((MAX_PROCESSED_TOKENS,), dtype=torch.int32, device="cuda")
        config = self.model.config
        max_kv_splits = 16
        self.attn_logits = torch.empty(
            (
                max_batch_size*2,
                config.num_attention_heads,
                max_kv_splits,
                config.head_dim,
            ),
            dtype=torch.float32,
            device="cuda",
        )
        self.attn_lse = torch.empty(
            (max_batch_size*2, config.num_attention_heads, max_kv_splits),
            dtype=torch.float32,
            device="cuda",
        )

        self.output_logits = {}
        self.cuda_graphs = {}
        self.graph_pool = torch.cuda.graph_pool_handle()

    def init_capture_graph(self, memory_pool):
        if self.server_args.disable_cuda_graph is False:
            self.capture_graph(memory_pool)

    def get_max_context_length(self):
        return self.model.config.max_position_embeddings
    
    def prepare_inputs(self, scheduled_batch: ScheduleBatch) -> Dict[str, Any]:
        batch_size = len(scheduled_batch)
        forward_metadata = scheduled_batch.forward_metadata
        padded_batch_size = batch_size + forward_metadata.padded_token
        total_seq_len = forward_metadata.kv_indices.shape[0]

        forward_metadata.attn_logits = self.attn_logits
        forward_metadata.attn_lse = self.attn_lse
        if forward_metadata.forward_mode == ForwardMode.EXTEND:
            ori_data = forward_metadata.kv_indptr
            forward_metadata.kv_indptr = self.kv_indptr_buffer[: batch_size + 1]
            forward_metadata.kv_indptr[1:].copy_(ori_data, non_blocking=True)
            ori_data = forward_metadata.kv_indices
            forward_metadata.kv_indices = self.kv_indices_buffer[
                :total_seq_len
            ]
            forward_metadata.kv_indices.copy_(ori_data, non_blocking=True)
            ori_data = forward_metadata.qo_indptr
            forward_metadata.qo_indptr = self.qo_indptr_buffer[
                : batch_size + 1
            ]
            forward_metadata.qo_indptr[1:].copy_(ori_data, non_blocking=True)
        else:
            ori_data = forward_metadata.kv_indptr
            forward_metadata.kv_indptr = self.kv_indptr_buffer[
                : padded_batch_size + 1
            ]
            forward_metadata.kv_indptr[1:].copy_(ori_data, non_blocking=True)
            ori_data = forward_metadata.kv_indices
            forward_metadata.kv_indices = self.kv_indices_buffer[
                :total_seq_len
            ]
            forward_metadata.kv_indices.copy_(ori_data, non_blocking=True)
            forward_metadata.num_kv_splits = (
                self.num_kv_splits_buffer[:padded_batch_size]
            )
        return {
            "input_ids": scheduled_batch.input_ids,
            "position_ids": scheduled_batch.position_ids,
            # "attention_mask": attention_mask,
            "forward_metadata": forward_metadata,
        }
    def prepare_inputs1(self, scheduled_batch: ScheduleBatch) -> Dict[str, Any]:
        cur_seq_lens_list = []
        input_ids_list = []
        position_ids_list = []
        cache_loc_ids_list = []
        kv_indices_list = []
        prefix_lens_list = []
        batch_size = len(scheduled_batch)
        if len(scheduled_batch[-1].new_tokens) == len(scheduled_batch[-1].tokens):
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

        for sequence in scheduled_batch:
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

        input_ids = torch.tensor(input_ids_list, dtype=torch.long, pin_memory=True).to(self.device_id, non_blocking=True)
        position_ids = torch.tensor(
            position_ids_list, dtype=torch.long, pin_memory=True).to(self.device_id, non_blocking=True)
        cur_seq_lens = torch.tensor(cur_seq_lens_list, dtype=torch.int32, pin_memory=True).to(self.device_id, non_blocking=True)
        cache_loc_ids = torch.tensor(cache_loc_ids_list, dtype=torch.int64, pin_memory=True).to(self.device_id, non_blocking=True)
        kv_indices = torch.tensor(kv_indices_list, dtype=torch.int64, pin_memory=True).to(self.device_id, non_blocking=True)
        prefix_lens = torch.tensor(prefix_lens_list, dtype=torch.int32, pin_memory=True).to(self.device_id, non_blocking=True)

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

    def prepare_replay(self, inputs: Dict[str, Any], scheduled_batch: ScheduleBatch):
        bs_size = len(scheduled_batch)
        forward_metadata = inputs["forward_metadata"]
        padded_bs_size = bs_size + forward_metadata.padded_token
        self.input_ids[:padded_bs_size].copy_(inputs["input_ids"], non_blocking=True)
        self.position_ids[:padded_bs_size].copy_(
            inputs["position_ids"], non_blocking=True
        )
        self.out_cache_loc[:padded_bs_size].copy_(
            forward_metadata.out_cache_loc, non_blocking=True
        )
        # self.attention_mask[:bs_size].copy_(inputs["attention_mask"], non_blocking=True)


    def forward(self, scheduled_batch: ScheduleBatch):
        forward_metadata = scheduled_batch.forward_metadata
        inputs = self.prepare_inputs(scheduled_batch)
        # self.prepare_sample(scheduled_batch) if self.rank == 0 else None
        bs_size = len(scheduled_batch)
        pad_bs_size = forward_metadata.padded_token + bs_size
        if (
            forward_metadata.forward_mode == ForwardMode.DECODE
            and self.cuda_graphs.get(pad_bs_size) is not None
        ):
            self.prepare_replay(inputs, scheduled_batch)
            self.cuda_graphs[pad_bs_size].replay()
            logits = self.output_logits[pad_bs_size][:bs_size]
        else:
            logits = self.model(**inputs)[:bs_size]

        token_ids = (
            self.sampler(logits, forward_metadata.sampling_batch_info) if self.rank == 0 else None
        )
        return token_ids

    def profile_run(self):
        logger.info("Profiling run to reserve activation memory...")
        self.input_ids.fill_(0)
        self.position_ids.fill_(0)
        forward_metadata = ForwardBatch(None)
        forward_metadata.qo_indptr = self.qo_indptr_buffer[: 1]
        forward_metadata.qo_indptr[0] = 0
        with torch.no_grad():
            self.model(
                self.input_ids*0,
                position_ids=self.position_ids*0,
                forward_metadata=forward_metadata,
            )


    def capture_graph(self, memory_pool):
        batch_size = 1
        forward_metadata = ForwardBatch(memory_pool)
        forward_metadata.attn_logits = self.attn_logits
        forward_metadata.attn_lse = self.attn_lse
        input_ids = self.input_ids[:batch_size]
        forward_metadata.kv_indptr = self.kv_indptr_buffer[: batch_size + 1]
        forward_metadata.kv_indices = self.kv_indices_buffer[
            : input_ids.shape[0]
        ]
        forward_metadata.num_kv_splits = self.num_kv_splits_buffer[:batch_size]
        forward_metadata.forward_mode = ForwardMode.DECODE
        forward_metadata.out_cache_loc = forward_metadata.kv_indices
        self.compute_stream.synchronize()

        with torch.no_grad():
            self.model(
                self.input_ids[:batch_size],
                position_ids=self.position_ids[:batch_size],
                forward_metadata=forward_metadata,
            )

        for batch_size in tqdm.tqdm(self.capture_batch_size, desc="Capturing CUDA Graphs"):
            forward_metadata.kv_indptr = self.kv_indptr_buffer[: batch_size + 1]
            forward_metadata.out_cache_loc = self.out_cache_loc[:batch_size]
            torch.cuda.synchronize()
            cudagraph = torch.cuda.CUDAGraph()
            # attention_mask = torch.empty((batch_size), dtype=torch.long, device=self.device_id)
            with torch.cuda.graph(cudagraph, stream=self.compute_stream, pool=self.graph_pool):
                output = self.model(
                    self.input_ids[:batch_size],
                    position_ids=self.position_ids[:batch_size],
                    forward_metadata=forward_metadata,
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
