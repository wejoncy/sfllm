
from collections import deque
import logging
import itertools
import torch
from typing import List
from sfllm.server_args import ServerArgs

logger = logging.getLogger(__name__)

class BlockMemory:
    """A simple block memory implementation."""
    def __init__(self, block_id: int):
        self.ref_count = 0
        self.block_id = block_id
        self.token_id = -1
        self.hashv = -1


    def alloc(self, token_id: int, hashv: int):
        """Allocate a block of memory."""
        self.ref_count += 1
        self.token_id = token_id
        self.hashv = hashv
    
    def free(self):
        """Free the block of memory."""
        self.ref_count -= 1
        assert self.ref_count >= 0, "BlockMemory ref_count < 0"
        self.token_id = -1
        self.hashv = -1


class BlockMemoryManager:
    """A simple block memory manager."""
    def __init__(self, server_args: ServerArgs, model_config, num_blocks: int = None):
        dtype = server_args.model_config.dtype
        self.dtype = (
            dtype if server_args.dtype == "auto" else getattr(torch, server_args.dtype)
        )
        self.config = model_config

        self.num_blocks = num_blocks
        self.num_blocks, self.block_shape = self.get_num_blocks(server_args)
        self.blocks = [BlockMemory(i) for i in range(self.num_blocks)]
        self.free_block_ids = list(range(1, self.num_blocks))
        self.release_block_ids = []
        self.used_block_ids = set([])
        self.kv_buffers = []
        self.create_physical_memory_pool(server_args)

    def get_num_blocks(self, server_args) -> int:
        config = self.config
        dim = (
            getattr(config, "head_dim", None)
            or config.hidden_size // config.num_attention_heads
        )
        n_heads = config.num_key_value_heads
        free, total = torch.cuda.mem_get_info("cuda:0")
        one_token_size = n_heads * dim * self.dtype.itemsize * 2  # key + value
        max_length = (
            int(free * server_args.mem_fraction)
            // one_token_size
            // config.num_hidden_layers
        )
        if self.num_blocks is not None:
            max_length = min(max_length, self.num_blocks)
        logger.info(
            f"GPU memory free: {free / (1024**3):.2f} GB, total: {total / (1024**3):.2f} GB"
            f", max tokens per layer: {max_length}"
        )
        return max_length, (n_heads, dim)

    def create_physical_memory_pool(self, server_args):
        config = self.config
        kv_buffers = (
                torch.zeros((config.num_hidden_layers,
                    self.num_blocks, *self.block_shape), dtype=self.dtype, device="cuda"
                ),
                torch.zeros((config.num_hidden_layers,
                    self.num_blocks, *self.block_shape), dtype=self.dtype, device="cuda"),
        )
        for _ in range(config.num_hidden_layers):
            self.kv_buffers.append(
                (
                    kv_buffers[0][_],
                    kv_buffers[1][_],
                )
            )

    def _alloc_block_by_id(self, block_id: int, token_id: int, hashv: int) -> BlockMemory:
        """Allocate a block of memory by block ID."""
        block = self.blocks[block_id]
        block.alloc(token_id, hashv)
        self.used_block_ids.add(block_id)
        return block
    
    def _free_block_by_id(self, block_id: int):
        """Free a block of memory by block ID."""
        # block = self.blocks[block_id]
        # block.free()
        self.used_block_ids.remove(block_id)
        self.release_block_ids.append(block_id)

    def sort_free_blocks(self):
        """Sort the free blocks."""
        self.release_block_ids.extend(self.free_block_ids)
        self.free_block_ids.clear()
        self.release_block_ids.sort()
        self.free_block_ids.extend(self.release_block_ids)
        self.release_block_ids.clear()

    def can_alloc(self, token_len: int) -> bool:
        """Check if a block of memory can be allocated."""
        if len(self.free_block_ids) < token_len:
            self.sort_free_blocks()
        return len(self.free_block_ids) >= token_len

    def persist_alloc_block_from_rear(self, num_tokens: int) -> List[int]:
        popped = self.free_block_ids[-num_tokens:]
        self.free_block_ids = self.free_block_ids[:-num_tokens]
        return popped

    def alloc_block(self, token_nums: int) -> List[int]:
        """Allocate a block of memory."""
        block_ids = self.free_block_ids[:token_nums]
        self.free_block_ids = self.free_block_ids[token_nums:]
        self.used_block_ids.update(block_ids)
        return block_ids

    def free_block(self, block_ids: List[int], force_sort: bool = False):
        """Free a block of memory."""
        for block_id in block_ids:
            self._free_block_by_id(block_id)
        if force_sort:
            self.sort_free_blocks()
    
    def num_available_blocks(self) -> int:
        """Get the number of available blocks."""
        return len(self.free_block_ids)+len(self.release_block_ids)

    def get_usage(self) -> float:
        """Get the memory usage percentage."""
        used_blocks = len(self.used_block_ids)
        return (used_blocks / self.num_blocks) * 100.0
