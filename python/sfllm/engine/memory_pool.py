
from collections import deque


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
    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self.blocks = [BlockMemory(i) for i in range(num_blocks)]
        self.free_block_ids = deque(range(num_blocks))
        self.free_block_ids.popleft()  # reserve block 0
        self.used_block_ids = set([0])

    def _alloc_block_by_id(self, block_id: int, token_id: int, hashv: int) -> BlockMemory:
        """Allocate a block of memory by block ID."""
        block = self.blocks[block_id]
        block.alloc(token_id, hashv)
        self.used_block_ids.add(block_id)
        return block
    
    def _free_block_by_id(self, block_id: int):
        """Free a block of memory by block ID."""
        block = self.blocks[block_id]
        block.free()
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_alloc(self, token_len: int) -> bool:
        """Check if a block of memory can be allocated."""
        return len(self.free_block_ids) >= token_len

    def alloc_block(self, token_ids: list[int], hashv: int) -> BlockMemory:
        """Allocate a block of memory."""
        block_ids = []
        for token_id in token_ids:
            block_ids.append(self.free_block_ids.popleft())
            self._alloc_block_by_id(block_ids[-1], token_id, hashv)

        return block_ids
    
    def free_block(self, block_ids: list[int]):
        """Free a block of memory."""
        for block_id in block_ids:
            self._free_block_by_id(block_id)
        
    def get_usage(self) -> float:
        """Get the memory usage percentage."""
        used_blocks = len(self.used_block_ids)
        return (used_blocks / self.num_blocks) * 100.0