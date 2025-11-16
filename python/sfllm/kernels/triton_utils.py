"""
Fully asynchronous Triton implementation for splitting next_token_ids based on accept_length.

This module provides high-performance GPU kernels that minimize CPU synchronization
for maximum throughput in speculative decoding scenarios. All heavy lifting is done
within Triton kernels on the GPU.
"""

import torch
import triton
import triton.language as tl
from typing import List, Optional

@triton.jit  
def optimized_batch_split_kernel(
    # Input tensors
    next_token_ids_ptr,     # Input tokens [total_tokens]
    accept_length_ptr,      # Accept lengths [batch_size]
    cumsum_offsets_ptr,     # Pre-computed cumulative offsets [batch_size + 1]
    # Output tensor
    output_tensor_ptr,      # Output tensor [batch_size, max_tokens]
    # Dimensions
    batch_size: tl.constexpr,
    max_tokens: tl.constexpr,
    total_tokens: tl.constexpr,
    SEQUENCES_PER_BLOCK: tl.constexpr,  # Process multiple sequences per block
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized kernel that processes multiple sequences per Triton block.
    
    This reduces grid size and SM allocation by having each block handle
    SEQUENCES_PER_BLOCK sequences instead of just one sequence per block.
    Ideal for small sequences (1-6 tokens) with large batch sizes.
    """
    # Block and thread identification
    block_id = tl.program_id(0)
    
    # Each block processes SEQUENCES_PER_BLOCK sequences
    seq_start = block_id * SEQUENCES_PER_BLOCK
    
    # Process all sequences assigned to this block
    for seq_offset in range(SEQUENCES_PER_BLOCK):
        seq_id = seq_start + seq_offset
        
        # Bounds check - use conditional processing instead of break
        if seq_id < batch_size:
            # Load sequence metadata
            accept_len = tl.load(accept_length_ptr + seq_id)
            input_start = tl.load(cumsum_offsets_ptr + seq_id)
            output_row_start = seq_id * max_tokens
            
            # Process tokens using fixed block size
            token_offsets = tl.arange(0, BLOCK_SIZE)
            num_blocks = (max_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
            
            for block_idx in range(num_blocks):
                block_start = block_idx * BLOCK_SIZE
                token_indices = block_start + token_offsets
                
                # Masks for bounds checking
                within_output_mask = token_indices < max_tokens
                within_accept_mask = token_indices < accept_len
                copy_mask = within_output_mask & within_accept_mask
                
                # Initialize with zeros
                output_tokens = tl.zeros([BLOCK_SIZE], dtype=tl.int64)
                
                # Copy valid tokens
                if tl.sum(copy_mask) > 0:
                    input_positions = input_start + token_indices
                    input_bounds_mask = copy_mask & (input_positions < total_tokens)
                    tokens = tl.load(next_token_ids_ptr + input_positions, 
                                   mask=input_bounds_mask, other=0)
                    output_tokens = tl.where(copy_mask, tokens, output_tokens)
                
                # Store to output tensor
                output_positions = output_row_start + token_indices
                tl.store(output_tensor_ptr + output_positions, output_tokens, 
                        mask=within_output_mask)


def split_tokens_async(
    next_token_ids: torch.Tensor,
    accept_length: torch.Tensor,
    output_buffers: Optional[torch.Tensor] = None
) -> List[torch.Tensor]:
    device = next_token_ids.device
    batch_size = accept_length.shape[0]
    total_tokens = next_token_ids.shape[0]
    
    cumsum_offsets = torch.cat([torch.zeros(1, device=device, dtype=accept_length.dtype), 
                               torch.cumsum(accept_length, dim=0)])
    
    max_tokens = total_tokens   # Conservative upper bound
    BLOCK_SIZE = 16
    if output_buffers is not None:
        assert output_buffers.is_contiguous() and output_buffers.shape[0] == batch_size
        output_tensor = output_buffers
        max_tokens = output_buffers.shape[1]
        BLOCK_SIZE = triton.next_power_of_2(max_tokens)
    else:
        output_tensor = torch.zeros(batch_size, max_tokens, dtype=next_token_ids.dtype, device=device)
        
    SEQUENCES_PER_BLOCK = 8  # Each block handles 8 sequences
    grid_size = (batch_size + SEQUENCES_PER_BLOCK - 1) // SEQUENCES_PER_BLOCK
    
    grid = (grid_size,)
    optimized_batch_split_kernel[grid](
        next_token_ids,
        accept_length,
        cumsum_offsets,
        output_tensor,
        batch_size,
        max_tokens,
        total_tokens,
        SEQUENCES_PER_BLOCK,
        BLOCK_SIZE  # BLOCK_SIZE
    )
    if output_buffers is not None:
        return output_buffers

    segments = []
    for i in range(batch_size):
        seq_length = accept_length[i]
        segment = output_tensor[i, :seq_length]
        segments.append(segment)
    
    return segments

def split_tokens_pytorch_reference(
    next_token_ids: torch.Tensor,
    accept_length: torch.Tensor  
) -> List[torch.Tensor]:
    """
    Reference PyTorch implementation for comparison and fallback.
    
    Uses standard PyTorch operations but with minimal CPU synchronization.
    """
    # Use PyTorch's cumsum for GPU computation
    cumsum_lengths = torch.cumsum(
        torch.cat([torch.zeros(1, device=accept_length.device, 
                               dtype=accept_length.dtype), accept_length]), 
        dim=0
    )
    segments = []
    for i in range(accept_length.shape[0]):
        start = cumsum_lengths[i]
        end = cumsum_lengths[i + 1]
        segment = next_token_ids[start:end]
        segments.append(segment)

    return segments


@triton.jit
def compact_non_negative_kernel(
    # Input tensor
    input_ptr,              # Input tensor [batch_size, seq_len]
    # Output tensors
    output_ptr,             # Output tensor [batch_size, seq_len] 
    # valid_count_ptr,        # Valid count per sequence [batch_size]
    # Dimensions
    batch_size: tl.constexpr,
    seq_len: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get sequence index
    seq_id = tl.program_id(0)
    if seq_id >= batch_size:
        return
    input_row_start = seq_id * seq_len
    output_row_start = seq_id * seq_len
    
    # Step 1: Vectorized read entire sequence into internal array
    indices = tl.arange(0, BLOCK_SIZE)
    load_mask = indices < seq_len
    
    input_positions = input_row_start + indices
    elements = tl.load(input_ptr + input_positions, mask=load_mask, other=-1)
    
    # Step 2: Vectorized compute validity mask and count
    valid_mask = (elements != -1) & load_mask
    valid_count = tl.sum(valid_mask.to(tl.int32))
    
    # Step 3: Direct parallel compaction using your insight!
    compacted = tl.full([BLOCK_SIZE], -1, dtype=elements.dtype)
    cumulative_count = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    temp_valid = valid_mask.to(tl.int32)
    cumulative_count = tl.cumsum(temp_valid, axis=0) - temp_valid  # 0-based positions
    for out_pos in tl.static_range(BLOCK_SIZE):
        if out_pos < valid_count:
            # Find which input element should go to position out_pos
            # It's the element whose cumulative_count equals out_pos
            source_mask = (cumulative_count == out_pos) & valid_mask & load_mask
            
            # Extract the element (should be exactly one)
            source_element = tl.sum(tl.where(source_mask, elements, 0))
            
            # Place it in the output
            compacted = tl.where(
                indices == out_pos,
                source_element,
                compacted
            )
    output_positions = output_row_start + indices
    tl.store(output_ptr + output_positions, compacted, mask=load_mask)


def compact_non_negative_one(
    input_tensor: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, seq_len = 1, input_tensor.shape[0]
    output_tensor = torch.zeros_like(input_tensor)
    BLOCK_SIZE = triton.next_power_of_2(seq_len)
    grid = (batch_size,)
    compact_non_negative_kernel[grid](
        input_tensor,
        output_tensor,
        batch_size,
        seq_len,
        BLOCK_SIZE
    )
    
    return output_tensor


def compact_non_negative_pytorch_reference(
    input_tensor: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reference PyTorch implementation for comparison.
    """
    device = input_tensor.device
    batch_size, seq_len = input_tensor.shape
    
    output_tensor = torch.zeros_like(input_tensor)
    valid_counts = torch.zeros(batch_size, dtype=torch.int64, device=device)
    
    for i in range(batch_size):
        a = input_tensor[i]
        # Simple and elegant: concatenate non-(-1) elements first, then -1 elements
        compacted = torch.cat([a[a != -1], a[a == -1]])
        output_tensor[i] = compacted
        valid_counts[i] = (a != -1).sum()
    
    return output_tensor, valid_counts


if __name__ == "__main__":
    # Test the compact non-negative implementation
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Test case: Compact non-negative numbers
    print("\nTest case: Compact non-negative numbers")
    input_data = torch.tensor([
        [1, -1, 3, -1, 5, -1],
        [-1, 2, -1, 4, -1, 6],
        [7, 8, 9, -1, -1, -1],
        [-1, -1, -1, 10, 11, 12]
    ], device=device)
    
    print(f"Input: {input_data.tolist()}")
    
    # PyTorch reference
    pytorch_result, pytorch_counts = compact_non_negative_pytorch_reference(input_data)
    print(f"PyTorch result: {pytorch_result.tolist()}")
    print(f"PyTorch counts: {pytorch_counts.tolist()}")
    
    # Triton implementation
    if device == "cuda":
        triton_result, triton_counts = compact_non_negative_one(input_data)
        print(f"Triton result: {triton_result.tolist()}")
        print(f"Triton counts: {triton_counts.tolist()}")
        
        # Verify results match
        result_match = torch.equal(pytorch_result, triton_result)
        count_match = torch.equal(pytorch_counts, triton_counts)
        print(f"Results match: {result_match}")
        print(f"Counts match: {count_match}")
    else:
        print("CUDA not available, skipping Triton test")
