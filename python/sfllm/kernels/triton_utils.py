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
def split_lastdim_kernel(
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


def split_lastdim_async(
    next_token_ids: torch.Tensor,
    accept_length: torch.Tensor,
    output_buffers: Optional[torch.Tensor] = None
) -> List[torch.Tensor]:
    device = next_token_ids.device
    batch_size = accept_length.shape[0]
    total_tokens = output_buffers.shape[-1]
    
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
    split_lastdim_kernel[grid](
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


def split_lastdim_pytorch_reference(
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
def move_neg1_to_tail_kernel(
    # Input tensor
    input_ptr,              # Input tensor [batch_size, seq_len]
    # Output tensors
    output_ptr,             # Output tensor [batch_size, seq_len] 
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
    
    # Step 1: Vectorized read entire sequence
    indices = tl.arange(0, BLOCK_SIZE)
    load_mask = indices < seq_len
    
    input_positions = input_row_start + indices
    elements = tl.load(input_ptr + input_positions, mask=load_mask, other=-1)
    
    # Step 2: Identify valid elements
    valid_mask = (elements != -1) & load_mask
    
    # Step 3: Compute destination indices using cumsum
    # cumsum gives 1-based index, subtract 1 to get 0-based index
    dest_indices = tl.cumsum(valid_mask.to(tl.int32), axis=0) - 1
    
    # Step 4: Store valid elements to their compacted positions
    # We write to output_ptr + row_start + dest_indices
    # Only write where valid_mask is True
    tl.store(output_ptr + output_row_start + dest_indices, elements, mask=valid_mask)
    
    # Step 5: Fill the tail with -1
    total_valid = tl.sum(valid_mask.to(tl.int32))
    tail_mask = (indices >= total_valid) & load_mask
    tl.store(output_ptr + output_row_start + indices, -1, mask=tail_mask)


def move_neg1_to_tail(
    input_tensor: torch.Tensor
) -> torch.Tensor:
    orig_shape = input_tensor.shape
    if input_tensor.dim() == 1:
        input_tensor = input_tensor.unsqueeze(0)
    
    batch_size, seq_len = input_tensor.shape
    output_tensor = torch.zeros_like(input_tensor)
    BLOCK_SIZE = triton.next_power_of_2(seq_len)
    grid = (batch_size,)
    move_neg1_to_tail_kernel[grid](
        input_tensor,
        output_tensor,
        batch_size,
        seq_len,
        BLOCK_SIZE
    )
    
    return output_tensor.view(orig_shape)


def move_neg1_to_tail_pytorch_reference(
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


@triton.jit
def split_firstdim_kernel(
    # Input tensor
    input_ptr,              # Input tensor [total_batch, hidden_size]
    accept_length_ptr,      # Accept lengths for each output batch [num_outputs]
    cumsum_offsets_ptr,     # Pre-computed cumulative offsets [num_outputs + 1]
    # Output tensor
    output_ptr,             # Output tensor [num_outputs, max_batch_size, hidden_size]
    # Dimensions
    num_outputs: tl.constexpr,
    max_batch_size: tl.constexpr,
    hidden_size: tl.constexpr,
    total_batch: tl.constexpr,
    BLOCK_SIZE_HIDDEN: tl.constexpr,
):
    """
    Kernel for splitting batch dimension dynamically.
    
    Each output group gets a variable number of batch items based on accept_length.
    Processes one (output_id, batch_idx) pair per program, vectorized over hidden_size.
    
    Args:
        input_ptr: Input tensor [total_batch, hidden_size]
        accept_length_ptr: Number of batch items for each output [num_outputs]
        cumsum_offsets_ptr: Cumulative sum of accept_length [num_outputs + 1]
        output_ptr: Output tensor [num_outputs, max_batch_size, hidden_size]
    """
    # Get 2D program ID
    output_id = tl.program_id(0)
    batch_idx = tl.program_id(1)
    
    if output_id >= num_outputs or batch_idx >= max_batch_size:
        return
    
    # Load metadata for this output group
    accept_len = tl.load(accept_length_ptr + output_id)
    input_batch_start = tl.load(cumsum_offsets_ptr + output_id)
    
    # Check if this batch index is valid for current output
    if batch_idx >= accept_len:
        # Fill with zeros for out-of-range batch indices
        hidden_offsets = tl.arange(0, BLOCK_SIZE_HIDDEN)
        num_blocks = (hidden_size + BLOCK_SIZE_HIDDEN - 1) // BLOCK_SIZE_HIDDEN
        
        for block_idx in range(num_blocks):
            block_start = block_idx * BLOCK_SIZE_HIDDEN
            indices = block_start + hidden_offsets
            mask = indices < hidden_size
            
            output_offset = (output_id * max_batch_size + batch_idx) * hidden_size + indices
            zeros = tl.zeros([BLOCK_SIZE_HIDDEN], dtype=tl.float16)
            tl.store(output_ptr + output_offset, zeros, mask=mask)
    else:
        # Copy from input to output
        input_batch_idx = input_batch_start + batch_idx
        
        hidden_offsets = tl.arange(0, BLOCK_SIZE_HIDDEN)
        num_blocks = (hidden_size + BLOCK_SIZE_HIDDEN - 1) // BLOCK_SIZE_HIDDEN
        
        for block_idx in range(num_blocks):
            block_start = block_idx * BLOCK_SIZE_HIDDEN
            indices = block_start + hidden_offsets
            mask = indices < hidden_size
            
            # Load from input
            input_offset = input_batch_idx * hidden_size + indices
            input_mask = mask & (input_batch_idx < total_batch)
            data = tl.load(input_ptr + input_offset, mask=input_mask, other=0.0)
            
            # Store to output
            output_offset = (output_id * max_batch_size + batch_idx) * hidden_size + indices
            tl.store(output_ptr + output_offset, data, mask=mask)


def split_firstdim_async(
    input_tensor: torch.Tensor,
    accept_length: torch.Tensor,
    output_buffers: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Split input tensor along batch dimension according to accept_length.
    
    Args:
        input_tensor: Input tensor [total_batch, hidden_size]
        accept_length: Number of batch items for each output [num_outputs]
        output_buffers: Optional pre-allocated output [num_outputs, max_batch_size, hidden_size]
    
    Returns:
        Output tensor [num_outputs, max_batch_size, hidden_size]
    """
    device = input_tensor.device
    total_batch, hidden_size = input_tensor.shape
    num_outputs = accept_length.shape[0]
    
    # Compute cumulative offsets
    cumsum_offsets = torch.cat([
        torch.zeros(1, device=device, dtype=accept_length.dtype),
        torch.cumsum(accept_length, dim=0)
    ])
    
    # Determine max batch size
    
    # Allocate or validate output buffer
    if output_buffers is not None:
        max_batch_size = output_buffers.shape[0]
        assert output_buffers.is_contiguous()
        output_tensor = output_buffers
    else:
        max_batch_size = accept_length.max().item()
        output_tensor = torch.zeros(
            num_outputs, max_batch_size, hidden_size,
            dtype=input_tensor.dtype,
            device=device
        )
    
    # Kernel configuration
    BLOCK_SIZE_HIDDEN = 128  # Vectorize over hidden dimension
    
    # Launch kernel with 2D grid
    grid = (num_outputs, max_batch_size)
    
    split_firstdim_kernel[grid](
        input_tensor,
        accept_length,
        cumsum_offsets,
        output_tensor,
        num_outputs,
        max_batch_size,
        hidden_size,
        total_batch,
        BLOCK_SIZE_HIDDEN
    )
    
    return output_tensor


def split_firstdim_pytorch_reference(
    input_tensor: torch.Tensor,
    accept_length: torch.Tensor
) -> torch.Tensor:
    device = input_tensor.device
    num_outputs = accept_length.shape[0]
    max_batch_size = accept_length.max().item()
    hidden_size = input_tensor.shape[1]
    
    output_tensor = torch.zeros(
        num_outputs, max_batch_size, hidden_size,
        dtype=input_tensor.dtype,
        device=device
    )
    
    cumsum_lengths = torch.cumsum(
        torch.cat([torch.zeros(1, device=device, dtype=accept_length.dtype), accept_length]),
        dim=0
    )
    
    for i in range(num_outputs):
        start = cumsum_lengths[i]
        end = cumsum_lengths[i + 1]
        batch_size = accept_length[i]
        output_tensor[i, :batch_size] = input_tensor[start:end]
    
    return output_tensor


@triton.jit
def compact_accepted_tokens_kernel(
    x_ptr,
    kv_indptr_ptr,
    out_loc_ptr,
    accept_len_ptr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    
    # Current batch index
    batch_idx = pid
    
    # Read source start and length
    src_start = tl.load(kv_indptr_ptr + batch_idx)
    length = tl.load(accept_len_ptr + batch_idx)
    
    # Read destination start
    dst_start = tl.load(out_loc_ptr + batch_idx)
    
    # Loop to copy (handle length > BLOCK_SIZE)
    for off in range(0, length, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < length
        
        # Read
        val = tl.load(x_ptr + src_start + cols, mask=mask)
        
        # Write
        tl.store(x_ptr + dst_start + cols, val, mask=mask)


@triton.jit
def fill_tail_kernel(
    x_ptr,
    start_idx,
    total_size,
    fill_value,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE + start_idx
    
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_size
    
    tl.store(x_ptr + offsets, fill_value, mask=mask)


def compact_accepted_tokens(
    x: torch.Tensor,
    kv_indptr: torch.Tensor,
    accept_length: torch.Tensor,
    fill_value: int = -1
) -> None:
    batch_size = kv_indptr.shape[0] - 1
    
    # 1. Calculate destination offsets
    out_loc = torch.zeros_like(kv_indptr)
    out_loc[1:] = torch.cumsum(accept_length, dim=0)
    
    # 2. Launch parallel copy kernel
    # One block per batch is usually enough for draft tokens
    grid = (batch_size,)
    # Use a reasonable block size, e.g., 256, enough to cover most draft lengths
    BLOCK_SIZE = 256 
    
    compact_accepted_tokens_kernel[grid](
        x,
        kv_indptr,
        out_loc,
        accept_length,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # 3. Fill tail with -1
    total_valid = out_loc[-1].item()
    total_size = x.numel()
    if total_valid < total_size:
        fill_len = total_size - total_valid
        BLOCK_SIZE_FILL = 1024
        grid_fill = (triton.cdiv(fill_len, BLOCK_SIZE_FILL),)
        fill_tail_kernel[grid_fill](
            x,
            total_valid,
            total_size,
            fill_value,
            BLOCK_SIZE=BLOCK_SIZE_FILL
        )


def compact_accepted_tokens_pytorch_reference(x, kv_indptr, accept_length):
    device = x.device
    batch_size = len(kv_indptr) - 1
    x_ref = x.clone()
    ind = 0
    for bs in range(batch_size):
        start = kv_indptr[bs]
        end = start+accept_length[bs]
        x_ref[ind:ind+accept_length[bs]] = x[start:end]
        ind += accept_length[bs]
    x_ref[ind:] = -1  # Fill the tail with -1
    assert (x_ref!=-1).sum() == accept_length.sum()
    return x_ref


@triton.jit
def prune_kv_indices_kernel(
    x_ptr,
    output_ptr,
    kv_indptr_ptr,
    dest_offsets_ptr,
    accept_len_ptr,
    draft_token_steps,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    
    # Get segment info
    src_start = tl.load(kv_indptr_ptr + pid)
    src_end = tl.load(kv_indptr_ptr + pid + 1)
    accept_len = tl.load(accept_len_ptr + pid)
    
    # Calculate keep length
    # kept region is [src_start, src_end - (draft_token_steps - accept_len))
    seg_len = src_end - src_start
    remove_len = draft_token_steps - accept_len
    keep_len = seg_len - remove_len
    
    dst_start = tl.load(dest_offsets_ptr + pid)
    
    # Copy loop
    for off in range(0, keep_len, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < keep_len
        
        # Read from src
        val = tl.load(x_ptr + src_start + cols, mask=mask, other=-1)
        
        # Write to dst
        tl.store(output_ptr + dst_start + cols, val, mask=mask)


def prune_kv_indices(
    x: torch.Tensor,
    kv_indptr: torch.Tensor,
    accept_length: torch.Tensor,
    draft_token_steps: int
) -> torch.Tensor:
    """
    Prunes the last (draft_token_steps - accept_length) tokens from each segment in x,
    and compacts the result.
    
    Args:
        x: Input tensor of indices [total_tokens]
        kv_indptr: Segment pointers [batch_size + 1]
        accept_length: Accepted length for each segment [batch_size]
        draft_token_steps: Number of draft tokens
        
    Returns:
        Compacted tensor with -1 padding at the end.
    """
    batch_size = len(accept_length)
    
    # Calculate keep lengths for each segment
    seg_lens = kv_indptr[1:] - kv_indptr[:-1]
    keep_lens = seg_lens - (draft_token_steps - accept_length)
    
    # Calculate destination offsets
    dest_offsets = torch.zeros_like(kv_indptr)
    dest_offsets[1:] = torch.cumsum(keep_lens, dim=0)
    
    output = torch.full_like(x, -1)
    
    grid = (batch_size,)
    BLOCK_SIZE = 1024
    
    prune_kv_indices_kernel[grid](
        x,
        output,
        kv_indptr,
        dest_offsets,
        accept_length,
        draft_token_steps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output


def prune_kv_indices_pytorch_reference(
    x: torch.Tensor,
    kv_indptr: torch.Tensor,
    accept_length: torch.Tensor,
    draft_token_steps: int
) -> torch.Tensor:
    """
    Reference PyTorch implementation for prune_kv_indices.
    """
    batch_size = len(accept_length)
    output = torch.full_like(x, -1)
    
    current_dest = 0
    for i in range(batch_size):
        src_start = kv_indptr[i]
        src_end = kv_indptr[i+1]
        
        # Calculate keep length
        seg_len = src_end - src_start
        remove_len = draft_token_steps - accept_length[i]
        keep_len = seg_len - remove_len
        
        if keep_len > 0:
            output[current_dest:current_dest+keep_len] = x[src_start:src_start+keep_len]
            current_dest += keep_len
            
    return output


if __name__ == "__main__":
    # Test the compact non-negative implementation
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Test case 1: Batch dimension split
    print("=" * 60)
    print("Test case 1: Batch dimension split")
    print("=" * 60)
    
    # Create test data: [total_batch=10, hidden_size=4096]
    total_batch = 10
    hidden_size = 4096
    input_tensor = torch.randn(total_batch, hidden_size, device=device, dtype=torch.float16)
    
    # Split into 3 groups with sizes [3, 4, 3]
    accept_length = torch.tensor([3, 4, 3], device=device, dtype=torch.int64)
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"Accept lengths: {accept_length.tolist()}")
    print(f"Total batches: {accept_length.sum().item()}")
    
    # PyTorch reference
    pytorch_result = split_firstdim_pytorch_reference(input_tensor, accept_length)
    print(f"PyTorch output shape: {pytorch_result.shape}")
    
    # Triton implementation
    if device == "cuda":
        triton_result = split_firstdim_async(input_tensor, accept_length)
        print(f"Triton output shape: {triton_result.shape}")
        
        # Verify results match
        result_match = torch.allclose(pytorch_result, triton_result, atol=1e-3, rtol=1e-3)
        max_diff = (pytorch_result - triton_result).abs().max().item()
        print(f"Results match: {result_match}")
        print(f"Max difference: {max_diff:.6e}")
        
        # Verify each group
        cumsum = torch.cat([torch.zeros(1, device=device, dtype=torch.int64), 
                           torch.cumsum(accept_length, dim=0)])
        for i in range(accept_length.shape[0]):
            start = cumsum[i]
            end = cumsum[i + 1]
            batch_size = accept_length[i]
            
            # Check if split matches original
            original_group = input_tensor[start:end]
            pytorch_group = pytorch_result[i, :batch_size]
            triton_group = triton_result[i, :batch_size]
            
            orig_vs_pytorch = torch.allclose(original_group, pytorch_group, atol=1e-5)
            orig_vs_triton = torch.allclose(original_group, triton_group, atol=1e-5)
            
            print(f"  Group {i} (size={batch_size}): PyTorch={orig_vs_pytorch}, Triton={orig_vs_triton}")
    else:
        print("CUDA not available, skipping Triton test")
    
    print()
    
    # Test case 2: Compact non-negative numbers
    print("=" * 60)
    print("Test case 2: Compact non-negative numbers")
    print("=" * 60)
    input_data = torch.tensor([
        [1, -1, 3, -1, 5, -1],
        [-1, 2, -1, 4, -1, 6],
        [7, 8, 9, -1, -1, -1],
        [-1, -1, -1, 10, 11, 12]
    ], device=device)
    
    print(f"Input: {input_data.tolist()}")
    
    # PyTorch reference
    pytorch_result, pytorch_counts = move_neg1_to_tail_pytorch_reference(input_data)
    print(f"PyTorch result: {pytorch_result.tolist()}")
    print(f"PyTorch counts: {pytorch_counts.tolist()}")
    
    # Triton implementation
    if device == "cuda":
        triton_result = move_neg1_to_tail(input_data)
        print(f"Triton result: {triton_result.tolist()}")
        # print(f"Triton counts: {triton_counts.tolist()}")
        
        # Verify results match
        result_match = torch.equal(pytorch_result, triton_result)
        # count_match = torch.equal(pytorch_counts, triton_counts)
        print(f"Results match: {result_match}")
        # print(f"Counts match: {count_match}")
    else:
        print("CUDA not available, skipping Triton test")
    
    print()

    # Test case 3: Mask last dynamic tokens
    print("=" * 60)
    print("Test case 3: Mask last dynamic tokens")
    print("=" * 60)
    
    if device == "cuda":
        # Setup test data
        batch_size = 80
        seq_lens = torch.randint(10, 30, (batch_size,), device=device)
        kv_indptr = torch.cat([torch.tensor([0], device=device), torch.cumsum(seq_lens, dim=0)])
        
        # Calculate total size
        total_size = kv_indptr[-1]
        
        x = torch.arange(total_size, device=device)
        
        # Dynamic accept lengths
        draft_token_steps = 7
        accept_length = torch.randint(0, draft_token_steps, (batch_size,), device=device)
        
        # PyTorch reference
        x_ref = compact_accepted_tokens_pytorch_reference(x, kv_indptr, accept_length)
        # Compact reference
        valid_mask = x_ref != -1
        valid_elements = x_ref[valid_mask]
        x_ref_compact = torch.full_like(x_ref, -1)
        x_ref_compact[:len(valid_elements)] = valid_elements
        
        # Triton implementation
        x_triton = x.clone()
        compact_accepted_tokens(x_triton, kv_indptr, accept_length)
        
        # Verify
        match = torch.equal(x_ref_compact, x_triton)
        print(f"Results match: {match}")
        
        if not match:
            print("Mismatch found!")
            print("Reference:", x_ref_compact[:100])
            print("Triton:", x_triton[:100])
            diff = torch.where(x_ref_compact != x_triton)[0]
            print(f"First mismatch at index {diff[0]}: Ref={x_ref_compact[diff[0]]}, Triton={x_triton[diff[0]]}")
            
        # Benchmark
        import time
        
        # Warmup
        for _ in range(10):
            # PyTorch benchmark: mask + compact
            x_temp = compact_accepted_tokens_pytorch_reference(x, kv_indptr, accept_length)
            valid = x_temp[x_temp != -1]
            out = torch.full_like(x_temp, -1)
            out[:len(valid)] = valid
            
            compact_accepted_tokens(x_triton, kv_indptr, accept_length)
            
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(1000):
            x_temp = compact_accepted_tokens_pytorch_reference(x, kv_indptr, accept_length)
            valid = x_temp[x_temp != -1]
            out = torch.full_like(x_temp, -1)
            out[:len(valid)] = valid
        torch.cuda.synchronize()
        torch_time = (time.time() - start) / 1000
        
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(1000):
            compact_accepted_tokens(x_triton, kv_indptr, accept_length)
        torch.cuda.synchronize()
        triton_time = (time.time() - start) / 1000
        
        print(f"PyTorch time: {torch_time*1000:.3f} ms")
        print(f"Triton time:  {triton_time*1000:.3f} ms")
        print(f"Speedup: {torch_time/triton_time:.2f}x")
    else:
        print("CUDA not available, skipping Triton test")
    
    print()
    
    # Test case 4: Prune KV indices
    print("=" * 60)
    print("Test case 4: Prune KV indices")
    print("=" * 60)
    
    if device == "cuda":
        # Setup test data
        batch_size = 80
        draft_token_steps = 7
        
        # Create segments with length > draft_token_steps
        seq_lens = torch.randint(draft_token_steps + 5, draft_token_steps + 20, (batch_size,), device=device)
        kv_indptr = torch.cat([torch.tensor([0], device=device), torch.cumsum(seq_lens, dim=0)])
        
        total_size = kv_indptr[-1]
        x = torch.arange(total_size, device=device)
        
        accept_length = torch.randint(0, draft_token_steps, (batch_size,), device=device)
        
        # PyTorch reference
        x_ref = prune_kv_indices_pytorch_reference(x, kv_indptr, accept_length, draft_token_steps)
        
        # Triton implementation
        x_triton = prune_kv_indices(x, kv_indptr, accept_length, draft_token_steps)
        
        # Verify
        match = torch.equal(x_ref, x_triton)
        print(f"Results match: {match}")
        
        if not match:
            print("Mismatch found!")
            diff = torch.where(x_ref != x_triton)[0]
            if len(diff) > 0:
                idx = diff[0]
                print(f"First mismatch at index {idx}: Ref={x_ref[idx]}, Triton={x_triton[idx]}")
        
        # Benchmark
        import time
        
        # Warmup
        for _ in range(10):
            prune_kv_indices_pytorch_reference(x, kv_indptr, accept_length, draft_token_steps)
            prune_kv_indices(x, kv_indptr, accept_length, draft_token_steps)
            
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(1000):
            prune_kv_indices_pytorch_reference(x, kv_indptr, accept_length, draft_token_steps)
        torch.cuda.synchronize()
        torch_time = (time.time() - start) / 1000
        
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(1000):
            prune_kv_indices(x, kv_indptr, accept_length, draft_token_steps)
        torch.cuda.synchronize()
        triton_time = (time.time() - start) / 1000
        
        print(f"PyTorch time: {torch_time*1000:.3f} ms")
        print(f"Triton time:  {triton_time*1000:.3f} ms")
        print(f"Speedup: {torch_time/triton_time:.2f}x")
    else:
        print("CUDA not available, skipping Triton test")
    
    print()
    
    # Test case 5: Stress test compact_accepted_tokens
    print("=" * 60)
    print("Test case 5: Stress test compact_accepted_tokens")
    print("=" * 60)
    
    if device == "cuda":
        # Try a larger size that might break the single-block assumption
        # 100 batches, 100 tokens each -> 10000 tokens total
        # BLOCK_SIZE will be 16384
        batch_size = 100
        seq_len_per_batch = 100
        total_tokens = batch_size * seq_len_per_batch
        
        print(f"Testing with total_tokens={total_tokens} (BLOCK_SIZE={triton.next_power_of_2(total_tokens)})")
        
        kv_indptr = torch.arange(0, total_tokens + 1, seq_len_per_batch, device=device, dtype=torch.int64)
        x = torch.arange(total_tokens, device=device)
        accept_length = torch.full((batch_size,), seq_len_per_batch // 2, device=device, dtype=torch.int64)
        
        try:
            start = time.time()
            compact_accepted_tokens(x.clone(), kv_indptr, accept_length)
            torch.cuda.synchronize()
            print(f"Execution time: {(time.time() - start)*1000:.3f} ms")
            print("Success!")
        except Exception as e:
            print(f"Failed: {e}")
    else:
        print("CUDA not available")
