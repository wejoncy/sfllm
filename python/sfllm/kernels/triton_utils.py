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
def update_kv_indices_kernel(
    future_kvindice_ptr,
    kv_indices_ptr,
    kv_indptr_ptr,
    accept_length_ptr,
    cum_accept_length_ptr,
    batch_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    if pid >= batch_size:
        return

    # Load accept length
    accept_len = tl.load(accept_length_ptr + pid)
    L = accept_len - 1
    
    # If L <= 0, nothing to copy
    if L <= 0:
        return

    # Destination range
    # kv_indptr has size batch_size + 1. We want kv_indptr[pid+1] as end.
    dst_end = tl.load(kv_indptr_ptr + pid + 1)
    dst_start = dst_end - L
    
    # Source range
    # cum_accept_length has size batch_size + 1.
    src_base = tl.load(cum_accept_length_ptr + pid)
    src_start = src_base + 1
    
    # Copy loop
    for off in range(0, L, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < L
        
        # Read from future_kvindice
        val = tl.load(future_kvindice_ptr + src_start + cols, mask=mask)
        
        # Write to kv_indices
        tl.store(kv_indices_ptr + dst_start + cols, val, mask=mask)


def prune_kv_indices(
    future_kvindice: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    accept_length: torch.Tensor,
) -> torch.Tensor:
    """
    Update kv_indices with accepted tokens from future_kvindice and compact.
    
    Args:
        future_kvindice: Tensor containing accepted tokens [total_accepted]
        kv_indices: Destination indices tensor to update [total_size]
        kv_indptr: Pointers to segments in kv_indices [batch_size + 1]
        accept_length: Number of tokens to accept for each batch [batch_size]
    """
    batch_size = len(accept_length)
    device = future_kvindice.device
    
    cum_accept_length = torch.cat([torch.zeros(1, device=device, dtype=torch.int64), accept_length.cumsum(dim=0)])
    
    grid = (batch_size,)
    BLOCK_SIZE = 128 
    
    update_kv_indices_kernel[grid](
        future_kvindice,
        kv_indices,
        kv_indptr,
        accept_length,
        cum_accept_length,
        batch_size,
        BLOCK_SIZE
    )

    kv_indices[:] = move_neg1_to_tail(kv_indices)
    
    return kv_indices


def prune_kv_indices_pytorch_reference(
    future_kvindice: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    accept_length: torch.Tensor,
) -> torch.Tensor:
    """
    Reference PyTorch implementation for prune_kv_indices.
    """
    kv_indptr = kv_indptr[1:]
    batch_size = len(kv_indptr)
    device_id = future_kvindice.device
    cum_accept_length = torch.cat([torch.tensor([0], device=device_id), accept_length.cumsum(dim=0)], dim=0)
    for i in range(batch_size):
        L = accept_length[i] - 1
        dst_end = kv_indptr[i]
        dst_start = dst_end - L
        kv_indices[dst_start:dst_end] = future_kvindice[cum_accept_length[i]+1:cum_accept_length[i+1]]
    kv_indices[:] = move_neg1_to_tail(kv_indices)
    return kv_indices

if __name__ == "__main__":
    import time
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def run_test(name, ref_fn, tri_fn, args, inplace_idx=None, check_ref=None):
        print(f"\n{'='*40}\n{name}\n{'='*40}")
        if device == "cpu": return print("Skipping (No CUDA)")
        
        # Clone for correctness check
        args_r = [x.clone() if isinstance(x, torch.Tensor) else x for x in args]
        args_t = [x.clone() if isinstance(x, torch.Tensor) else x for x in args]
        
        out_r = ref_fn(*args_r)
        out_t = tri_fn(*args_t)
        
        if inplace_idx is not None: out_t = args_t[inplace_idx]
        if check_ref: out_r = check_ref(out_r)
        if isinstance(out_r, tuple): out_r = out_r[0]
            
        match = torch.allclose(out_r, out_t) if out_r.is_floating_point() else torch.equal(out_r, out_t)
        print(f"Match: {match}")
        if not match: return

        # Benchmark
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(1000): ref_fn(*args)
        torch.cuda.synchronize()
        t_ref = time.time() - t0
        
        t0 = time.time()
        for _ in range(1000): tri_fn(*args)
        torch.cuda.synchronize()
        t_tri = time.time() - t0
        print(f"Ref: {t_ref*1000:.2f}ms | Tri: {t_tri*1000:.2f}ms | Speedup: {t_ref/t_tri:.2f}x")

    # 1. Batch Split
    B, H = 10, 4096
    inp = torch.randn(B, H, device=device, dtype=torch.float16)
    lens = torch.tensor([3, 4, 3], device=device)
    run_test("Batch Split", split_firstdim_pytorch_reference, split_firstdim_async, [inp, lens])

    # 2. Move Neg1
    inp = torch.tensor([[1, -1, 3, -1], [-1, 2, -1, 4]], device=device)
    run_test("Move Neg1", move_neg1_to_tail_pytorch_reference, move_neg1_to_tail, [inp])

    # 3. Compact Tokens
    B = 80
    seq_lens = torch.randint(10, 30, (B,), device=device)
    kv_indptr = torch.cat([torch.tensor([0], device=device), torch.cumsum(seq_lens, 0)])
    x = torch.arange(kv_indptr[-1], device=device)
    acc_len = torch.randint(0, 7, (B,), device=device)        
    run_test("Compact Tokens", compact_accepted_tokens_pytorch_reference, compact_accepted_tokens, 
             [x, kv_indptr, acc_len], inplace_idx=0, check_ref=None)

    # 4. Prune KV
    draft_steps = 7
    seq_lens = torch.randint(draft_steps + 5, draft_steps + 20, (B,), device=device)
    kv_indptr = torch.cat([torch.tensor([0], device=device), torch.cumsum(seq_lens, 0)])
    kv_indices = torch.arange(kv_indptr[-1], device=device)
    acc_len = torch.randint(1, draft_steps, (B,), device=device)
    future = torch.randint(10000, 20000, (acc_len.sum() + 100,), device=device)
    
    run_test("Prune KV", prune_kv_indices_pytorch_reference, prune_kv_indices, 
             [future, kv_indices, kv_indptr, acc_len], inplace_idx=1)
