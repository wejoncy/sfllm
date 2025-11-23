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
    next_token_ids_ptr,
    accept_length_ptr,
    output_tensor_ptr,
    batch_size,
    max_tokens,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_BATCH: tl.constexpr
):
    pid = tl.program_id(0)
    
    # 1. Compute src_start (Internal Prefix Sum)
    offs_batch = tl.arange(0, BLOCK_BATCH)
    mask_batch = offs_batch < batch_size
    lens = tl.load(accept_length_ptr + offs_batch, mask=mask_batch, other=0)
    offsets = tl.cumsum(lens, 0) - lens
    src_start = tl.sum(tl.where(offs_batch == pid, offsets, 0))

    # 2. Copy and Fill
    cur_len = tl.load(accept_length_ptr + pid)
    output_ptr = output_tensor_ptr + pid * max_tokens
    
    for off in range(0, max_tokens, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask_out = cols < max_tokens
        mask_valid = cols < cur_len
        
        # Default to -1
        vals = tl.full([BLOCK_SIZE], -1, dtype=tl.int64)
        
        # Load valid data
        mask_load = mask_out & mask_valid
        # Optimization: only load if we have valid elements
        if tl.max(mask_load, 0):
            src_vals = tl.load(next_token_ids_ptr + src_start + cols, mask=mask_load, other=0)
            vals = tl.where(mask_valid, src_vals, vals)
            
        tl.store(output_ptr + cols, vals, mask=mask_out)


def split_lastdim_async(
    next_token_ids: torch.Tensor,
    accept_length: torch.Tensor,
    output_buffers: Optional[torch.Tensor] = None
) -> List[torch.Tensor]:
    device = next_token_ids.device
    batch_size = accept_length.shape[0]
    
    if output_buffers is not None:
        output_tensor = output_buffers
        max_tokens = output_buffers.shape[1]
    else:
        # Fallback if not provided (though user said it is given)
        total_tokens = next_token_ids.shape[0]
        max_tokens = total_tokens 
        output_tensor = torch.empty(batch_size, max_tokens, dtype=next_token_ids.dtype, device=device)
        
    BLOCK_SIZE = 64
    if max_tokens < 64:
        BLOCK_SIZE = triton.next_power_of_2(max_tokens)
        
    BLOCK_BATCH = triton.next_power_of_2(batch_size)
    BLOCK_BATCH = max(16, BLOCK_BATCH)

    grid = (batch_size,)
    split_lastdim_kernel[grid](
        next_token_ids,
        accept_length,
        output_tensor,
        batch_size,
        max_tokens,
        BLOCK_SIZE=BLOCK_SIZE,
        BLOCK_BATCH=BLOCK_BATCH
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
    accept_length: torch.Tensor,
    output_buffer: Optional[torch.Tensor] = None
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
    
    if output_buffer is not None:
        output_buffer.fill_(-1)
        for i in range(accept_length.shape[0]):
            start = cumsum_lengths[i]
            end = cumsum_lengths[i + 1]
            length = end - start
            output_buffer[i, :length] = next_token_ids[start:end]
        return output_buffer

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
    batch_size,
    seq_len,
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
    
    # Use a fixed block size to avoid frequent recompilation
    # 32768 is a reasonable upper bound that fits in shared memory on modern GPUs
    BLOCK_SIZE = 4096
    if seq_len > BLOCK_SIZE:
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
    # Output tensor
    output_ptr,             # Output tensor [num_outputs, max_batch_size, hidden_size]
    # Dimensions
    num_outputs,
    max_batch_size,
    hidden_size: tl.constexpr,
    total_batch,
    BLOCK_SIZE_HIDDEN: tl.constexpr,
    BLOCK_OUTPUTS: tl.constexpr
):
    """
    Kernel for splitting batch dimension dynamically.
    
    Each output group gets a variable number of batch items based on accept_length.
    Processes one (output_id, batch_idx) pair per program, vectorized over hidden_size.
    
    Args:
        input_ptr: Input tensor [total_batch, hidden_size]
        accept_length_ptr: Number of batch items for each output [num_outputs]
        output_ptr: Output tensor [num_outputs, max_batch_size, hidden_size]
    """
    # Get 2D program ID
    output_id = tl.program_id(0)
    batch_idx = tl.program_id(1)
    
    if output_id >= num_outputs or batch_idx >= max_batch_size:
        return
    
    # 1. Compute offsets internally
    offs = tl.arange(0, BLOCK_OUTPUTS)
    mask_outputs = offs < num_outputs
    lens = tl.load(accept_length_ptr + offs, mask=mask_outputs, other=0)
    offsets = tl.cumsum(lens, 0) - lens

    # Load metadata for this output group
    accept_len = tl.load(accept_length_ptr + output_id)
    # input_batch_start = tl.load(cumsum_offsets_ptr + output_id)
    input_batch_start = tl.sum(tl.where(offs == output_id, offsets, 0))
    
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
    # cumsum_offsets = torch.cat([
    #     torch.zeros(1, device=device, dtype=accept_length.dtype),
    #     torch.cumsum(accept_length, dim=0)
    # ])
    
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
    BLOCK_OUTPUTS = 128
    if num_outputs > BLOCK_OUTPUTS:
        BLOCK_OUTPUTS = triton.next_power_of_2(num_outputs)
    
    # Launch kernel with 2D grid
    grid = (num_outputs, max_batch_size)
    
    split_firstdim_kernel[grid](
        input_tensor,
        accept_length,
        output_tensor,
        num_outputs,
        max_batch_size,
        hidden_size,
        total_batch,
        BLOCK_SIZE_HIDDEN,
        BLOCK_OUTPUTS
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
    accept_len_ptr,
    batch_size,
    total_size,
    fill_value,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_BATCH: tl.constexpr
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    
    # 1. Compute offsets internally
    offs_batch = tl.arange(0, BLOCK_BATCH)
    mask_batch = offs_batch < batch_size
    lens = tl.load(accept_len_ptr + offs_batch, mask=mask_batch, other=0)
    offsets = tl.cumsum(lens, 0) - lens
    total_accepted = tl.sum(lens)

    # 2. Copy Logic
    if pid < batch_size:
        # Read source start and length
        src_start = tl.load(kv_indptr_ptr + pid)
        length = tl.load(accept_len_ptr + pid)
        
        # Read destination start
        dst_start = tl.sum(tl.where(offs_batch == pid, offsets, 0))
        
        # Loop to copy (handle length > BLOCK_SIZE)
        for off in range(0, length, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < length
            
            # Read
            val = tl.load(x_ptr + src_start + cols, mask=mask)
            
            # Write
            tl.store(x_ptr + dst_start + cols, val, mask=mask)

    # 3. Fill Tail Logic
    tail_start = total_accepted
    tail_len = total_size - tail_start
    
    if tail_len > 0:
        # Distribute tail filling among all blocks
        items_per_pid = (tail_len + num_pids - 1) // num_pids
        my_start = tail_start + pid * items_per_pid
        my_end = min(tail_start + (pid + 1) * items_per_pid, total_size)
        
        for off in range(my_start, my_end, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < my_end
            tl.store(x_ptr + cols, fill_value, mask=mask)


def compact_accepted_tokens(
    x: torch.Tensor,
    kv_indptr: torch.Tensor,
    accept_length: torch.Tensor,
    fill_value: int = -1
) -> None:
    batch_size = kv_indptr.shape[0] - 1
    total_size = x.numel()
    
    # 2. Launch parallel copy kernel
    # One block per batch is usually enough for draft tokens
    # Use a reasonable block size, e.g., 256, enough to cover most draft lengths
    BLOCK_SIZE = 256 
    BLOCK_BATCH = 128
    if batch_size > BLOCK_BATCH:
        BLOCK_BATCH = triton.next_power_of_2(batch_size)
    
    # Ensure enough blocks for filling if batch_size is small
    grid_size = max(batch_size, 32)
    grid = (grid_size,)
    
    compact_accepted_tokens_kernel[grid](
        x,
        kv_indptr,
        accept_length,
        batch_size,
        total_size,
        fill_value,
        BLOCK_SIZE=BLOCK_SIZE,
        BLOCK_BATCH=BLOCK_BATCH
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
    batch_size,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_BATCH: tl.constexpr
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
    # src_base = tl.load(cum_accept_length_ptr + pid)
    
    # Compute offsets internally
    offs_batch = tl.arange(0, BLOCK_BATCH)
    mask_batch = offs_batch < batch_size
    lens = tl.load(accept_length_ptr + offs_batch, mask=mask_batch, other=0)
    offsets = tl.cumsum(lens, 0) - lens
    
    src_base = tl.sum(tl.where(offs_batch == pid, offsets, 0))
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
    
    # cum_accept_length = torch.cat([torch.zeros(1, device=device, dtype=torch.int64), accept_length.cumsum(dim=0)])
    
    grid = (batch_size,)
    BLOCK_SIZE = 128 
    BLOCK_BATCH = 128
    if batch_size > BLOCK_BATCH:
        BLOCK_BATCH = triton.next_power_of_2(batch_size)
    
    update_kv_indices_kernel[grid](
        future_kvindice,
        kv_indices,
        kv_indptr,
        accept_length,
        batch_size,
        BLOCK_SIZE,
        BLOCK_BATCH
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


@triton.jit
def update_eagle_inputs_kernel(
    # Dest pointers
    verified_id_ptr, spec_pos_ptr, spec_loc_ptr, 
    spec_kv_indptr_ptr, spec_kv_indices_ptr, spec_qo_indptr_ptr, 
    hidden_states_ptr,
    pos_ids_ptr, spec_kv_indices_mtd_ptr, accept_len_ptr, input_ids_ptr,
    qo_indptr_ptr, kv_indptr_ptr, kv_indices_ptr, out_cache_loc_ptr, mask_indptr_ptr, seq_lens_ptr,
    
    # Src pointers
    src_verified_id_ptr, src_spec_pos_ptr, src_spec_loc_ptr, 
    src_spec_kv_indptr_ptr, src_spec_kv_indices_ptr, src_spec_qo_indptr_ptr, 
    src_hidden_states_ptr,
    src_pos_ids_ptr, src_spec_kv_indices_mtd_ptr, src_accept_len_ptr, src_input_ids_ptr,
    src_qo_indptr_ptr, src_kv_indptr_ptr, src_kv_indices_ptr, src_out_cache_loc_ptr, src_mask_indptr_ptr, src_seq_lens_ptr,
    
    # Sizes
    token_nums, pad_token_nums, batch_size, 
    ind_size, ind_size_mtd, ind_size_verify,
    draft_tokens_expand, hidden_size,
    
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    grid_size = tl.num_programs(0)
    
    # Block 0 handles small metadata copies
    if pid == 0:
        # 1. verified_id: Copy [0, token_nums), Fill 0 [token_nums, pad_token_nums)
        for i in range(0, pad_token_nums, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = offsets < pad_token_nums
            is_copy = offsets < token_nums
            val = tl.load(src_verified_id_ptr + offsets, mask=mask & is_copy, other=0)
            tl.store(verified_id_ptr + offsets, val, mask=mask)
            
        # 2. spec_position_ids_extend
        for i in range(0, pad_token_nums, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = offsets < pad_token_nums
            is_copy = offsets < token_nums
            val = tl.load(src_spec_pos_ptr + offsets, mask=mask & is_copy, other=0)
            tl.store(spec_pos_ptr + offsets, val, mask=mask)
            
        # 3. spec_out_cache_loc
        for i in range(0, pad_token_nums, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = offsets < pad_token_nums
            is_copy = offsets < token_nums
            val = tl.load(src_spec_loc_ptr + offsets, mask=mask & is_copy, other=0)
            tl.store(spec_loc_ptr + offsets, val, mask=mask)
            
        # 4. spec_kv_indptr (batch_size + 1)
        bs_1 = batch_size + 1
        for i in range(0, bs_1, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = offsets < bs_1
            val = tl.load(src_spec_kv_indptr_ptr + offsets, mask=mask)
            tl.store(spec_kv_indptr_ptr + offsets, val, mask=mask)
            
        # 5. spec_qo_indptr
        for i in range(0, bs_1, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = offsets < bs_1
            val = tl.load(src_spec_qo_indptr_ptr + offsets, mask=mask)
            tl.store(spec_qo_indptr_ptr + offsets, val, mask=mask)
            
        # 6. position_ids (batch_size)
        for i in range(0, batch_size, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = offsets < batch_size
            val = tl.load(src_pos_ids_ptr + offsets, mask=mask)
            tl.store(pos_ids_ptr + offsets, val, mask=mask)
            
        # 7. accept_length (batch_size)
        for i in range(0, batch_size, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = offsets < batch_size
            val = tl.load(src_accept_len_ptr + offsets, mask=mask)
            tl.store(accept_len_ptr + offsets, val, mask=mask)
            
        # 8. input_ids (batch_size)
        for i in range(0, batch_size, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = offsets < batch_size
            val = tl.load(src_input_ids_ptr + offsets, mask=mask)
            tl.store(input_ids_ptr + offsets, val, mask=mask)
            
        # 9. qo_indptr (batch_size + 1)
        for i in range(0, bs_1, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = offsets < bs_1
            val = tl.load(src_qo_indptr_ptr + offsets, mask=mask)
            tl.store(qo_indptr_ptr + offsets, val, mask=mask)
            
        # 10. kv_indptr (batch_size + 1)
        for i in range(0, bs_1, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = offsets < bs_1
            val = tl.load(src_kv_indptr_ptr + offsets, mask=mask)
            tl.store(kv_indptr_ptr + offsets, val, mask=mask)
            
        # 11. out_cache_loc (batch_size * draft_tokens_expand)
        ocl_size = batch_size * draft_tokens_expand
        for i in range(0, ocl_size, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = offsets < ocl_size
            val = tl.load(src_out_cache_loc_ptr + offsets, mask=mask)
            tl.store(out_cache_loc_ptr + offsets, val, mask=mask)
            
        # 12. mask_indptr (batch_size + 1)
        for i in range(0, bs_1, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = offsets < bs_1
            val = tl.load(src_mask_indptr_ptr + offsets, mask=mask)
            tl.store(mask_indptr_ptr + offsets, val, mask=mask)
            
        # 13. seq_lens (batch_size)
        for i in range(0, batch_size, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = offsets < batch_size
            val = tl.load(src_seq_lens_ptr + offsets, mask=mask)
            tl.store(seq_lens_ptr + offsets, val, mask=mask)

    # Large tasks distributed among other blocks
    else:
        # 4 large tasks
        # 0: spec_kv_indices
        # 1: spec_kv_indices_mtd
        # 2: kv_indices
        # 3: hidden_states
        
        # Adjust pid to be 0-based for large tasks
        worker_id = pid - 1
        num_workers = grid_size - 1
        
        # Interleaved assignment
        # Task 0
        for i in range(worker_id * BLOCK_SIZE, ind_size, num_workers * BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = offsets < ind_size
            val = tl.load(src_spec_kv_indices_ptr + offsets, mask=mask)
            tl.store(spec_kv_indices_ptr + offsets, val, mask=mask)
            
        # Task 1
        for i in range(worker_id * BLOCK_SIZE, ind_size_mtd, num_workers * BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = offsets < ind_size_mtd
            val = tl.load(src_spec_kv_indices_mtd_ptr + offsets, mask=mask)
            tl.store(spec_kv_indices_mtd_ptr + offsets, val, mask=mask)
            
        # Task 2
        for i in range(worker_id * BLOCK_SIZE, ind_size_verify, num_workers * BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = offsets < ind_size_verify
            val = tl.load(src_kv_indices_ptr + offsets, mask=mask)
            tl.store(kv_indices_ptr + offsets, val, mask=mask)
            
        # Task 3: Hidden States
        hs_total = token_nums * hidden_size
        for i in range(worker_id * BLOCK_SIZE, hs_total, num_workers * BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = offsets < hs_total
            val = tl.load(src_hidden_states_ptr + offsets, mask=mask)
            tl.store(hidden_states_ptr + offsets, val, mask=mask)


def update_eagle_inputs(
    # Dest tensors
    verified_id, spec_pos, spec_loc, 
    spec_kv_indptr, spec_kv_indices, spec_qo_indptr, 
    hidden_states,
    pos_ids, spec_kv_indices_mtd, accept_len, input_ids,
    qo_indptr, kv_indptr, kv_indices, out_cache_loc, mask_indptr, seq_lens,
    
    # Src tensors
    src_verified_id, src_spec_pos, src_spec_loc, 
    src_spec_kv_indptr, src_spec_kv_indices, src_spec_qo_indptr, 
    src_hidden_states,
    src_pos_ids, src_spec_kv_indices_mtd, src_accept_len, src_input_ids,
    src_qo_indptr, src_kv_indptr, src_kv_indices, src_out_cache_loc, src_mask_indptr, src_seq_lens,
    
    # Sizes
    token_nums, pad_token_nums, batch_size, 
    ind_size, ind_size_mtd, ind_size_verify,
    draft_tokens_expand, hidden_size
):
    grid = (128, )
    BLOCK_SIZE = 256
    
    update_eagle_inputs_kernel[grid](
        # Dest pointers
        verified_id, spec_pos, spec_loc, 
        spec_kv_indptr, spec_kv_indices, spec_qo_indptr, 
        hidden_states,
        pos_ids, spec_kv_indices_mtd, accept_len, input_ids,
        qo_indptr, kv_indptr, kv_indices, out_cache_loc, mask_indptr, seq_lens,
        
        # Src pointers
        src_verified_id, src_spec_pos, src_spec_loc, 
        src_spec_kv_indptr, src_spec_kv_indices, src_spec_qo_indptr, 
        src_hidden_states,
        src_pos_ids, src_spec_kv_indices_mtd, src_accept_len, src_input_ids,
        src_qo_indptr, src_kv_indptr, src_kv_indices, src_out_cache_loc, src_mask_indptr, src_seq_lens,
        
        # Sizes
        token_nums, pad_token_nums, batch_size, 
        ind_size, ind_size_mtd, ind_size_verify,
        draft_tokens_expand, hidden_size,
        
        BLOCK_SIZE=BLOCK_SIZE
    )


@triton.jit
def multi_tensor_copy_kernel(
    src_ptrs_ptr,       # [N], int64, pointers to source tensors
    src_ranges_ptr,     # [N, 2], int32, [src_start, src_end] indices in source tensors
    out_ptr,            # Output buffer pointer
    hidden_size,
    num_tensors,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_TENSORS: tl.constexpr
):
    # pid_batch: which tensor to process
    pid_batch = tl.program_id(0)
    # pid_m: which row block within the tensor
    pid_m = tl.program_id(1)
    # pid_n: which col block within hidden_size
    pid_n = tl.program_id(2)
    
    # 1. Load all ranges to compute offsets locally
    # This avoids a separate kernel launch for prefix sum
    offs = tl.arange(0, BLOCK_TENSORS)
    mask = offs < num_tensors
    
    # Load starts and ends [N, 2]
    starts = tl.load(src_ranges_ptr + offs * 2, mask=mask, other=0)
    ends = tl.load(src_ranges_ptr + offs * 2 + 1, mask=mask, other=0)
    lengths = ends - starts
    
    # Exclusive scan for destination offsets
    offsets = tl.cumsum(lengths, 0) - lengths
    
    # Extract parameters for the current tensor (pid_batch)
    is_my_batch = (offs == pid_batch)
    dst_start = tl.sum(tl.where(is_my_batch, offsets, 0))
    src_start = tl.sum(tl.where(is_my_batch, starts, 0))
    src_end = tl.sum(tl.where(is_my_batch, ends, 0))
    
    num_rows = src_end - src_start
    
    # 2. Check if this block is within valid rows
    row_offset = pid_m * BLOCK_M
    if row_offset >= num_rows:
        return
    
    # 3. Load source base address
    # src_ptrs_ptr stores int64 addresses. We load it and cast to pointer.
    src_addr_int = tl.load(src_ptrs_ptr + pid_batch)
    
    # 4. Calculate column offset
    n_offset = pid_n * BLOCK_N
    if n_offset >= hidden_size:
        return

    rows = row_offset + tl.arange(0, BLOCK_M)
    cols = n_offset + tl.arange(0, BLOCK_N)
    
    # Masks
    row_mask = rows < num_rows
    col_mask = cols < hidden_size
    mask = row_mask[:, None] & col_mask[None, :]
    
    # Load from Source
    # src_ptr = src_base + (src_start + rows) * hidden_size + cols
    src_ptr = src_addr_int.to(tl.pointer_type(out_ptr.dtype.element_ty))
    src_offsets = (src_start + rows[:, None]) * hidden_size + cols[None, :]
    val = tl.load(src_ptr + src_offsets, mask=mask)
    
    # Store to Dest
    # dst_index = (dst_start + rows) * hidden_size + cols
    dst_offsets_val = (dst_start + rows[:, None]) * hidden_size + cols[None, :]
    tl.store(out_ptr + dst_offsets_val, val, mask=mask)


def copy_tensors_to_buffer_pytorch_reference(
    tensors: List[torch.Tensor],
    ranges: torch.Tensor, # [N, 2]
    out_buffer: torch.Tensor
) -> None:
    out = out_buffer.clone()
    offset = 0
    for i, t in enumerate(tensors):
        r = ranges[i]
        length = r[1] - r[0]
        out[offset : offset + length] = t[r[0]:r[1]]
        offset += length
    return out

def copy_tensors_to_buffer(
    tensors: List[torch.Tensor],
    ranges: torch.Tensor, # [N, 2]
    out_buffer: torch.Tensor,
    max_blocks_m: int = 8
) -> None:
    if not tensors:
        return
        
    num_tensors = len(tensors)
    assert ranges.shape[0] == num_tensors
    
    device = out_buffer.device
    hidden_size = out_buffer.shape[1]
    assert out_buffer.dim() == 2
    
    # 1. Extract data pointers
    ptr_list = [t.data_ptr() for t in tensors]
    # Use non_blocking=True to avoid CPU-GPU sync
    src_ptrs = torch.tensor(ptr_list, dtype=torch.int64, device="cpu", 
                            pin_memory=True).to(device, non_blocking=True)
    BLOCK_M = 8 
    BLOCK_N = 512 
    BLOCK_TENSORS = 128
    if num_tensors > BLOCK_TENSORS:
        BLOCK_TENSORS = triton.next_power_of_2(num_tensors)
    
    # Grid: [Num_Tensors, Max_Rows_Blocks, Num_Hidden_Blocks]
    grid_n = triton.cdiv(hidden_size, BLOCK_N)
    
    grid = (num_tensors, max_blocks_m, grid_n)
    
    multi_tensor_copy_kernel[grid](
        src_ptrs,
        ranges,
        out_buffer,
        hidden_size,
        num_tensors,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_TENSORS=BLOCK_TENSORS
    )
    return out_buffer.view(-1, hidden_size)


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
        
        match = True
        if isinstance(out_r, list) and isinstance(out_t, list):
            if len(out_r) != len(out_t):
                match = False
            else:
                for r, t in zip(out_r, out_t):
                    m = torch.allclose(r, t) if r.is_floating_point() else torch.equal(r, t)
                    if not m:
                        match = False
                        break
        else:
            if inplace_idx is not None: out_t = args_t[inplace_idx]
            if check_ref: out_r = check_ref(out_r)
            if isinstance(out_r, tuple): out_r = out_r[0]
                
            match = torch.allclose(out_r, out_t) if out_r.is_floating_point() else torch.equal(out_r, out_t)
            
        print(f"Match: {match}")
        if not match: return

        # Benchmark
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(400): ref_fn(*args)
        torch.cuda.synchronize()
        t_ref = time.time() - t0
        
        t0 = time.time()
        for _ in range(400): tri_fn(*args)
        torch.cuda.synchronize()
        t_tri = time.time() - t0
        print(f"Ref: {t_ref*400:.2f}ms | Tri: {t_tri*400:.2f}ms | Speedup: {t_ref/t_tri:.2f}x")

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

    # 6. Multi Tensor Copy (Gather & Pack)
    H = 3072
    out_buffer = torch.zeros((100, H), dtype=torch.float16, device=device)
    # Create tensors with enough data
    tensors = [torch.randn(20, H, dtype=torch.float16, device=device) for _ in range(3)]
    ranges = torch.tensor([[0, 2], [1, 4], [2, 4]], dtype=torch.int32, device=device)
    def copy_tri(ts, rs, out):
        copy_tensors_to_buffer(ts, rs, out, max_blocks_m=4)
        return out

    run_test("Multi Tensor Copy", copy_tensors_to_buffer_pytorch_reference, copy_tri, [tensors, ranges, out_buffer], inplace_idx=2)

    # 7. Split Last Dim
    B = 10
    accept_len = torch.tensor([2, 3, 1, 4, 2, 3, 1, 4, 2, 3], device=device)
    total_tokens = accept_len.sum().item()
    next_token_ids = torch.arange(total_tokens, device=device)
    
    # Create output buffer for testing
    max_tokens = accept_len.max().item()
    output_buffer = torch.empty(B, max_tokens, dtype=next_token_ids.dtype, device=device)
    
    run_test("Split Last Dim", split_lastdim_pytorch_reference, split_lastdim_async, [next_token_ids, accept_len, output_buffer])
