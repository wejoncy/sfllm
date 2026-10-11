"""Slot ownership and fixed KV storage with explicit token shape and execution space."""

from math import prod
from typing import NamedTuple

import numpy as np
import torch


class Slots:
    def __init__(self, capacity, *, reserved=1, refcounted=True):
        if not 0 <= reserved <= capacity:
            raise ValueError('Reserved slots must fit within capacity')
        self.capacity = capacity
        self.reserved = reserved
        self.refs = np.zeros((2, capacity), dtype=np.int32) if refcounted else None
        self.free = np.arange(capacity - 1, reserved - 1, -1, dtype=np.int64)
        self.free_count = capacity - reserved

    def alloc(self, count, references=1):
        if count < 0:
            raise ValueError('Slot count must be nonnegative')
        if count > self.free_count:
            raise MemoryError(f"Need {count} slots; {self.free_count} available")
        start = self.free_count - count
        ids = self.free[start:self.free_count][::-1].copy()
        self.free_count = start
        if self.refs is not None:
            self.refs[0, ids] = references
        return ids

    def retain(self, ids, *, cache=False, unique=False):
        if self.refs is None:
            raise RuntimeError("Sharing is disabled for this slot pool")
        ids = np.asarray(ids, dtype=np.int64)
        if np.any((ids < self.reserved) | (ids >= self.capacity)):
            raise ValueError("Slot index outside the allocated pool")
        if np.any((self.refs[0, ids] == 0) & (self.refs[1, ids] == 0)):
            raise RuntimeError("Cannot retain released slots")
        if unique:
            self.refs[int(cache), ids] += 1
        else:
            np.add.at(self.refs[int(cache)], ids, 1)

    def release(self, ids, *, cache=False, unique=False):
        ids = np.asarray(ids, dtype=np.int64)
        if not ids.size:
            return
        if np.any((ids < self.reserved) | (ids >= self.capacity)):
            raise ValueError("Slot index outside the allocated pool")
        freed = ids
        if self.refs is not None:
            # One request/prefix owns distinct slots. A batch of shared windows can
            # repeat slots and must count every ownership reference instead.
            multiplicity = 1
            if not unique and ids.size > 1 and not np.all(ids[1:] > ids[:-1]):
                ids, multiplicity = np.unique(ids, return_counts=True)
            counts = self.refs[int(cache)]
            if np.any(counts[ids] < multiplicity):
                raise RuntimeError("Slot released without an ownership reference")
            counts[ids] -= multiplicity
            freed = ids[(self.refs[0, ids] == 0) & (self.refs[1, ids] == 0)]
        self.free[self.free_count:self.free_count + freed.size] = freed[::-1]
        self.free_count += freed.size

    def sort_free(self):
        # Chunked releases hit quicksort's slow path; integer stable sort avoids it.
        self.free[:self.free_count] = np.sort(self.free[:self.free_count], kind="stable")[::-1]

    def __len__(self):
        return self.free_count


class KVWrite(NamedTuple):
    layers: slice
    key: torch.Tensor
    value: torch.Tensor
    locations: torch.Tensor | None = None
    rows: torch.Tensor | slice | None = None
    secondary: tuple | None = None


class KVStorage:
    def __init__(self, layers, token_shape, heads, key_dim, value_dim, dtype, device,
                 reserved=(0, 0)):
        before, after = reserved
        tokens = prod(token_shape)
        self.buffers = tuple(
            torch.zeros((layers, before + tokens + after, heads, dim), dtype=dtype, device=device)
            for dim in (key_dim, value_dim)
        )
        # Full: [layer, token, head, dim]. Window: [layer, slot, token, head, dim].
        # Reserved execution slots stay addressable by existing attention kernels.
        self.key, self.value = (
            t[:, before:before + tokens].view(layers, *token_shape, heads, t.shape[-1])
            for t in self.buffers
        )

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in self.buffers)
