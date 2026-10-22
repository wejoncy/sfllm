"""Recurrent working/verify buffers and fixed prefix-checkpoint storage."""

import os

import torch

from sfllm.kernels.gdn import scatter_recurrent_state, update_recurrent_state_indices

from .storage import Slots


class StatePool:
    @staticmethod
    def entry_bytes(hidden_size, dtype, recurrent_states=()):
        return (hidden_size * dtype.itemsize
                + sum(t[:, 0].numel() * t.element_size() for t in recurrent_states))

    def __init__(self, capacity, max_requests, hidden_size, dtype, device,
                 recurrent_states=()):
        self.capacity = capacity
        self.slots = Slots(capacity + 1)
        self.hidden = torch.empty(
            (capacity + 1, hidden_size), dtype=dtype, device=device,
        )
        self.ssm = self.conv = None
        if recurrent_states:
            self.ssm, self.conv = (
                t.new_empty((t.shape[0], capacity + 1, *t.shape[2:]))
                for t in recurrent_states
            )
        self.prefill_hidden = self.hidden.new_empty(
            (max_requests, hidden_size),
        )

    def save_hidden(self, slots, rows=None):
        scatter_recurrent_state(self.prefill_hidden, self.hidden, slots,
                                source_indices=rows)


class GatedDeltaNetState:
    """Own GDN working/verify storage and all-layer checkpoint transfers.

    The model supplies resolved entry shapes and backends at runner startup.
    No model inspection, checkpoint matching, or eviction belongs here.
    """

    def __init__(self, backends, ssm_shape, conv_shape, *, max_requests,
                 spec_steps, ssm_dtype, dtype, device):
        self.backends = backends
        layers = len(backends)
        use_journal = (bool(spec_steps) and ssm_dtype == torch.float32
                       and os.environ.get('SFLLM_GDN_JOURNAL', '1') == '1')
        self.conv = torch.zeros((layers, max_requests + 1, *conv_shape),
                                dtype=dtype, device=device)
        self.ssm = torch.zeros(
            (layers, max_requests * (1 if use_journal else spec_steps + 1) + 1, *ssm_shape),
            dtype=ssm_dtype, device=device,
        )
        self.indices = torch.arange(1, max_requests + 1, dtype=torch.int32, device=device)
        self.current_slots = self.read_slots = self.write_slots = None
        self.conv_journal = self.journal = None
        self.checkpoint_ssm = self.checkpoint_conv = None
        if spec_steps:
            self.conv_journal = torch.empty(
                (layers, max_requests, spec_steps, conv_shape[0]), dtype=dtype, device=device,
            )
            if use_journal:
                heads, value_dim, key_dim = ssm_shape
                shape = (layers, max_requests, heads, spec_steps)
                self.journal = tuple(
                    torch.empty((*shape, *tail), dtype=torch.float32, device=device)
                    for tail in ((value_dim,), (key_dim,), ())
                )
            else:
                self.current_slots = (
                    (torch.arange(max_requests + 1, dtype=torch.int32, device=device) - 1)
                    * (spec_steps + 1) + 1
                )
                self.read_slots = torch.empty_like(self.indices)
                self.write_slots = torch.empty(
                    (max_requests, spec_steps), dtype=torch.int32, device=device,
                )

    def bind_checkpoints(self, ssm, conv):
        self.checkpoint_ssm, self.checkpoint_conv = ssm, conv
        for layer, backend in enumerate(self.backends):
            backend.checkpoint_states = ssm[layer], conv[layer]
            # Capture includes restores; eager misses select the zero-state path.
            backend.use_initial_state = True

    def tensor(self, values):
        return torch.tensor(values, dtype=torch.int32, device='cpu', pin_memory=True).to(self.ssm.device, non_blocking=True)

    def reuse_checkpoints(self, plan, restores, *, use_prefill_graph=False):
        """Bind initial/shared states, reading checkpoints or restoring work slots.

        Missing checkpoints start from zero; graph replay uses fixed work slots.
        """
        use_initial = use_prefill_graph or any(slot >= 0 for _, slot in restores)
        # Eager prefill can read immutable checkpoints without a working-pool restore.
        # Exact hits skip model.forward; graph capture retains its fixed restore path.
        read_checkpoints = not use_prefill_graph and plan is not None and use_initial
        sources = [slot for _, slot in restores]
        if read_checkpoints and plan.active:
            sources.extend(sources[row] for row, _, _ in plan.internal)
        contiguous = (bool(sources) and sources[0] >= 0
                      and all(slot == sources[0] + i for i, slot in enumerate(sources)))
        source_indices = self.tensor(sources) if read_checkpoints and not contiguous else None
        for layer, backend in enumerate(self.backends):
            initial = None
            if read_checkpoints:
                selection = slice(sources[0], sources[0] + len(sources)) if contiguous else slice(None)
                initial = (self.checkpoint_ssm[layer, selection],
                           self.checkpoint_conv[layer, selection], source_indices)
            backend.prefill_initial_states = initial
            backend.prefill_plan = plan if plan is not None and plan.active else None
            backend.use_initial_state = use_initial
        if not restores or not use_initial or read_checkpoints:
            return
        rows = self.tensor([row for row, _ in restores])
        source = self.tensor([slot for _, slot in restores])
        if self.write_slots is not None:
            # Exact hits skip model.forward, so select their base slot here too.
            update_recurrent_state_indices(
                self.current_slots, rows, self.read_slots, self.write_slots,
                prefill=True,
            )
        ssm_rows = self.current_slots[rows] if self.write_slots is not None else rows
        scatter_recurrent_state(self.checkpoint_ssm, self.ssm, ssm_rows,
                                source_indices=source, layered=True)
        scatter_recurrent_state(self.checkpoint_conv, self.conv, rows,
                                source_indices=source, layered=True)

    def save_checkpoints(self, batch):
        """Save prompt-end states; internal checkpoints are written during prefill."""
        plan = batch.prefill_plan
        if plan is None:
            return
        final = [(row, slot) for row, position, slot in plan.records
                 if position == batch[row].prefill_end]
        if not final:
            return
        rows = self.tensor([batch[row].request_index + 1 for row, _ in final])
        slots = self.tensor([slot for _, slot in final])
        source = self.current_slots[rows] if self.write_slots is not None else rows
        scatter_recurrent_state(self.ssm, self.checkpoint_ssm, slots,
                                source_indices=source, layered=True)
        scatter_recurrent_state(self.conv, self.checkpoint_conv, slots,
                                source_indices=rows, layered=True)
