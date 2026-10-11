class HasBatchState:
    """Model capability for state that follows requests across batch reordering."""

    def init_batch_state(self, server_args):
        """Create cache-owned runtime storage and bind layer views before warmup."""
        raise NotImplementedError

    def prepare_batch_state(self, scheduled_batch) -> None:
        raise NotImplementedError

    def commit_speculative_state(self, accepted_steps) -> None:
        raise NotImplementedError
