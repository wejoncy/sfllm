class HasBatchState:
    """Model capability for state that follows requests across batch reordering."""

    def prepare_batch_state(self, scheduled_batch) -> None:
        raise NotImplementedError

    def commit_speculative_state(self, accepted_steps) -> None:
        raise NotImplementedError
