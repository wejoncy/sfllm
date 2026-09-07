class HasBatchState:
    """Model capability for state that follows requests across batch reordering."""

    def prepare_batch_state(self, scheduled_batch) -> None:
        raise NotImplementedError
