from threading import Event


class PipelineCancelledError(RuntimeError):
    """The user requested cancellation of the pipeline."""


class CancellationToken:
    """Share a cancellation request between GUI and service threads."""

    def __init__(self) -> None:
        self._event: Event = Event()

    def request(self) -> None:
        self._event.set()

    def is_requested(self) -> bool:
        return self._event.is_set()

    def raise_if_requested(self) -> None:
        if self.is_requested():
            raise PipelineCancelledError("Pipeline cancelled by user.")
