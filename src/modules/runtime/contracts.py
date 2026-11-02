from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, TypeVar

T = TypeVar("T")


class MessageQueue(Protocol[T]):
    def put(self, obj: T, block: bool = True, timeout: float | None = None) -> None: ...

    def put_nowait(self, obj: T) -> None: ...

    def get(self, block: bool = True, timeout: float | None = None) -> T: ...

    def get_nowait(self) -> T: ...


class Runnable(Protocol):
    def run(self) -> None: ...


class PipelineUI(Protocol):
    @property
    def close_requested(self) -> bool: ...

    def run_ui(self) -> None: ...

    def destroy(self) -> None: ...

    def mark_finished(self) -> None: ...

    def schedule(
        self,
        delay_ms: int,
        callback: Callable[[], None],
    ) -> None: ...

    def update_ui(
        self,
        text: str,
        _value: int,
        *,
        max_steps: int | None = None,
        overall_value: int | None = None,
    ) -> None: ...


class PipelineUIFactory(Protocol):
    def __call__(
        self,
        *,
        max_value: int,
        title: str,
        overall_max_value: int | None,
        on_cancel: Callable[[], None],
    ) -> PipelineUI: ...
