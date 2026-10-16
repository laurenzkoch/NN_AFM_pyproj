from __future__ import annotations

from typing import Protocol, TypeVar

T = TypeVar("T")


class MessageQueue(Protocol[T]):
    def put(self, obj: T, block: bool = True, timeout: float | None = None) -> None: ...

    def put_nowait(self, obj: T) -> None: ...

    def get(self, block: bool = True, timeout: float | None = None) -> T: ...

    def get_nowait(self) -> T: ...


class Runnable(Protocol):
    def run(self) -> None: ...
