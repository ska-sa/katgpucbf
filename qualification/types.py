"""Utilities for typing."""

from collections.abc import Awaitable, Callable
from typing import Protocol


class AsyncRunner(Protocol):
    """Protocol for a callable that takes another callable and runs it asynchronously."""

    def __call__[*A, R](self, func: Callable[[*A], R], *args: *A) -> Awaitable[R]: ...  # noqa: D102
