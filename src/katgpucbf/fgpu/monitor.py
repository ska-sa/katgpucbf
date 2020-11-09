"""Utilities for performance monitoring."""

from abc import ABC, abstractmethod
import asyncio
import contextlib
import json
import threading
from time import monotonic
from typing import Generator, TypeVar, Any


_M = TypeVar('_M', bound='Monitor')


class Monitor(ABC):
    """Base class for performance monitor backends.

    Each subclass will need to override the abstract methods to record the
    performance events.
    """

    def __init__(self) -> None:
        self._time_base = monotonic()

    def time(self) -> float:
        """Get a timestamp, relative to the creation time of the monitor."""
        return monotonic() - self._time_base

    @abstractmethod
    def event_qsize(self, name: str, qsize: int, maxsize: int) -> None:
        """Report the size and capacity of a queue.

        The queue `name` has current size `qsize` and capacity `maxsize`.
        All calls with the same name must report the same `maxsize`.
        """

    @abstractmethod
    def event_qsize_delta(self, name: str, delta: int) -> None:
        """Report addition/removal of items from a queue.

        The queue `name` has `delta` new items in it (or removed if `delta`
        is negative). This is an alternative to using :meth:`event_qsize`
        when there is no easy way to obtain the absolute size of the queue.
        There must have been a previous call to :meth:`event_qsize` to
        specify the initial capacity.
        """

    @abstractmethod
    def event_state(self, name: str, state: str) -> None:
        """Report the current state of a task.

        The state ``other`` is conventional when no more specific information is
        available.
        """

    @contextlib.contextmanager
    def with_state(self, name: str, state: str,
                   return_state: str = 'other') -> Generator[None, None, None]:
        """Set a state for the duration of a block."""
        self.event_state(name, state)
        yield
        self.event_state(name, return_state)

    def make_queue(self, name: str, maxsize: int = 0) -> asyncio.Queue:
        """Create a queue that reports its size.

        The returned queue is a subclass of :class:`asyncio.Queue` that calls
        :meth:`event_qsize` on each change.
        """
        self.event_qsize(name, 0, maxsize)
        return Queue(self, name, maxsize)

    def close(self) -> None:
        """Close any files or other OS resources."""
        pass

    def __enter__(self: _M) -> _M:
        return self

    def __exit__(self, *args) -> None:
        self.close()


class FileMonitor(Monitor):
    """Write events to a file.

    The file contains JSON-formatted records, one per line. Each record
    contains ``time`` and ``type`` keys, with additional type-specific
    information corresponding to the arguments to the notification functions.
    """

    def __init__(self, filename: str) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._file = open(filename, 'w')

    def close(self) -> None:
        super().close()
        self._file.close()

    def _event(self, data) -> None:
        with self._lock:
            json.dump(data, self._file)
            print(file=self._file)

    def event_qsize(self, name: str, qsize: int, maxsize: int) -> None:
        self._event(
            {
                'time': self.time(),
                'type': 'qsize',
                'name': name,
                'qsize': qsize,
                'maxsize': maxsize
            }
        )

    def event_qsize_delta(self, name: str, delta: int) -> None:
        self._event(
            {
                'time': self.time(),
                'type': 'qsize-delta',
                'name': name,
                'delta': delta
            }
        )

    def event_state(self, name: str, state: str) -> None:
        self._event(
            {
                'time': self.time(),
                'type': 'state',
                'name': name,
                'state': state
            }
        )


class NullMonitor(Monitor):
    """A do-nothing monitor that presents the required interface."""

    def event_qsize(self, name: str, qsize: int, maxsize: int) -> None:
        pass

    def event_qsize_delta(self, name: str, delta: int) -> None:
        pass

    def event_state(self, name: str, state: str) -> None:
        pass

    def make_queue(self, name: str, maxsize: int = 0) -> asyncio.Queue:
        return asyncio.Queue(maxsize)


class Queue(asyncio.Queue):
    """Wrap a :class:`asyncio.Queue` to call :meth:`Monitor.event_qsize`."""

    def __init__(self, monitor: Monitor, name: str, maxsize: int = 0):
        super().__init__(maxsize)
        self.monitor = monitor
        self.name = name

    def put_nowait(self, item: object) -> None:
        super().put_nowait(item)
        self.monitor.event_qsize(self.name, self.qsize(), self.maxsize)

    def get_nowait(self) -> Any:
        item = super().get_nowait()
        self.monitor.event_qsize(self.name, self.qsize(), self.maxsize)
        return item

    async def put(self, item: object) -> None:
        await super().put(item)
        self.monitor.event_qsize(self.name, self.qsize(), self.maxsize)

    async def get(self) -> Any:
        item = await super().get()
        self.monitor.event_qsize(self.name, self.qsize(), self.maxsize)
        return item
