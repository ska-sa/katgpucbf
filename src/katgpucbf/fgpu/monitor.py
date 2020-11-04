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
    def __init__(self) -> None:
        self._time_base = monotonic()

    def time(self) -> float:
        return monotonic() - self._time_base

    @abstractmethod
    def event_qsize(self, name: str, qsize: int, maxsize: int) -> None:
        pass

    @abstractmethod
    def event_qsize_delta(self, name: str, delta: int) -> None:
        pass

    @abstractmethod
    def event_state(self, name: str, state: str) -> None:
        pass

    @contextlib.contextmanager
    def with_state(self, name: str, state: str,
                   return_state: str = 'other') -> Generator[None, None, None]:
        self.event_state(name, state)
        yield
        self.event_state(name, return_state)

    def make_queue(self, name: str, maxsize: int = 0) -> asyncio.Queue:
        self.event_qsize(name, 0, maxsize)
        return Queue(self, name, maxsize)

    def close(self) -> None:
        pass

    def __enter__(self: _M) -> _M:
        return self

    def __exit__(self, *args) -> None:
        self.close()


class FileMonitor(Monitor):
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
    def event_qsize(self, name: str, qsize: int, maxsize: int) -> None:
        pass

    def event_qsize_delta(self, name: str, delta: int) -> None:
        pass

    def event_state(self, name: str, state: str) -> None:
        pass

    def make_queue(self, name: str, maxsize: int = 0) -> asyncio.Queue:
        return asyncio.Queue(maxsize)


class Queue(asyncio.Queue):
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
