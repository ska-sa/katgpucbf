"""Utilities for performance monitoring."""

from abc import ABC, abstractmethod
import asyncio
import json
from time import monotonic
from typing import TypeVar, Any


_M = TypeVar('_M', bound='Monitor')


class Monitor(ABC):
    def __init__(self) -> None:
        self._time_base = monotonic()

    def time(self) -> float:
        return monotonic() - self._time_base

    @abstractmethod
    def event_qsize(self, name: str, qsize: int, maxsize: int) -> None:
        pass

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
        self._file = open(filename, 'w')

    def close(self) -> None:
        super().close()
        self._file.close()

    def event_qsize(self, name: str, qsize: int, maxsize: int) -> None:
        json.dump(
            {
                'time': self.time(),
                'type': 'qsize',
                'name': name,
                'qsize': qsize,
                'maxsize': maxsize
            },
            self._file
        )
        print(file=self._file)


class NullMonitor(Monitor):
    def event_qsize(self, name: str, qsize: int, maxsize: int) -> None:
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
