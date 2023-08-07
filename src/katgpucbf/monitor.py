################################################################################
# Copyright (c) 2020-2021, National Research Foundation (SARAO)
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy
# of the License at
#
#   https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

"""Monitor classes allowing for rudimentary performance monitoring.

Queues in the form of :class:`asyncio.Queue` are used for synchronisation
between coroutines in :mod:`katgpucbf.fgpu`, but we may like to know a bit more about
what's happening to them as items are pushed and popped. These metrics help us
to see what bottlenecks there are, because if the queues get full (or the "free"
queues get empty) it will result in dropped packets.
"""

import asyncio
import contextlib
import json
import threading
from abc import ABC, abstractmethod
from collections.abc import Generator
from time import monotonic
from typing import Any, TypeVar

_M = TypeVar("_M", bound="Monitor")


class Monitor(ABC):
    """Base class for performance monitors.

    Subclasses can create :class:`Queue` objects which report their size when it
    changes via the mechanism defined in the derived :class:`Monitor` class.

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
    def with_state(self, name: str, state: str, return_state: str = "other") -> Generator[None, None, None]:
        """Set a state for the duration of a block."""
        self.event_state(name, state)
        yield
        self.event_state(name, return_state)

    def make_queue(self, name: str, maxsize: int = 0) -> asyncio.Queue:
        """Create a :class:`Queue` that reports its size via this :class:`Monitor`."""
        self.event_qsize(name, 0, maxsize)
        return Queue(self, name, maxsize)

    def close(self) -> None:  # noqa: B027
        """Close the Monitor.

        In the base class this does nothing, but if derived classes implement
        something that needs to close cleanly (such as an output file), then
        this function can be overridden to do that. It is called when you
        ``__exit__`` from using the :class:`Monitor` as a context manager.
        """
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
        self._file = open(filename, "w")

    def close(self) -> None:
        """Close the output file cleanly."""
        super().close()
        self._file.close()

    def _event(self, data) -> None:
        with self._lock:
            json.dump(data, self._file)
            print(file=self._file)

    def event_qsize(self, name: str, qsize: int, maxsize: int) -> None:  # noqa: D102
        self._event({"time": self.time(), "type": "qsize", "name": name, "qsize": qsize, "maxsize": maxsize})

    def event_qsize_delta(self, name: str, delta: int) -> None:  # noqa: D102
        self._event({"time": self.time(), "type": "qsize-delta", "name": name, "delta": delta})

    def event_state(self, name: str, state: str) -> None:  # noqa: D102
        self._event({"time": self.time(), "type": "state", "name": name, "state": state})


class NullMonitor(Monitor):
    """A do-nothing monitor that presents the required interface."""

    def event_qsize(self, name: str, qsize: int, maxsize: int) -> None:  # noqa: D102
        pass

    def event_qsize_delta(self, name: str, delta: int) -> None:  # noqa: D102
        pass

    def event_state(self, name: str, state: str) -> None:  # noqa: D102
        pass

    def make_queue(self, name: str, maxsize: int = 0) -> asyncio.Queue:  # noqa: D102
        return asyncio.Queue(maxsize)


class Queue(asyncio.Queue):
    """Extend  :class:`asyncio.Queue` with performance monitoring.

    The only functionality added by any of the overridden functions is to
    call :meth:`~Monitor.event_qsize` upon put/get events, transmitting an event
    to the parent :class:`Monitor` object, alerting it about the change.
    """

    def __init__(self, monitor: Monitor, name: str, maxsize: int = 0):
        super().__init__(maxsize)
        self.monitor = monitor
        self.name = name

    def put_nowait(self, item: object) -> None:  # noqa: D102
        super().put_nowait(item)
        self.monitor.event_qsize(self.name, self.qsize(), self.maxsize)

    def get_nowait(self) -> Any:  # noqa: D102
        item = super().get_nowait()
        self.monitor.event_qsize(self.name, self.qsize(), self.maxsize)
        return item

    async def put(self, item: object) -> None:  # noqa: D102
        await super().put(item)
        self.monitor.event_qsize(self.name, self.qsize(), self.maxsize)

    async def get(self) -> Any:  # noqa: D102
        item = await super().get()
        self.monitor.event_qsize(self.name, self.qsize(), self.maxsize)
        return item
