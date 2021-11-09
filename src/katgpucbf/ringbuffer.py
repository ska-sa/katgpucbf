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

"""Wraps :class:`spead2.recv.ChunkRingbuffer` with monitoring capabilities."""

import spead2.recv.asyncio

from .monitor import Monitor


class ChunkRingbuffer(spead2.recv.asyncio.ChunkRingbuffer):
    """Wraps :class:`spead2.recv.ChunkRingbuffer` with monitoring capabilities.

    When waiting for the next heap, it uses :meth:`.Monitor.with_state` to
    indicate that heaps are being waited for. Whenever a heap is retrieved,
    it updates the size of the queue.
    """

    def __init__(self, maxsize: int, *, name: str, task_name: str, monitor: Monitor) -> None:
        super().__init__(maxsize)
        monitor.event_qsize(name, 0, maxsize)
        self._name = name
        self._task_name = task_name
        self._monitor = monitor

    async def get(self) -> spead2.recv.Chunk:
        """Override base class method to use the monitor."""
        with self._monitor.with_state(self._task_name, f"wait {self._name}"):
            chunk = await super().get()
        # This doesn't give the full picture of changes to the buffer
        # depth because it is just taking samples. Nevertheless it should
        # be useful for diagnosing issues.
        self._monitor.event_qsize(self._name, self.qsize(), self.maxsize)
        return chunk
