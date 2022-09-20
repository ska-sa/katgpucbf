################################################################################
# Copyright (c) 2022, National Research Foundation (SARAO)
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

"""Shared utilities for sending data over SPEAD."""

import asyncio
import logging

import spead2.send.asyncio

logger = logging.getLogger(__name__)


class DescriptorSender:
    """Manage sending descriptors at regular intervals.

    The descriptors are first sent once immediately, then after `first_interval`
    seconds, then every `interval` seconds. Using a different `first_interval`
    makes it possible to stagger different senders so that their descriptors
    do not all arrive at a common receiver at the same time.

    Parameters
    ----------
    stream
        The stream to which the descriptor will be sent. It will be sent to all
        substreams simultaneously.
    descriptors
        The descriptor heap to send.
    first_interval
        Delay (in seconds) immediately after starting.
    interval
        Interval (in seconds) between sending descriptors.
    """

    def __init__(
        self,
        stream: "spead2.send.asyncio.AsyncStream",
        descriptors: spead2.send.Heap,
        first_interval: float,
        interval: float,
    ) -> None:
        self._stream = stream
        self._heap_reference_list = spead2.send.HeapReferenceList(
            [spead2.send.HeapReference(descriptors, substream_index=i) for i in range(stream.num_substreams)]
        )
        self._first_interval = first_interval
        self._interval = interval

    async def _send_descriptors(self) -> None:
        logger.debug("Sending descriptors")
        await self._stream.async_send_heaps(heaps=self._heap_reference_list, mode=spead2.send.GroupMode.ROUND_ROBIN)

    async def run(self) -> None:
        """Send the descriptors indefinitely (cancel to stop)."""
        await self._send_descriptors()
        t = self._first_interval
        while True:
            # Send the descriptor heap concurrently with the sleep, so that if
            # it takes a while (because there is data to send) it doesn't delay
            # the next iteration.
            await asyncio.gather(asyncio.sleep(t), self._send_descriptors())
            t = self._interval
