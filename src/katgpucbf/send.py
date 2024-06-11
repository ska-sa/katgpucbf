################################################################################
# Copyright (c) 2022-2024, National Research Foundation (SARAO)
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
from typing import Iterable

import spead2.send.asyncio

logger = logging.getLogger(__name__)


class DescriptorSender:
    """Manage sending descriptors at regular intervals.

    The descriptors are first sent once immediately, then after `first_interval`
    seconds, then every `interval` seconds. Using a different `first_interval`
    makes it possible to stagger different senders so that their descriptors
    do not all arrive at a common receiver at the same time.

    The descriptors are sent with zero rate, which means they will not affect
    the timing of other packets in the same stream.

    Parameters
    ----------
    stream
        The stream to which the descriptor will be sent. It will be sent to all
        substreams simultaneously.
    descriptors
        The descriptor heap to send.
    interval
        Interval (in seconds) between sending descriptors.
    first_interval
        Delay (in seconds) immediately after starting. If not specified, it
        defaults to `interval`.
    substreams
        Substream indices to which descriptors are sent. If not specified,
        send only to the first substream.
    """

    def __init__(
        self,
        stream: "spead2.send.asyncio.AsyncStream",
        descriptors: spead2.send.Heap,
        interval: float,
        first_interval: float | None = None,
        *,
        substreams: Iterable[int] = (0,),
    ) -> None:
        self._stream = stream
        self._heap_reference_list = spead2.send.HeapReferenceList(
            [spead2.send.HeapReference(descriptors, substream_index=i, rate=0.0) for i in substreams]
        )
        self._interval = interval
        self._first_interval = interval if first_interval is None else first_interval
        self._halt_event = asyncio.Event()

    async def _send_descriptors(self) -> None:
        logger.debug("Sending descriptors")
        await self._stream.async_send_heaps(heaps=self._heap_reference_list, mode=spead2.send.GroupMode.ROUND_ROBIN)

    def halt(self) -> None:
        """Request :meth:`run` to stop, but do not wait for it."""
        self._halt_event.set()

    async def run(self) -> None:
        """Send the descriptors indefinitely (use :meth:`halt` or cancel to stop)."""
        t = self._first_interval
        loop = asyncio.get_running_loop()
        deadline = loop.time()
        while not self._halt_event.is_set():
            await self._send_descriptors()
            # Compute absolute time to send the next one (this ensure that there is
            # no systematic drift).
            deadline += t
            # Turn into a relative time. Ensure we always sleep for a small
            # interval, even if we fell behind.
            delay = max(t * 0.01, deadline - loop.time())
            try:
                # wait_for will time out if _halt_event is not set by the deadline.
                await asyncio.wait_for(self._halt_event.wait(), timeout=delay)
            except asyncio.TimeoutError:
                pass
            t = self._interval


def send_rate(
    packet_header: int,
    packet_payload: int,
    heap_payload: int,
    heap_interval: float,
    send_rate_factor: float,
) -> float:
    """Compute the send rate (in bytes per second) to pass to spead2.

    Parameters
    ----------
    packet_header
        Overhead bytes in each SPEAD packet, including the SPEAD header (but
        excluding UDP/IP etc headers)
    packet_payload
        Number of payload bytes that should be included in each packet
    heap_payload
        Number of payload bytes in each heap
    heap_interval
        Time (in seconds) between sending heaps (or 0 for as fast as possible)
    send_rate_factor
        Safety factor by which the transmission rate should exceed the
        incoming data rate
    """
    if heap_interval == 0.0:
        return 0.0
    packets_per_heap = (heap_payload + packet_payload - 1) // packet_payload
    heap_overhead = packets_per_heap * packet_header
    return (heap_payload + heap_overhead) / heap_interval * send_rate_factor
