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

"""Recv module."""

import logging
import time
from typing import AsyncGenerator, List, Optional

from aiokatcp import Sensor, SensorSet

from ..monitor import Monitor
from ._katfgpu.recv import Chunk, Ringbuffer, Stream
from .ringbuffer import AsyncRingbuffer

logger = logging.getLogger(__name__)


async def chunk_sets(
    streams: List[Stream], monitor: Monitor, sensors: Optional[SensorSet] = None
) -> AsyncGenerator[List[Chunk], None]:
    """Asynchronous generator yielding timestamp-matched sets of chunks.

    This code receives chunks of data from the C++-domain Ringbuffer, matches
    them by timestamp, and ``yield`` to the caller.

    The input streams must all share the same ringbuffer, and their array
    indices must match their ``pol`` attributes. Whenever the most recent chunk
    from each of the streams all have the same timestamp, they are yielded.
    Chunks that are not yielded are returned to their streams.

    Parameters
    ----------
    streams
        A list of stream objects - there should be only two of them, because
        each represents a polarisation.
    monitor
        Used for performance monitoring of the ringbuffer.
    sensors
        Sensors through which networking statistics will be reported (if provided).
    """
    n_pol = len(streams)
    buf: List[Optional[Chunk]] = [None] * n_pol  # Working buffer to match up pairs of chunks from both pols.
    ring = AsyncRingbuffer(streams[0].ringbuffer, monitor, "recv_ringbuffer", "run_receive")
    lost = 0
    first_timestamp = -1  # Updated to the actual first timestamp on the first chunk
    packet_samples = streams[0].packet_samples
    packet_bytes = packet_samples * streams[0].sample_bits // 8
    chunk_packets = streams[0].chunk_packets
    chunk_samples = streams[0].chunk_samples

    # `try`/`finally` block acting as a quick-and-dirty context manager,
    # to ensure that we clean up nicely after ourselves if we are stopped.
    try:
        async for chunk in ring:
            # Inspect the chunk we have just received.
            if first_timestamp == -1:
                # TODO: use chunk.present to determine the actual first timestamp
                first_timestamp = chunk.timestamp
            good = chunk.n_present
            lost += chunk_packets - good
            if good < chunk_packets:
                logger.debug(
                    "Received chunk: timestamp=%#x pol=%d (%d/%d, lost %d)",
                    chunk.timestamp,
                    chunk.pol,
                    good,
                    chunk_packets,
                    lost,
                )

            # Check whether we have a chunk already for this pol.
            old = buf[chunk.pol]
            if old is not None:
                logger.warning("Chunk not matched: timestamp=%#x pol=%d", old.timestamp, old.pol)
                # Chunk was passed by without getting used. Return to the pool.
                streams[chunk.pol].add_chunk(old)
                buf[chunk.pol] = None

            # Stick the chunk in the buffer.
            buf[chunk.pol] = chunk

            # If we have both chunks and they match up, then we can yield.
            if all(c is not None and c.timestamp == chunk.timestamp for c in buf):
                if sensors:
                    # Get explicit timestamp to ensure that all the updated sensors
                    # have the same timestamp.
                    sensor_timestamp = time.time()

                    def increment(sensor: Sensor, incr: int):
                        sensor.set_value(sensor.value + incr, timestamp=sensor_timestamp)

                    # mypy isn't smart enough to see that the list can't have Nones
                    # in it at this point.
                    buf_good = sum(c.n_present for c in buf)  # type: ignore
                    increment(sensors["input-heaps-total"], buf_good)
                    increment(sensors["input-chunks-total"], n_pol)
                    increment(sensors["input-bytes-total"], buf_good * packet_bytes)
                    # Determine how many heaps we expected to have seen by
                    # now, and subtract from it the number actually seen to
                    # determine the number missing. This accounts for both
                    # packets lost within chunks and lost chunks.
                    received_heaps = sensors["input-heaps-total"].value
                    expected_heaps = (chunk.timestamp + chunk_samples) * n_pol // packet_samples
                    missing = expected_heaps - received_heaps
                    if missing > sensors["input-missing-heaps-total"].value:
                        sensors["input-missing-heaps-total"].set_value(missing, timestamp=sensor_timestamp)

                # mypy isn't smart enough to see that the list can't have Nones
                # in it at this point.
                yield buf  # type: ignore
                # Empty the buffer again for next use.
                buf = [None] * n_pol
    finally:
        for c in buf:
            if c is not None:
                streams[c.pol].add_chunk(c)


__all__ = ["chunk_sets", "Stream", "Chunk", "Ringbuffer"]
