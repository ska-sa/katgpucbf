################################################################################
# Copyright (c) 2021-2025, National Research Foundation (SARAO)
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

"""Transmission of SPEAD data."""

import asyncio
import functools
import ipaddress
import itertools
import time
from collections.abc import Iterable, Sequence
from typing import Self

import numpy as np
import spead2.send.asyncio
import xarray as xr
from prometheus_client import Counter, Gauge

from .. import BYTE_BITS, spead
from ..utils import TimeConverter
from . import METRIC_NAMESPACE
from .shared_array import SharedArray

output_heaps_counter = Counter("output_heaps", "number of heaps transmitted", namespace=METRIC_NAMESPACE)
output_bytes_counter = Counter("output_bytes", "number of payload bytes transmitted", namespace=METRIC_NAMESPACE)
time_error_gauge = Gauge("time_error_s", "elapsed time minus expected elapsed time", namespace=METRIC_NAMESPACE)


class HeapSet:
    """Collection of heaps making up a signal.

    The heaps are split into two parts, each of which is preprocessed to
    allow efficient transmission.

    This class should normally be constructed with :meth:`create`.

    Parameters
    ----------
    data
        An xarray data set with the following variables:

        timestamps
            1D array of timestamps, big-endian 64-bit
        digitiser_status
            2D array of digitiser status values, big-endian 64-bit (indexed by
            polarisation and time)
        payload
            2D array of raw sample data (indexed by polarisation and time)
        heaps
            Heaps referencing the timestamps and payload

        The dimensions must be ``time``, ``pol`` and ``data``.
    """

    def __init__(self, data: xr.Dataset) -> None:
        if data.sizes["time"] < 2:
            raise ValueError("time dimension must have at least 2 elements")
        middle = data.sizes["time"] // 2
        self.data = data
        self.parts = [data.isel(time=np.s_[:middle]), data.isel(time=np.s_[middle:])]
        for part in self.parts:
            part.attrs["heap_reference_list"] = spead2.send.HeapReferenceList(part["heaps"].data.ravel().tolist())

    @classmethod
    def create(
        cls, timestamps: np.ndarray, n_substreams: Sequence[int], heap_size: int, digitiser_id: Sequence[int]
    ) -> Self:
        """
        Create from shape parameters.

        Parameters
        ----------
        timestamps
            The timestamp array to associate with the :class:`HeapSet` (must be
            big-endian 64-bit).
        n_substreams
            Number of substreams to distribute the heaps across, per polarisation
        heap_size
            Number of bytes of payload per heap
        digitiser_id
            Digitiser ID to insert into the packets, per polarisation (LSB should
            indicate polarisation)
        """
        assert len(n_substreams) == len(digitiser_id)
        n_pols = len(n_substreams)
        # TODO: make sure that this uses huge pages, as that is more
        # efficient for ibverbs.
        n = len(timestamps)
        shared_payload = SharedArray.create("dsim_payload", (n_pols, n, heap_size), np.uint8)
        payload = shared_payload.buffer
        heaps = []
        substream_offset = list(itertools.accumulate(n_substreams, initial=0))
        digitiser_id_items = [spead.make_immediate(spead.DIGITISER_ID_ID, dig_id) for dig_id in digitiser_id]
        digitiser_status = np.zeros((n_pols, n), dtype=spead.IMMEDIATE_DTYPE)
        for i in range(n):
            # The ... in indexing causes numpy to give a 0d array view, rather than
            # a scalar.
            heap_timestamp = timestamps[i, ...]
            cur_heaps = []
            timestamp_item = spead.make_immediate(spead.TIMESTAMP_ID, heap_timestamp)
            for j in range(n_pols):
                heap_status = digitiser_status[j, i, ...]
                digitiser_status_item = spead.make_immediate(spead.DIGITISER_STATUS_ID, heap_status)
                heap_payload = payload[j, i]
                heap = spead2.send.Heap(spead.FLAVOUR)
                heap.add_item(timestamp_item)
                heap.add_item(digitiser_id_items[j])
                heap.add_item(digitiser_status_item)
                heap.add_item(
                    spead2.Item(
                        spead.ADC_SAMPLES_ID,
                        "",
                        "",
                        shape=heap_payload.shape,
                        dtype=heap_payload.dtype,
                        value=heap_payload,
                    )
                )
                heap.repeat_pointers = True
                substream_index = substream_offset[j] + i % n_substreams[j]
                cur_heaps.append(spead2.send.HeapReference(heap, substream_index=substream_index))
            heaps.append(cur_heaps)
        data = xr.Dataset(
            {
                "timestamps": (["time"], timestamps),
                "payload": (["pol", "time", "data"], payload, {"shared_array": shared_payload}),
                "heaps": (["time", "pol"], heaps),
                "digitiser_status": (["pol", "time"], digitiser_status),
            }
        )
        return cls(data)


def _is_multicast(address: str) -> bool:
    """Determine whether an address is a multicast address.

    This makes the guess that anything that doesn't parse as an IP address is
    a DNS name and that DNS names will resolve to unicast addresses.
    """
    try:
        return ipaddress.ip_address(address).is_multicast
    except ValueError:
        return False


def make_stream_base(
    *,
    config: spead2.send.StreamConfig,
    endpoints: Iterable[tuple[str, int]],
    ttl: int,
    interface_address: str,
    ibv: bool = False,
    affinity: int = -1,
    memory_regions: list | None = None,
) -> "spead2.send.asyncio.AsyncStream":
    """Create a spead2 stream for sending.

    This is the low-level support for making either a data or a descriptor
    stream. Refer to :func:`make_stream` for explanations of the arguments.
    """
    endpoints_list = list(endpoints)
    thread_pool = spead2.ThreadPool(1, [] if affinity < 0 else [affinity])
    if ibv:
        ibv_config = spead2.send.UdpIbvConfig(
            endpoints=endpoints_list,
            interface_address=interface_address,
            ttl=ttl,
        )
        if memory_regions is not None:
            ibv_config.memory_regions = memory_regions
        return spead2.send.asyncio.UdpIbvStream(thread_pool, config, ibv_config)
    elif any(_is_multicast(endpoint[0]) for endpoint in endpoints_list):
        return spead2.send.asyncio.UdpStream(
            thread_pool, endpoints_list, config, ttl=ttl, interface_address=interface_address
        )
    else:
        return spead2.send.asyncio.UdpStream(thread_pool, endpoints_list, config)


def make_stream(
    *,
    endpoints: Iterable[tuple[str, int]],
    heap_sets: Iterable[HeapSet],
    n_pols: int,
    adc_sample_rate: float,
    heap_samples: int,
    sample_bits: int,
    max_heaps: int,
    ttl: int,
    interface_address: str,
    ibv: bool,
    affinity: int,
) -> "spead2.send.asyncio.AsyncStream":
    """Create a spead2 stream for sending.

    Parameters
    ----------
    endpoints
        Destinations (host and port) for all substreams
    n_pols
        Number of single-pol streams to send
    adc_sample_rate
        Sample rate for each single-pol stream, in Hz
    heap_samples
        Number of samples to send in each heap (each heap will be sent as a single packet)
    sample_bits
        Number of bits per sample
    max_heaps
        Maximum number of heaps that may be in flight at once
    ttl
        IP TTL field
    interface_address
        IP address of the interface from which to send the data
    ibv
        If true, use ibverbs for acceleration
    affinity
        If non-negative, bind the sending thread to this CPU core
    """
    preamble = 72  # SPEAD header, 4 standard item pointers, 4 application-specific item pointers
    heap_size = heap_samples * sample_bits // BYTE_BITS
    overhead_ratio = (heap_size + preamble) / heap_size
    config = spead2.send.StreamConfig(
        rate=adc_sample_rate * n_pols * sample_bits / BYTE_BITS * overhead_ratio,
        max_packet_size=heap_size + preamble,
        max_heaps=max_heaps,
    )
    return make_stream_base(
        config=config,
        endpoints=endpoints,
        ttl=ttl,
        interface_address=interface_address,
        ibv=ibv,
        affinity=affinity,
        memory_regions=[heap_set.data["payload"].data for heap_set in heap_sets],
    )


class Sender:
    """Manage sending packets."""

    def __init__(
        self,
        stream: "spead2.send.asyncio.AsyncStream",
        heap_set: HeapSet,
        heap_samples: int,
    ) -> None:
        self.stream = stream
        self.heap_set = heap_set
        self.heap_samples = heap_samples
        self.time_converter = TimeConverter(0.0, 1.0)  # Dummy value; run() will initialise
        # The futures serve two functions:
        # - prevent concurrent access to the timestamps while they're being sent
        # - limiting the amount of data in flight
        self._futures: list[asyncio.Future[int] | None] = [None] * len(heap_set.parts)
        self._running = True  # Set to false to start shutdown
        self._finished = asyncio.Event()
        # First timestamp that we haven't yet submitted to async_send_heaps
        # (value is a dummy; real initial value is set by run)
        self._next_timestamp = 0

    def halt(self) -> None:
        """Request :meth:`run` to stop, but do not wait for it."""
        self._running = False

    async def join(self) -> None:
        """Wait for :meth:`run` to finish.

        This does not cause it to stop: use :meth:`halt` for that.
        """
        await self._finished.wait()

    async def stop(self) -> None:
        """Stop :meth:`run` and wait for it to finish."""
        self.halt()
        await self.join()

    def _update_metrics(self, end_timestamp: int, heaps: int, bytes: int, _future: asyncio.Future) -> None:
        end_time = self.time_converter.adc_to_unix(end_timestamp)
        time_error_gauge.set(time.time() - end_time)
        output_heaps_counter.inc(heaps)
        output_bytes_counter.inc(bytes)

    async def run(self, first_timestamp: int, time_converter: TimeConverter) -> None:
        """Send heaps continuously."""
        self._next_timestamp = first_timestamp
        self.time_converter = time_converter
        # Prepare initial timestamps
        first_end_timestamp = first_timestamp + self.heap_set.data.sizes["time"] * self.heap_samples
        self.heap_set.data["timestamps"][:] = np.arange(
            first_timestamp,
            first_end_timestamp,
            self.heap_samples,
            dtype=spead.IMMEDIATE_DTYPE,
        )
        while self._running:
            for i, part in enumerate(self.heap_set.parts):
                await asyncio.sleep(0)  # ensure other tasks get time to run
                if self._futures[i] is not None:
                    await asyncio.shield(self._futures[i])  # type: ignore
                    # set_heaps may have swapped heap_set out from under us during
                    # the await, so re-initialise part.
                    part = self.heap_set.parts[i]
                    part["timestamps"] += self.heap_set.data.sizes["time"] * self.heap_samples
                send_future = self.stream.async_send_heaps(
                    part.attrs["heap_reference_list"], spead2.send.GroupMode.SERIAL
                )
                self._futures[i] = send_future
                self._next_timestamp += part.sizes["time"] * self.heap_samples
                send_future.add_done_callback(
                    functools.partial(
                        self._update_metrics, self._next_timestamp, part["heaps"].size, part["payload"].nbytes
                    )
                )

        for future in self._futures:
            if future is not None:
                await future
        self._finished.set()  # Wake up join()

    async def set_heaps(self, heap_set: HeapSet) -> int:
        """Switch out the heap set for a different one.

        This does not return until the payload of the previous :class:`HeapSet`
        is no longer in use (the timestamps may still be in use).

        The new heap_set must share timestamps with the old one.

        Returns
        -------
        timestamp
            First timestamp which will use the new heap set
        """
        if heap_set.data["timestamps"].data is not self.heap_set.data["timestamps"].data:
            raise ValueError("new heap set does not share timestamps with the old")
        old_futures = []
        for future in self._futures:
            if future is not None:
                old_futures.append(future)
        self.heap_set = heap_set
        timestamp = self._next_timestamp
        if old_futures:
            await asyncio.wait(old_futures)
        return timestamp
