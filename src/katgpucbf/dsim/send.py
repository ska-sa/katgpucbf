"""Transmission of SPEAD data."""

import asyncio
import itertools
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import spead2.send.asyncio
import xarray as xr
from prometheus_client import Counter

from .. import BYTE_BITS, spead
from . import METRIC_NAMESPACE

output_heaps_counter = Counter("output_heaps", "number of heaps transmitted", namespace=METRIC_NAMESPACE)
output_bytes_counter = Counter("output_bytes", "number of payload bytes transmitted", namespace=METRIC_NAMESPACE)


class HeapSet:
    """Collection of heaps making up a signal.

    The heaps are split into two parts, each of which is preprocessed to
    allow efficient transmission.

    This class should normally be constructed with :meth:`factory`.

    Parameters
    ----------
    data
        An xarray data set with the following variables:

        timestamps
            1D array of timestamps, big-endian 64-bit
        payload
            2D array of raw sample data (indexed by polarisation and time)
        heaps
            Heaps referencing the timestamps and payload

        The dimensions must be ``time``, ``pol`` and ``data``.
    """

    def __init__(self, data: xr.Dataset) -> None:
        if data.dims["time"] < 2:
            raise ValueError("time dimension must have at least 2 elements")
        middle = data.dims["time"] // 2
        self.data = data
        self.parts = [data.isel(time=np.s_[:middle]), data.isel(time=np.s_[middle:])]
        for part in self.parts:
            part.attrs["heap_reference_list"] = spead2.send.HeapReferenceList(part["heaps"].data.ravel().tolist())

    @classmethod
    def create(
        cls, timestamps: np.ndarray, n_substreams: Sequence[int], heap_size: int, digitiser_id: Sequence[int]
    ) -> "HeapSet":
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
        payload = np.zeros((n_pols, n, heap_size), np.uint8)
        heaps = []
        substream_offset = list(itertools.accumulate(n_substreams, initial=0))
        digitiser_id_items = [spead.make_immediate(spead.DIGITISER_ID_ID, dig_id) for dig_id in digitiser_id]
        digitiser_status_item = spead.make_immediate(spead.DIGITISER_STATUS_ID, 0)
        for i in range(n):
            # The ... in indexing causes numpy to give a 0d array view, rather than
            # a scalar.
            heap_timestamp = timestamps[i, ...]
            cur_heaps = []
            timestamp_item = spead.make_immediate(spead.TIMESTAMP_ID, heap_timestamp)
            for j in range(n_pols):
                heap_payload = payload[j, i]
                heap = spead2.send.Heap(spead.FLAVOUR)
                heap.add_item(timestamp_item)
                heap.add_item(digitiser_id_items[j])
                heap.add_item(digitiser_status_item)
                heap.add_item(
                    spead2.Item(
                        spead.RAW_DATA_ID,
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
                "payload": (["pol", "time", "data"], payload),
                "heaps": (["time", "pol"], heaps),
            }
        )
        return cls(data)


def make_stream(
    *,
    endpoints: Iterable[Tuple[str, int]],
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
    thread_pool = spead2.ThreadPool(1, [] if affinity < 0 else [affinity])
    if ibv:
        ibv_config = spead2.send.UdpIbvConfig(
            endpoints=list(endpoints),
            interface_address=interface_address,
            ttl=ttl,
            memory_regions=[heap_set.data["payload"].data for heap_set in heap_sets],
        )
        return spead2.send.asyncio.UdpIbvStream(thread_pool, config, ibv_config)
    else:
        return spead2.send.asyncio.UdpStream(
            thread_pool, list(endpoints), config, ttl=ttl, interface_address=interface_address
        )


class Sender:
    """Manage sending packets."""

    def __init__(
        self, stream: "spead2.send.asyncio.AsyncStream", heap_set: HeapSet, first_timestamp: int, heap_samples: int
    ) -> None:
        self.stream = stream
        self.heap_set = heap_set
        self.heap_samples = heap_samples
        # The futures serve two functions:
        # - prevent concurrent access to the timestamps while they're being sent
        # - limiting the amount of data in flight
        self._futures: List[Optional[asyncio.Future[int]]] = [None] * len(heap_set.parts)
        self._running = True  # Set to false to start shutdown
        self._finished = asyncio.Event()
        # Prepare initial timestamps
        first_end_timestamp = first_timestamp + self.heap_set.data.dims["time"] * self.heap_samples
        heap_set.data["timestamps"][:] = np.arange(first_timestamp, first_end_timestamp, heap_samples, dtype=">u8")

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

    async def run(self) -> None:
        """Send heaps continuously."""
        while self._running:
            for i, part in enumerate(self.heap_set.parts):
                if self._futures[i] is not None:
                    await asyncio.shield(self._futures[i])  # type: ignore
                    part["timestamps"] += self.heap_set.data.dims["time"] * self.heap_samples
                self._futures[i] = self.stream.async_send_heaps(
                    part.attrs["heap_reference_list"], spead2.send.GroupMode.SERIAL
                )
                # Not actually sent yet, but close enough for monitoring the transmission speed
                output_heaps_counter.inc(part.dims["time"])
                output_bytes_counter.inc(part["payload"].nbytes)

        for future in self._futures:
            if future is not None:
                await future
        self._finished.set()  # Wake up join()
