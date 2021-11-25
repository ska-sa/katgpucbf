"""Transmission of SPEAD data."""

import itertools
from typing import Iterable, Sequence, Tuple

import numpy as np
import spead2.send.asyncio
import xarray as xr

from .. import BYTE_BITS, spead


def make_heap_set(n: int, n_substreams: Sequence[int], heap_size: int, digitiser_id: Sequence[int]) -> xr.Dataset:
    """
    Create a set of heaps from shape parameters.

    Parameters
    ----------
    n
        Number of heaps along the time axis
    n_substreams
        Number of substreams to distribute the heaps across, per polarisation
    heap_size
        Number of bytes of payload per heap
    digitiser_id
        Digitiser ID to insert into the packets, per polarisation (LSB should
        indicate polarisation)

    Returns
    -------
    heapset
        An xarray data set with the following variables:

        timestamps
            1D array of timestamps, big-endian 64-bit
        payload
            2D array of raw sample data (indexed by polarisation and time)
        heaps
            Heaps referencing the timestamps and payload
    """
    assert len(n_substreams) == len(digitiser_id)
    n_pols = len(n_substreams)
    # TODO: make sure that this uses huge pages, as that is more
    # efficient for ibverbs.
    payload = np.zeros((n_pols, n, heap_size), np.uint8)
    timestamps = np.zeros(n, ">u8")
    heaps = []
    substream_offset = list(itertools.accumulate(n_substreams, initial=0))
    for i in range(n):
        # The ... in indexing causes numpy to give a 0d array view, rather than
        # a scalar.
        heap_timestamp = timestamps[i, ...]
        cur_heaps = []
        for j in range(n_pols):
            heap_payload = payload[j, i]
            heap = spead2.send.Heap(spead.FLAVOUR)
            heap.add_item(spead.make_immediate(spead.TIMESTAMP_ID, heap_timestamp))
            heap.add_item(spead.make_immediate(spead.DIGITISER_ID_ID, digitiser_id[j]))
            heap.add_item(spead.make_immediate(spead.DIGITISER_STATUS_ID, 0))
            heap.add_item(
                spead2.Item(
                    spead.RAW_DATA_ID, "", "", shape=heap_payload.shape, dtype=heap_payload.dtype, value=heap_payload
                )
            )
            heap.repeat_pointers = True
            substream_index = substream_offset[j] + i % n_substreams[j]
            cur_heaps.append(spead2.send.HeapReference(heap, substream_index=substream_index))
        heaps.append(cur_heaps)
    return xr.Dataset(
        {
            "timestamps": (["time"], timestamps),
            "payload": (["pol", "time", "data"], payload),
            "heaps": (["time", "pol"], heaps),
        }
    )


def make_stream(
    *,
    endpoints: Iterable[Tuple[str, int]],
    heap_sets: Iterable[xr.Dataset],
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
            memory_regions=[heap_set["payload"].data for heap_set in heap_sets],
        )
        return spead2.send.asyncio.UdpIbvStream(thread_pool, config, ibv_config)
    else:
        return spead2.send.asyncio.UdpStream(
            thread_pool, list(endpoints), config, ttl=ttl, interface_address=interface_address
        )
