"""Transmission of SPEAD data."""

from typing import Iterable, Tuple

import numpy as np
import spead2.send.asyncio

from .. import BYTE_BITS, spead


class HeapSet:
    """A collection of heaps backed by contiguous blocks of memory.

    The timestamps for the heaps can be efficiently updated as a block, as can
    the raw data.

    Parameters
    ----------
    n
        Number of heaps
    n_substreams
        Number of substreams to distribute the heaps across
    substream_offset
        First substream to use
    heap_size
        Number of bytes of payload per heap
    digitiser_id
        Digitiser ID to insert into the packets (LSB should indicate polarisation)
    """

    def __init__(self, n: int, n_substreams: int, substream_offset: int, heap_size: int, digitiser_id: int) -> None:
        # TODO: make sure that this uses huge pages, as that is more
        # efficient for ibverbs.
        payload = np.zeros((n, heap_size), np.uint8)
        self.payload = payload.ravel()
        self.timestamps = np.zeros(n, ">u8")
        self.heaps = []
        for i in range(n):
            # The ... in indexing causes numpy to give a 0d array view, rather than
            # a scalar.
            heap_timestamp = self.timestamps[i, ...]
            heap_payload = payload[i]
            heap = spead2.send.Heap(spead.FLAVOUR)
            heap.add_item(spead.make_immediate(spead.TIMESTAMP_ID, heap_timestamp))
            heap.add_item(spead.make_immediate(spead.DIGITISER_ID_ID, digitiser_id))
            heap.add_item(spead.make_immediate(spead.DIGITISER_STATUS_ID, 0))
            heap.add_item(
                spead2.Item(
                    spead.RAW_DATA_ID, "", "", shape=heap_payload.shape, dtype=heap_payload.dtype, value=heap_payload
                )
            )
            heap.repeat_pointers = True
            self.heaps.append(spead2.send.HeapReference(heap, substream_index=substream_offset + i % n_substreams))


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
) -> "spead2.send.asyncio.AsyncStream":
    """Create a spead2 stream for sending.

    TODO: document parameters
    """
    preamble = 72  # SPEAD header, 4 standard item pointers, 4 application-specific item pointers
    heap_size = heap_samples * sample_bits // BYTE_BITS
    overhead_ratio = (heap_size + preamble) / heap_size
    config = spead2.send.StreamConfig(
        rate=adc_sample_rate * n_pols * sample_bits / BYTE_BITS * overhead_ratio,
        max_packet_size=heap_size + preamble,
        max_heaps=max_heaps,
    )
    thread_pool = spead2.ThreadPool()
    if ibv:
        ibv_config = spead2.send.UdpIbvConfig(
            endpoints=list(endpoints),
            interface_address=interface_address,
            ttl=ttl,
            memory_regions=[heap_set.payload for heap_set in heap_sets],
        )
        return spead2.send.asyncio.UdpIbvStream(thread_pool, config, ibv_config)
    else:
        return spead2.send.asyncio.UdpStream(
            thread_pool, list(endpoints), config, ttl=ttl, interface_address=interface_address
        )
