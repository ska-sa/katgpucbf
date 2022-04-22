"""A few handy things intended for correlator qualification."""
import logging
from dataclasses import dataclass
from typing import List, Tuple, Union

import aiokatcp
import numba
import numpy as np
import scipy
import spead2
import spead2.recv
import spead2.recv.asyncio
from katsdptelstate.endpoint import Endpoint, endpoint_parser
from numba import types
from spead2.numba import intp_to_voidptr
from spead2.recv.numba import chunk_place_data

logger = logging.getLogger(__name__)
CPLX = 2


@dataclass
class Band:
    """Holds presets for a known band."""

    adc_sample_rate: float
    centre_frequency: float


BANDS = {
    "l": Band(adc_sample_rate=1712e6, centre_frequency=1284e6),
    "u": Band(adc_sample_rate=1088e6, centre_frequency=816e6),
}


async def get_sensor_val(client: aiokatcp.Client, sensor_name: str) -> Union[int, float, str]:
    """Get the value of a katcp sensor.

    If the sensor value can't be cast as an int or a float (in that order), the
    value will get returned as a string. This simple implementation ignores the
    actual type advertised by the server.
    """
    _reply, informs = await client.request("sensor-value", sensor_name)

    expected_types = [int, float, str]
    for t in expected_types:
        try:
            return aiokatcp.decode(t, informs[0].arguments[4])
        except ValueError:
            continue


async def get_dsim_endpoint(pc_client: Endpoint, adc_sample_rate: float) -> Endpoint:
    """Get the katcp address for a dsim on a product controller.

    The assumption is made that a single dsim is used, with the name
    ``sim.m800`` suffixed with the ADC sample rate.
    """
    return endpoint_parser(None)(await get_sensor_val(pc_client, f"sim.m800.{int(adc_sample_rate)}.0.port"))


class CorrelatorRemoteControl:
    """A container class for katcp clients needed by qualification tests."""

    def __init__(self, product_controller_client: aiokatcp.Client, dsim_client: aiokatcp.Client) -> None:
        self.product_controller_client = product_controller_client
        self.dsim_client = dsim_client


def create_stream(
    interface_address: str,
    multicast_endpoints: List[Tuple[str, int]],
    n_bls: int,
    n_chans: int,
    n_chans_per_substream: int,
    n_bits_per_sample: int,
    n_spectra_per_acc: int,
    int_time: float,
    n_samples_between_spectra: int,
    use_ibv: bool = False,
) -> spead2.recv.ChunkRingStream:
    """Create a spead2 recv stream."""
    # Lifted from :class:`katgpucbf.xbgpu.XSend`.
    HEAP_PAYLOAD_SIZE = n_chans_per_substream * n_bls * CPLX * n_bits_per_sample // 8  # noqa: N806
    HEAPS_PER_CHUNK = n_chans // n_chans_per_substream  # noqa: N806

    # According to the ICD.
    TIMESTAMP = 0x1600  # noqa: N806
    FREQUENCY = 0x4103  # noqa: N806

    # Needed for placing the individual heaps within the chunk.
    items = [FREQUENCY, TIMESTAMP, spead2.HEAP_LENGTH_ID]
    timestamp_step = n_samples_between_spectra * n_spectra_per_acc

    # Heap placement function. Gets compiled so that spead2's C code can call it.
    # A chunk consists of all channels and all baselines for a single point in time.
    @numba.cfunc(types.void(types.CPointer(chunk_place_data), types.uintp), nopython=True)
    def chunk_place(data_ptr, data_size):
        data = numba.carray(data_ptr, 1)
        items = numba.carray(intp_to_voidptr(data[0].items), 3, dtype=np.int64)
        channel_offset = items[0]
        timestamp = items[1]
        payload_size = items[2]
        # If the payload size doesn't match, discard the heap (could be descriptors etc).
        if payload_size == HEAP_PAYLOAD_SIZE:
            data[0].chunk_id = timestamp // timestamp_step
            data[0].heap_index = channel_offset // n_chans_per_substream
            data[0].heap_offset = data[0].heap_index * HEAP_PAYLOAD_SIZE

    stream_config = spead2.recv.StreamConfig(substreams=HEAPS_PER_CHUNK)

    # Assuming X-engines are at most 1 second out of sync with each other, with
    # one extra chunk for luck. May need to revisit that assumption for much
    # larger array sizes.
    max_chunks = round(1 // int_time) + 1

    chunk_stream_config = spead2.recv.ChunkStreamConfig(
        items=items,
        max_chunks=max_chunks,
        place=scipy.LowLevelCallable(chunk_place.ctypes, signature="void (void *, size_t)"),
    )

    free_ringbuffer = spead2.recv.ChunkRingbuffer(max_chunks)
    data_ringbuffer = spead2.recv.asyncio.ChunkRingbuffer(max_chunks)

    stream = spead2.recv.ChunkRingStream(
        spead2.ThreadPool(),
        stream_config,
        chunk_stream_config,
        data_ringbuffer,
        free_ringbuffer,
    )

    for _ in range(max_chunks):
        chunk = spead2.recv.Chunk(
            present=np.empty(HEAPS_PER_CHUNK, np.uint8),
            data=np.empty((n_chans, n_bls, CPLX), dtype=getattr(np, f"int{n_bits_per_sample}")),
        )
        stream.add_free_chunk(chunk)

    if use_ibv:
        config = spead2.recv.UdpIbvConfig(
            endpoints=multicast_endpoints, interface_address=interface_address, buffer_size=int(16e6), comp_vector=-1
        )
        stream.add_udp_ibv_reader(config)
    else:
        for ep in multicast_endpoints:
            stream.add_udp_reader(*ep, interface_address=interface_address)

    return stream
