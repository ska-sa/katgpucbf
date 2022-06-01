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

"""A few handy things intended for correlator qualification.

.. todo::

    This is a bit more than what I'm comfortable to inhabit an __init__.py file,
    but there's not enough really for it to make its way into a proper module.
    Maybe just a ``utils.py`` or something like that would be better.
"""
import ast
import logging
from dataclasses import dataclass
from typing import AsyncGenerator, List, Literal, Optional, Tuple, overload

import aiokatcp
import numba
import numpy as np
import scipy
import spead2
import spead2.recv
import spead2.recv.asyncio
from katsdptelstate.endpoint import Endpoint, endpoint_list_parser, endpoint_parser
from numba import types
from spead2.numba import intp_to_voidptr
from spead2.recv.numba import chunk_place_data

from katgpucbf import COMPLEX

logger = logging.getLogger(__name__)
DSIM_NAME = "sim.m800"


async def get_sensor_val(client: aiokatcp.Client, sensor_name: str):
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


async def get_dsim_endpoint(product_controller_client: aiokatcp.Client, adc_sample_rate: float) -> Endpoint:
    """Get the katcp address for a dsim on a product controller.

    The assumption is made that a single dsim is used, with its name suffixed
    with the ADC sample rate.
    """
    return endpoint_parser(None)(
        await get_sensor_val(product_controller_client, f"{DSIM_NAME}.{int(adc_sample_rate)}.0.port")  # type: ignore
    )


@dataclass
class CorrelatorRemoteControl:
    """A container class for katcp clients needed by qualification tests."""

    product_controller_client: aiokatcp.Client
    dsim_client: aiokatcp.Client
    n_ants: int
    n_inputs: int
    n_chans: int
    n_bls: int
    n_chans_per_substream: int
    n_bits_per_sample: int
    n_spectra_per_acc: int
    int_time: float
    n_samples_between_spectra: int
    bls_ordering: List[Tuple[str, str]]
    sync_time: float
    scale_factor_timestamp: float
    bandwidth: float
    multicast_endpoints: List[Tuple[str, int]]

    @classmethod
    async def connect(
        cls, pcc: aiokatcp.Client, dsim_client: aiokatcp.Client, correlator_config: dict
    ) -> "CorrelatorRemoteControl":
        """Connect to a correlator's product controller.

        The function connects and gathers sufficient metadata in order for the
        user to know how to use the correlator for whatever testing needs to be
        done.
        """
        # Some metadata we know already from the config.
        n_inputs = len(correlator_config["outputs"]["antenna_channelised_voltage"]["src_streams"])
        n_ants = n_inputs // 2
        n_chans = correlator_config["outputs"]["antenna_channelised_voltage"]["n_chans"]

        # But some can't.
        n_bls = await get_sensor_val(pcc, "baseline_correlation_products-n-bls")
        n_chans_per_substream = await get_sensor_val(pcc, "baseline_correlation_products-n-chans-per-substream")
        n_bits_per_sample = await get_sensor_val(pcc, "baseline_correlation_products-xeng-out-bits-per-sample")
        n_spectra_per_acc = await get_sensor_val(pcc, "baseline_correlation_products-n-accs")
        int_time = await get_sensor_val(pcc, "baseline_correlation_products-int-time")
        n_samples_between_spectra = await get_sensor_val(pcc, "antenna_channelised_voltage-n-samples-between-spectra")
        bls_ordering = ast.literal_eval(await get_sensor_val(pcc, "baseline_correlation_products-bls-ordering"))
        sync_time = await get_sensor_val(pcc, "antenna_channelised_voltage-sync-time")
        scale_factor_timestamp = await get_sensor_val(pcc, "antenna_channelised_voltage-scale-factor-timestamp")
        bandwidth = await get_sensor_val(pcc, "antenna_channelised_voltage-bandwidth")
        multicast_endpoints = [
            (endpoint.host, endpoint.port)
            for endpoint in endpoint_list_parser(7148)(
                await get_sensor_val(pcc, "baseline_correlation_products-destination")
            )
        ]

        return CorrelatorRemoteControl(
            product_controller_client=pcc,
            dsim_client=dsim_client,
            n_ants=n_ants,
            n_inputs=n_inputs,
            n_chans=n_chans,
            n_bls=n_bls,
            n_chans_per_substream=n_chans_per_substream,
            n_bits_per_sample=n_bits_per_sample,
            n_spectra_per_acc=n_spectra_per_acc,
            int_time=int_time,
            n_samples_between_spectra=n_samples_between_spectra,
            bls_ordering=bls_ordering,
            sync_time=sync_time,
            scale_factor_timestamp=scale_factor_timestamp,
            bandwidth=bandwidth,
            multicast_endpoints=multicast_endpoints,
        )


class BaselineCorrelationProductsReceiver:
    """Wrap a receive stream with helper functions."""

    def __init__(self, correlator: CorrelatorRemoteControl, interface_address: str, use_ibv: bool = False) -> None:
        self.stream = create_baseline_correlation_product_receive_stream(
            interface_address,
            multicast_endpoints=correlator.multicast_endpoints,
            n_bls=correlator.n_bls,
            n_chans=correlator.n_chans,
            n_chans_per_substream=correlator.n_chans_per_substream,
            n_bits_per_sample=correlator.n_bits_per_sample,
            n_spectra_per_acc=correlator.n_spectra_per_acc,
            int_time=correlator.int_time,
            n_samples_between_spectra=correlator.n_samples_between_spectra,
            use_ibv=use_ibv,
        )
        self.timestamp_step = correlator.n_samples_between_spectra * correlator.n_spectra_per_acc

    # The overloads ensure that when all_timestamps is known to be False, the
    # returned chunks are inferred to not be optional.
    @overload
    async def complete_chunks(
        self,
        min_timestamp: Optional[int] = None,
        all_timestamps: Literal[False] = False,
    ) -> AsyncGenerator[Tuple[int, spead2.recv.Chunk], None]:  # noqa: D102
        yield ...  # type: ignore

    @overload
    async def complete_chunks(
        self,
        min_timestamp: Optional[int] = None,
        all_timestamps: Optional[bool] = False,
    ) -> AsyncGenerator[Tuple[int, Optional[spead2.recv.Chunk]], None]:  # noqa: D102
        yield ...  # type: ignore

    async def complete_chunks(
        self,
        min_timestamp=None,
        all_timestamps=False,
    ) -> AsyncGenerator[Tuple[int, Optional[spead2.recv.Chunk]], None]:
        """Iterate over the complete chunks of the stream.

        Each yielded value is a ``(timestamp, chunk)`` pair.

        Parameters
        ----------
        min_timestamp
            If specified, chunks with a timestamp less then this value are
            discarded.
        all_timestamps
            If set to true (the default is false), discarded chunks still
            yield a ``(timestamp, None)`` pair.
        """
        data_ringbuffer = self.stream.data_ringbuffer
        assert isinstance(data_ringbuffer, spead2.recv.asyncio.ChunkRingbuffer)
        async for chunk in data_ringbuffer:
            assert isinstance(chunk.present, np.ndarray)  # keeps mypy happy
            timestamp = chunk.chunk_id * self.timestamp_step
            if min_timestamp is not None and timestamp < min_timestamp:
                logger.debug("Skipping chunk with timestamp %d (< %d)", timestamp, min_timestamp)
            elif not np.all(chunk.present):
                logger.debug("Incomplete chunk %d", chunk.chunk_id)
            else:
                yield timestamp, chunk
                continue
            # If we get here, the chunk is ignored
            self.stream.add_free_chunk(chunk)
            if all_timestamps:
                yield timestamp, None
        return

    async def next_complete_chunk(self, min_timestamp: Optional[int] = None) -> Tuple[int, spead2.recv.Chunk]:
        """Return the next complete chunk from the stream.

        The return value includes the timestamp.

        Parameters
        ----------
        min_timestamp
            If specified, chunks with a timestamp less then this value are
            discarded.
        """
        async for timestamp, chunk in self.complete_chunks(min_timestamp=min_timestamp):
            return timestamp, chunk
        assert False  # noqa: B011  # Tells mypy that this isn't reachable


def create_baseline_correlation_product_receive_stream(
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
    """Create a spead2 recv stream for ingesting baseline correlation product data."""
    # Lifted from :class:`katgpucbf.xbgpu.XSend`.
    HEAP_PAYLOAD_SIZE = n_chans_per_substream * n_bls * COMPLEX * n_bits_per_sample // 8  # noqa: N806
    HEAPS_PER_CHUNK = n_chans // n_chans_per_substream  # noqa: N806

    # According to the ICD.
    TIMESTAMP_ID = 0x1600  # noqa: N806
    FREQUENCY_ID = 0x4103  # noqa: N806

    # Needed for placing the individual heaps within the chunk.
    items = [FREQUENCY_ID, TIMESTAMP_ID, spead2.HEAP_LENGTH_ID]
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
    max_chunks = round(1 / int_time) + 1
    n_extra_chunks = 2  # A couple extra to make sure we have breathing room.
    chunk_stream_config = spead2.recv.ChunkStreamConfig(
        items=items,
        max_chunks=max_chunks,
        place=scipy.LowLevelCallable(chunk_place.ctypes, signature="void (void *, size_t)"),
    )

    free_ringbuffer = spead2.recv.ChunkRingbuffer(max_chunks + n_extra_chunks)
    data_ringbuffer = spead2.recv.asyncio.ChunkRingbuffer(max_chunks)

    stream = spead2.recv.ChunkRingStream(
        spead2.ThreadPool(),
        stream_config,
        chunk_stream_config,
        data_ringbuffer,
        free_ringbuffer,
    )

    for _ in range(max_chunks + n_extra_chunks):
        chunk = spead2.recv.Chunk(
            present=np.empty(HEAPS_PER_CHUNK, np.uint8),
            data=np.empty((n_chans, n_bls, COMPLEX), dtype=getattr(np, f"int{n_bits_per_sample}")),
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
