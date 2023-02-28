################################################################################
# Copyright (c) 2022-2023, National Research Foundation (SARAO)
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
import asyncio
import logging
import re
from collections.abc import AsyncGenerator, Mapping
from dataclasses import dataclass, field
from typing import Literal, overload
from uuid import UUID, uuid4

import aiokatcp
import async_timeout
import numba
import numpy as np
import scipy
import spead2
import spead2.recv
import spead2.recv.asyncio
from katsdptelstate.endpoint import endpoint_list_parser
from numba import types
from numpy.typing import NDArray
from spead2.numba import intp_to_voidptr
from spead2.recv.numba import chunk_place_data

import katgpucbf.recv
from katgpucbf import COMPLEX
from katgpucbf.utils import TimeConverter

logger = logging.getLogger(__name__)
DEFAULT_MAX_DELAY = 1000000  # Around 0.5-1ms, depending on band. Increase if necessary


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


@dataclass
class CorrelatorRemoteControl:
    """A container class for katcp clients needed by qualification tests."""

    name: str
    product_controller_client: aiokatcp.Client
    dsim_clients: list[aiokatcp.Client]
    config: dict  # JSON dictionary used to configure the correlator
    mode_config: dict  # Configuration values used for MeerKAT mode string
    sensor_watcher: aiokatcp.SensorWatcher
    uuid: UUID = field(default_factory=uuid4)

    @property
    def sensors(self) -> aiokatcp.SensorSet:  # noqa: D401
        """Current sensor values from the product controller.

        Note that if a command is issued to a dsim, there will be an unknown
        delay before any sensors that change as a result are visible in this
        sensor set, because it comes via the product controller. In such
        cases it may be necessary to directly query the dsim for the sensor
        value.
        """
        return self.sensor_watcher.sensors

    @classmethod
    async def connect(
        cls, name: str, host: str, port: int, config: Mapping, mode_config: dict
    ) -> "CorrelatorRemoteControl":
        """Connect to a correlator's product controller.

        The function connects and gathers sufficient metadata in order for the
        user to know how to use the correlator for whatever testing needs to be
        done.
        """
        pcc = aiokatcp.Client(host, port)
        sensor_watcher = aiokatcp.SensorWatcher(pcc)
        pcc.add_sensor_watcher(sensor_watcher)
        await sensor_watcher.synced.wait()  # Implicitly waits for connection too

        dsim_endpoints = []
        for sensor_name, sensor in sensor_watcher.sensors.items():
            if match := re.fullmatch(r"sim\.dsim(\d+)\.\d+\.0\.port", sensor_name):
                idx = int(match.group(1))
                dsim_endpoints.append((idx, sensor.value))
        assert dsim_endpoints
        dsim_endpoints.sort()  # sorts by index

        dsim_clients = []
        for _, endpoint in dsim_endpoints:
            dsim_clients.append(await aiokatcp.Client.connect(str(endpoint.host), endpoint.port))

        logger.info("Sensors synchronised; %d dsims found", len(dsim_clients))

        return CorrelatorRemoteControl(
            name=name,
            product_controller_client=pcc,
            dsim_clients=list(dsim_clients),
            config=dict(config),
            mode_config=mode_config,
            sensor_watcher=sensor_watcher,
        )

    async def steady_state_timestamp(self, *, max_delay: int = DEFAULT_MAX_DELAY) -> int:
        """Get a timestamp by which the system will be in a steady state.

        In other words, the effects of previous commands will be in place for
        data with this timestamp.

        Because delays affect timestamps, the caller must provide an upper
        bound on the delay of any F-engine. The default for this should be
        acceptable for most cases.
        """
        timestamp = 0
        # Although the dsim sensors will also appear in the product controller,
        # we can't rely on that due to a race condition: if we make a change
        # directly on the dsim, the subscription update it sends to the product
        # controller might not be received before we ask the product controller
        # for the sensor value. So we have to query every device server that we
        # make state changes though.
        clients = [self.product_controller_client] + self.dsim_clients
        responses = await asyncio.gather(
            *(client.request("sensor-value", r"/.*steady-state-timestamp$/") for client in clients)
        )
        for client, (_, informs) in zip(clients, responses):
            for inform in informs:
                # In theory there could be multiple sensors per inform, but aiokatcp
                # never does this because timestamps are seldom shared.
                sensor_value = int(inform.arguments[4])
                if client is not self.product_controller_client:
                    # values returned from the dsim do not account for delay,
                    # so need to be offset to get an output timestamp.
                    sensor_value += max_delay
                timestamp = max(timestamp, sensor_value)
        logger.debug("steady_state_timestamp: %d", timestamp)
        return timestamp

    async def close(self) -> None:
        """Shut down all the connections."""
        clients = self.dsim_clients + [self.product_controller_client]
        for client in clients:
            client.close()
        await asyncio.gather(*[client.wait_closed() for client in clients])

    async def dsim_time(self, dsim_idx: int = 0) -> float:
        """Get the current UNIX time, as reported by a dsim.

        This helps make tests independent of the clock on the machine running
        the test; it depends only on the dsims to be synchronised with each other.
        """
        reply, _ = await self.dsim_clients[dsim_idx].request("time")
        return aiokatcp.decode(float, reply[0])


class BaselineCorrelationProductsReceiver:
    """Wrap a receive stream with helper functions."""

    def __init__(
        self, correlator: CorrelatorRemoteControl, stream_name: str, interface_address: str, use_ibv: bool = False
    ) -> None:
        # Some metadata we know already from the config.
        acv_name = correlator.config["outputs"][stream_name]["src_streams"][0]
        self.n_inputs = len(correlator.config["outputs"][acv_name]["src_streams"])
        self.n_ants = self.n_inputs // 2
        self.n_chans = correlator.config["outputs"][acv_name]["n_chans"]
        self.input_labels = correlator.config["outputs"][acv_name]["input_labels"]

        # But some we don't. Note: these could be properties. But copying them up
        # front ensures we get an exception early if the sensor is missing.
        self.n_bls = correlator.sensors[f"{stream_name}.n-bls"].value
        self.n_chans_per_substream = correlator.sensors[f"{stream_name}.n-chans-per-substream"].value
        self.n_bits_per_sample = correlator.sensors[f"{stream_name}.xeng-out-bits-per-sample"].value
        self.n_spectra_per_acc = correlator.sensors[f"{stream_name}.n-accs"].value
        self.int_time = correlator.sensors[f"{stream_name}.int-time"].value
        self.spectra_per_heap = correlator.sensors[f"{acv_name}.spectra-per-heap"].value
        self.n_samples_between_spectra = correlator.sensors[f"{acv_name}.n-samples-between-spectra"].value
        self.bls_ordering = ast.literal_eval(correlator.sensors[f"{stream_name}.bls-ordering"].value.decode())
        self.sync_time = correlator.sensors[f"{acv_name}.sync-time"].value
        self.scale_factor_timestamp = correlator.sensors[f"{acv_name}.scale-factor-timestamp"].value
        self.bandwidth = correlator.sensors[f"{acv_name}.bandwidth"].value
        self.multicast_endpoints = [
            (endpoint.host, endpoint.port)
            for endpoint in endpoint_list_parser(7148)(correlator.sensors[f"{stream_name}.destination"].value.decode())
        ]
        self.timestamp_step = self.n_samples_between_spectra * self.n_spectra_per_acc
        self.time_converter = TimeConverter(self.sync_time, self.scale_factor_timestamp)

        self.stream = create_baseline_correlation_product_receive_stream(
            interface_address,
            multicast_endpoints=self.multicast_endpoints,
            n_bls=self.n_bls,
            n_chans=self.n_chans,
            n_chans_per_substream=self.n_chans_per_substream,
            n_bits_per_sample=self.n_bits_per_sample,
            n_spectra_per_acc=self.n_spectra_per_acc,
            int_time=self.int_time,
            n_samples_between_spectra=self.n_samples_between_spectra,
            use_ibv=use_ibv,
        )
        self.correlator = correlator

    # The overloads ensure that when all_timestamps is known to be False, the
    # returned chunks are inferred to not be optional.
    @overload
    async def complete_chunks(
        self,
        min_timestamp: int | None = None,
        *,
        all_timestamps: Literal[False] = False,
        max_delay: int = DEFAULT_MAX_DELAY,
    ) -> AsyncGenerator[tuple[int, katgpucbf.recv.Chunk], None]:  # noqa: D102
        yield ...  # type: ignore

    @overload
    async def complete_chunks(
        self,
        min_timestamp: int | None = None,
        *,
        all_timestamps: bool = False,
        max_delay: int = DEFAULT_MAX_DELAY,
    ) -> AsyncGenerator[tuple[int, katgpucbf.recv.Chunk | None], None]:  # noqa: D102
        yield ...  # type: ignore

    async def complete_chunks(
        self,
        min_timestamp=None,
        *,
        all_timestamps=False,
        max_delay=DEFAULT_MAX_DELAY,
    ) -> AsyncGenerator[tuple[int, katgpucbf.recv.Chunk | None], None]:
        """Iterate over the complete chunks of the stream.

        Each yielded value is a ``(timestamp, chunk)`` pair.

        Parameters
        ----------
        min_timestamp
            Chunks with a timestamp less than this value are discarded. If the
            default of ``None`` is used, a value is computed via
            :meth:`CorrelatorRemoteControl.steady_state_timestamp`.
        all_timestamps
            If set to true (the default is false), discarded chunks still
            yield a ``(timestamp, None)`` pair.
        max_delay
            An upper bound on the delay set on any F-engine. This is used in
            the calculation of `min_timestamp` when no value is provided.
        """
        if min_timestamp is None:
            min_timestamp = await self.correlator.steady_state_timestamp(max_delay=max_delay)

        data_ringbuffer = self.stream.data_ringbuffer
        assert isinstance(data_ringbuffer, spead2.recv.asyncio.ChunkRingbuffer)
        async for chunk in data_ringbuffer:
            assert isinstance(chunk, katgpucbf.recv.Chunk)  # keeps mypy happy
            timestamp = chunk.chunk_id * self.timestamp_step
            if min_timestamp is not None and timestamp < min_timestamp:
                logger.debug("Skipping chunk with timestamp %d (< %d)", timestamp, min_timestamp)
            elif not np.all(chunk.present):
                logger.debug("Incomplete chunk %d", chunk.chunk_id)
            elif np.any(chunk.data == -(2**31)):
                logger.debug("Chunk with missing antenna(s)", chunk.chunk_id)
            else:
                yield timestamp, chunk
                continue
            # If we get here, the chunk is ignored
            chunk.recycle()
            if all_timestamps:
                yield timestamp, None
        return

    async def next_complete_chunk(
        self,
        min_timestamp: int | None = None,
        *,
        max_delay: int = DEFAULT_MAX_DELAY,
        timeout: float | None = 10.0,
    ) -> tuple[int, NDArray[np.int32]]:
        """Return the data from the next complete chunk from the stream.

        The return value includes the timestamp.

        Parameters
        ----------
        min_timestamp, max_delay
            See :meth:`complete_chunks`
        """
        async with async_timeout.timeout(timeout):
            async for timestamp, chunk in self.complete_chunks(min_timestamp=min_timestamp, max_delay=max_delay):
                with chunk:
                    return timestamp, np.array(chunk.data)  # Makes a copy before we return the chunk
        raise RuntimeError("stream was shut down before we received a complete chunk")

    async def consecutive_chunks(
        self,
        n: int,
        min_timestamp: int | None = None,
        *,
        max_delay: int = DEFAULT_MAX_DELAY,
        timeout: float | None = 10.0,
    ) -> list[tuple[int, katgpucbf.recv.Chunk]]:
        """Obtain `n` consecutive complete chunks from the stream.

        .. warning::

           This is not safe to use with large values of `n`, because the chunks
           are removed from the stream's pool. If you plan to use any value
           larger than 2 you should check that the free ring is initialised
           with enough chunks and update this comment.
        """
        chunks: list[tuple[int, katgpucbf.recv.Chunk]] = []
        async with async_timeout.timeout(timeout):
            async for timestamp, chunk in self.complete_chunks(min_timestamp=min_timestamp, all_timestamps=True):
                if chunk is None:
                    # Throw away failed attempt at getting an adjacent set
                    for _, old_chunk in chunks:
                        old_chunk.recycle()
                    chunks.clear()
                    continue
                chunks.append((timestamp, chunk))
                if len(chunks) == n:
                    return chunks
        raise RuntimeError(f"stream was shut down before we received {n} complete chunk(s)")


def create_baseline_correlation_product_receive_stream(
    interface_address: str,
    multicast_endpoints: list[tuple[str, int]],
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

    # Assuming X-engines are at most 500ms out of sync with each other, with
    # one extra chunk for luck. May need to revisit that assumption for much
    # larger array sizes.
    max_chunks = max(round(0.5 / int_time), 1) + 1
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
        chunk = katgpucbf.recv.Chunk(
            present=np.empty(HEAPS_PER_CHUNK, np.uint8),
            data=np.empty((n_chans, n_bls, COMPLEX), dtype=getattr(np, f"int{n_bits_per_sample}")),
            stream=stream,
        )
        chunk.recycle()

    if use_ibv:
        config = spead2.recv.UdpIbvConfig(
            endpoints=multicast_endpoints, interface_address=interface_address, buffer_size=int(16e6), comp_vector=-1
        )
        stream.add_udp_ibv_reader(config)
    else:
        for ep in multicast_endpoints:
            stream.add_udp_reader(*ep, interface_address=interface_address)

    return stream
