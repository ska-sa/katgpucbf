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

"""A few handy things intended for CBF qualification.

.. todo::

    This is a bit more than what I'm comfortable to inhabit an __init__.py file,
    but there's not enough really for it to make its way into a proper module.
    Maybe just a ``utils.py`` or something like that would be better.
"""
import ast
import asyncio
import ctypes
import logging
import math
import re
from collections.abc import AsyncGenerator, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Literal, Sequence, overload
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
from katgpucbf import COMPLEX, DIG_SAMPLE_BITS
from katgpucbf.spead import BEAM_ANTS_ID, DEFAULT_PORT, FREQUENCY_ID, TIMESTAMP_ID
from katgpucbf.utils import TimeConverter

from .reporter import Reporter

logger = logging.getLogger(__name__)
DEFAULT_MAX_DELAY = 1000000  # Around 0.5-1ms, depending on band. Increase if necessary


@dataclass
class CBFRemoteControl:
    """A container class for katcp clients needed by qualification tests."""

    name: str
    product_controller_client: aiokatcp.Client
    dsim_clients: list[aiokatcp.Client]
    config: dict  # JSON dictionary used to configure the CBF
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
    async def connect(cls, name: str, host: str, port: int, config: Mapping, mode_config: dict) -> "CBFRemoteControl":
        """Connect to a CBF's product controller.

        The function connects and gathers sufficient metadata in order for the
        user to know how to use the CBF for whatever testing needs to be
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

        return CBFRemoteControl(
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

    async def dsim_gaussian(
        self, amplitude: float, pdf_report: Reporter | None = None, *, dsim_idx: int = 0, period: int | None = None
    ) -> None:
        """Configure a dsim with Gaussian noise.

        The identical signal is produced on both polarisations.

        Parameters
        ----------
        amplitude
            Standard deviation, in units of the LSB of the digitiser output
        pdf_report
            Reporter to which this process will be reported
        dsim_idx
            Index of the dsim to set
        period
            If specified, override the period of the dsim signal
        """
        if pdf_report is not None:
            pdf_report.step("Configure the D-sim with Gaussian noise.")
        dig_max = 2 ** (DIG_SAMPLE_BITS - 1) - 1
        amplitude /= dig_max  # Convert to be relative to full-scale
        signal = f"common=nodither(wgn({amplitude}));common;common;"
        if period is None:
            await self.dsim_clients[0].request("signals", signal)
            suffix = ""
        else:
            await self.dsim_clients[0].request("signals", signal, period)
            suffix = f" and period={period} samples"
        if pdf_report is not None:
            pdf_report.detail(f"Set D-sim with wgn amplitude={amplitude}{suffix}.")


class XBReceiver:
    """Base for :class:`BaselineCorrelationProductsReceiver` and :class:`TiedArrayChannelisedVoltageReceiver`."""

    # Attributes instantiated by the derived classes
    stream: spead2.recv.ChunkStreamRingGroup
    timestamp_step: int  # Step (in ADC samples) between chunk timestamps

    def __init__(self, cbf: CBFRemoteControl, stream_names: Sequence[str]) -> None:
        # Some metadata we know already from the config.
        acv_name = cbf.config["outputs"][stream_names[0]]["src_streams"][0]
        acv_config = cbf.config["outputs"][acv_name]
        self.stream_names = list(stream_names)
        self.n_inputs = len(acv_config["src_streams"])
        self.n_ants = self.n_inputs // 2
        self.n_chans = acv_config["n_chans"]
        self.input_labels = acv_config["input_labels"]
        if "narrowband" in acv_config:
            self.decimation_factor = acv_config["narrowband"]["decimation_factor"]
        else:
            self.decimation_factor = 1
        self.adc_sample_rate = cbf.config["outputs"][acv_config["src_streams"][0]]["adc_sample_rate"]

        # But some we don't. Note: these could be properties. But copying them up
        # front ensures we get an exception early if the sensor is missing.
        #
        # We assume the streams all have the same information except for addresses.
        self.n_chans_per_substream = cbf.sensors[f"{stream_names[0]}.n-chans-per-substream"].value
        self.n_spectra_per_heap = cbf.sensors[f"{acv_name}.spectra-per-heap"].value
        self.n_samples_between_spectra = cbf.sensors[f"{acv_name}.n-samples-between-spectra"].value
        self.sync_time = cbf.sensors[f"{acv_name}.sync-time"].value
        self.scale_factor_timestamp = cbf.sensors[f"{acv_name}.scale-factor-timestamp"].value
        self.bandwidth = cbf.sensors[f"{acv_name}.bandwidth"].value
        self.center_freq = cbf.sensors[f"{acv_name}.center-freq"].value
        self.multicast_endpoints = [
            [
                (endpoint.host, endpoint.port)
                for endpoint in endpoint_list_parser(DEFAULT_PORT)(
                    cbf.sensors[f"{stream_name}.destination"].value.decode()
                )
            ]
            for stream_name in stream_names
        ]

        self.time_converter = TimeConverter(self.sync_time, self.scale_factor_timestamp)
        self.cbf = cbf
        self._acv_name = acv_name

    # The overloads ensure that when all_timestamps is known to be False, the
    # returned chunks are inferred to not be optional.
    @overload
    async def complete_chunks(
        self,
        min_timestamp: int | None = None,
        *,
        all_timestamps: Literal[False] = False,
        max_delay: int = DEFAULT_MAX_DELAY,
        time_limit: float | None = None,
    ) -> AsyncGenerator[tuple[int, katgpucbf.recv.Chunk], None]:  # noqa: D102
        yield ...  # type: ignore

    @overload
    async def complete_chunks(
        self,
        min_timestamp: int | None = None,
        *,
        all_timestamps: bool = False,
        max_delay: int = DEFAULT_MAX_DELAY,
        time_limit: float | None = None,
    ) -> AsyncGenerator[tuple[int, katgpucbf.recv.Chunk | None], None]:  # noqa: D102
        yield ...  # type: ignore

    async def complete_chunks(
        self,
        min_timestamp=None,
        *,
        all_timestamps=False,
        max_delay=DEFAULT_MAX_DELAY,
        time_limit=None,
    ) -> AsyncGenerator[tuple[int, katgpucbf.recv.Chunk | None], None]:
        """Iterate over the complete chunks of the stream.

        Each yielded value is a ``(timestamp, chunk)`` pair.

        Parameters
        ----------
        min_timestamp
            Chunks with a timestamp less than this value are discarded. If the
            default of ``None`` is used, a value is computed via
            :meth:`CBFRemoteControl.steady_state_timestamp`.
        all_timestamps
            If set to true (the default is false), discarded chunks still
            yield a ``(timestamp, None)`` pair.
        max_delay
            An upper bound on the delay set on any F-engine. This is used in
            the calculation of `min_timestamp` when no value is provided.
        time_limit
            If a floating-point value is given, the iteration will end after
            this many seconds. Note that no :exc:`asyncio.TimeoutError` will be
            raised.
        """
        if min_timestamp is None:
            min_timestamp = await self.cbf.steady_state_timestamp(max_delay=max_delay)

        data_ringbuffer = self.stream.data_ringbuffer
        assert isinstance(data_ringbuffer, spead2.recv.asyncio.ChunkRingbuffer)
        try:
            async with async_timeout.timeout(time_limit) as timer:
                async for chunk in data_ringbuffer:
                    assert isinstance(chunk, katgpucbf.recv.Chunk)  # keeps mypy happy
                    timestamp = chunk.chunk_id * self.timestamp_step
                    if min_timestamp is not None and timestamp < min_timestamp:
                        logger.debug("Skipping chunk with timestamp %d (< %d)", timestamp, min_timestamp)
                    elif not np.all(chunk.present):
                        logger.debug("Incomplete chunk %d", chunk.chunk_id)
                    elif (chunk.data.dtype == np.dtype(np.int32) and np.any(chunk.data == -(2**31))) or (
                        chunk.extra is not None and np.min(chunk.extra) < self.n_ants
                    ):
                        logger.debug("Chunk with missing antenna(s) (%d)", chunk.chunk_id)
                    else:
                        yield timestamp, chunk
                        continue
                    # If we get here, the chunk is ignored
                    chunk.recycle()
                    if all_timestamps:
                        yield timestamp, None
        except asyncio.TimeoutError:
            if not timer.expired:
                raise  # The TimeoutError came from something else

    async def next_complete_chunk(
        self,
        min_timestamp: int | None = None,
        *,
        max_delay: int = DEFAULT_MAX_DELAY,
        timeout: float | None = 10.0,
    ) -> tuple[int, np.ndarray]:
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

    @overload
    def channel_frequency(self, channel: float) -> float:
        ...

    @overload
    def channel_frequency(self, channel: np.ndarray) -> np.ndarray:
        ...

    def channel_frequency(self, channel):
        """Compute the frequency (in Hz) for a given channel.

        The channel number may be a real value to select a frequency that is
        not at the centre of a channel bin. Integral values correspond to bin
        centres. This is the frequency of the signal received from the
        digitiser rather than sky frequency.

        Either a scalar or a numpy array may be used.
        """
        return (channel - self.n_chans / 2) * self.bandwidth / self.n_chans + self.center_freq

    def compute_tone_gain(self, amplitude: float, target_voltage: float) -> float:
        """Compute F-Engine gain.

        Compute gain to be applied to the F-Engine to maximise output dynamic range
        when the input is a tone (for example, for use with
        :func:`.sample_tone_response`). The F-Engine output is 8-bit signed
        (max 127).

        Parameters
        ----------
        amplitude
            Amplitude of the tones, on a scale of 0 to 1.
        target_voltage
            Desired magnitude of F-engine output values. The calculation uses
            an approximation, so the actual value may be slightly higher than
            the target. The target may also be reduced if necessary to avoid
            saturating the X-engine output.
        """
        dig_max = 2 ** (DIG_SAMPLE_BITS - 1) - 1
        # The PFB is scaled for fixed incoherent gain, but we need to be concerned
        # about coherent gain to avoid overflowing the F-engine output. Coherent gain
        # scales approximately with sqrt(bw / chan_bw / 2).
        return target_voltage / (amplitude * dig_max * np.sqrt(self.n_chans * self.decimation_factor / 2))


class BaselineCorrelationProductsReceiver(XBReceiver):
    """Wrap a baseline-correlation-products stream with helper functions."""

    def __init__(self, cbf: CBFRemoteControl, stream_name: str, interface_address: str, use_ibv: bool = False) -> None:
        super().__init__(cbf, [stream_name])

        # Fill in extra sensors specific to baseline-correlation-products
        self.n_bls = cbf.sensors[f"{stream_name}.n-bls"].value
        self.n_bits_per_sample = cbf.sensors[f"{stream_name}.xeng-out-bits-per-sample"].value
        self.n_spectra_per_acc = cbf.sensors[f"{stream_name}.n-accs"].value
        self.int_time = cbf.sensors[f"{stream_name}.int-time"].value
        self.bls_ordering = ast.literal_eval(cbf.sensors[f"{stream_name}.bls-ordering"].value.decode())
        self.timestamp_step = self.n_samples_between_spectra * self.n_spectra_per_acc

        self.stream = create_baseline_correlation_product_receive_stream(
            interface_address,
            multicast_endpoints=self.multicast_endpoints[0],
            n_bls=self.n_bls,
            n_chans=self.n_chans,
            n_chans_per_substream=self.n_chans_per_substream,
            n_bits_per_sample=self.n_bits_per_sample,
            n_spectra_per_acc=self.n_spectra_per_acc,
            int_time=self.int_time,
            n_samples_between_spectra=self.n_samples_between_spectra,
            use_ibv=use_ibv,
        )

    def compute_tone_gain(self, amplitude: float, target_voltage: float) -> float:  # noqa: D102
        # We need to avoid saturating the signed 32-bit X-engine accumulation as
        # well (2e9 is comfortably less than 2^31).
        target_voltage = min(target_voltage, np.sqrt(2e9 / self.n_spectra_per_acc))
        return super().compute_tone_gain(amplitude, target_voltage)

    if TYPE_CHECKING:
        # Just refine the return type, without any run-time implementation
        async def next_complete_chunk(
            self,
            min_timestamp: int | None = None,
            *,
            max_delay: int = DEFAULT_MAX_DELAY,
            timeout: float | None = 10.0,
        ) -> tuple[int, NDArray[np.int32]]:  # noqa: D102
            ...


class TiedArrayChannelisedVoltageReceiver(XBReceiver):
    """Wrap a tied-array-channelised-voltage stream with helper functions."""

    def __init__(
        self, cbf: CBFRemoteControl, stream_names: Sequence[str], interface_address: str, use_ibv: bool = False
    ) -> None:
        super().__init__(cbf, stream_names)

        self.n_bits_per_sample = cbf.sensors[f"{stream_names[0]}.beng-out-bits-per-sample"].value
        self.timestamp_step = self.n_samples_between_spectra * self.n_spectra_per_heap
        self.source_indices: list[list[int]] = [
            ast.literal_eval(cbf.sensors[f"{stream_name}.source-indices"].value.decode())
            for stream_name in stream_names
        ]

        self.stream = create_tied_array_channelised_voltage_receive_stream(
            interface_address,
            multicast_endpoints=self.multicast_endpoints,
            n_chans=self.n_chans,
            n_chans_per_substream=self.n_chans_per_substream,
            n_bits_per_sample=self.n_bits_per_sample,
            n_spectra_per_heap=self.n_spectra_per_heap,
            n_samples_between_spectra=self.n_samples_between_spectra,
            use_ibv=use_ibv,
        )


def _create_receive_stream_group(
    interface_address: str,
    multicast_endpoints: list[list[tuple[str, int]]],
    use_ibv: bool,
    stream_config: spead2.recv.StreamConfig,
    max_chunks: int,
    max_heap_extra: int,
    chunk_place: Callable,  # Actual type comes from ctypes.CFUNCTYPE, but it doesn't have a static name
    chunk_factory: Callable[[spead2.recv.ChunkRingPair], katgpucbf.recv.Chunk],
) -> spead2.recv.ChunkStreamRingGroup:
    """Create a stream group to receive data from an engine.

    Parameters
    ----------
    interface_address
        IP address of receiving interface
    multicast_endpoints
        List of list of (group, port) pairs. Each list corresponds to a single
        stream.
    use_ibv
        If true, use ibverbs.
    stream_config
        Configuration for individual streams. It *must* have `explicit_start` set
        to true.
    max_chunks
        Maximum number of chunks to have under construction at once. Some
        additional chunks are allocated to account for those in the data ringbuffer
        or being processed.
    max_heap_extra
        Maximum bytes to be written to the extra field per heap.
    chunk_place
        Chunk placement callback (compatible with
        :class:`scipy.LowLevelCallable`). It will receive the `frequency`,
        `timestamp`, `beam_ants` and heap length items. It must also accept a
        user data pointer, which will point to the stream index (indexing into
        `multicast_endpoints`) as an int64.
    chunk_factory
        Factory function to initialise the chunks.
    """
    n_extra_chunks = 2  # Chunks that are being processed
    free_ringbuffer = spead2.recv.ChunkRingbuffer(max_chunks + n_extra_chunks)
    data_ringbuffer = spead2.recv.asyncio.ChunkRingbuffer(n_extra_chunks)
    group_config = spead2.recv.ChunkStreamGroupConfig(
        eviction_mode=spead2.recv.ChunkStreamGroupConfig.EvictionMode.LOSSY,
        max_chunks=max_chunks,
    )
    group = spead2.recv.ChunkStreamRingGroup(group_config, data_ringbuffer, free_ringbuffer)
    for _ in range(free_ringbuffer.maxsize):
        chunk = chunk_factory(group)
        chunk.recycle()

    # Needed for placing the individual heaps within the chunk.
    items = [FREQUENCY_ID, TIMESTAMP_ID, BEAM_ANTS_ID, spead2.HEAP_LENGTH_ID]
    for i, endpoints in enumerate(multicast_endpoints):
        user_data = np.array([i], np.int64)
        chunk_stream_config = spead2.recv.ChunkStreamConfig(
            items=items,
            max_chunks=max_chunks,
            max_heap_extra=max_heap_extra,
            place=scipy.LowLevelCallable(
                chunk_place,
                signature="void (void *, size_t, void *)",
                user_data=user_data.ctypes.data_as(ctypes.c_void_p),
            ),
        )
        stream = group.emplace_back(spead2.ThreadPool(), stream_config, chunk_stream_config)

        if use_ibv:
            config = spead2.recv.UdpIbvConfig(
                endpoints=endpoints, interface_address=interface_address, buffer_size=int(16e6), comp_vector=-1
            )
            stream.add_udp_ibv_reader(config)
        else:
            for ep in endpoints:
                stream.add_udp_reader(*ep, interface_address=interface_address)

    for stream in group:
        stream.start()
    return group


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
) -> spead2.recv.ChunkStreamRingGroup:
    """Create a spead2 recv stream for ingesting baseline correlation product data."""
    # Lifted from :class:`katgpucbf.xbgpu.XSend`.
    HEAP_PAYLOAD_SIZE = n_chans_per_substream * n_bls * COMPLEX * n_bits_per_sample // 8  # noqa: N806
    HEAPS_PER_CHUNK = n_chans // n_chans_per_substream  # noqa: N806
    timestamp_step = n_samples_between_spectra * n_spectra_per_acc

    # Heap placement function. Gets compiled so that spead2's C code can call it.
    # A chunk consists of all channels and all baselines for a single point in time.
    @numba.cfunc(types.void(types.CPointer(chunk_place_data), types.size_t, types.CPointer(types.int64)), nopython=True)
    def chunk_place(data_ptr, data_size, user_data_ptr):
        data = numba.carray(data_ptr, 1)
        items = numba.carray(intp_to_voidptr(data[0].items), 4, dtype=np.int64)
        channel_offset = items[0]
        timestamp = items[1]
        payload_size = items[3]
        # If the payload size doesn't match, discard the heap (could be descriptors etc).
        if payload_size == HEAP_PAYLOAD_SIZE:
            data[0].chunk_id = timestamp // timestamp_step
            data[0].heap_index = channel_offset // n_chans_per_substream
            data[0].heap_offset = data[0].heap_index * HEAP_PAYLOAD_SIZE

    stream_config = spead2.recv.StreamConfig(substreams=HEAPS_PER_CHUNK, explicit_start=True)

    # Assuming X-engines are at most 500ms out of sync with each other, with
    # one extra chunk for luck. May need to revisit that assumption for much
    # larger array sizes.
    max_chunks = max(round(0.5 / int_time), 1) + 1

    return _create_receive_stream_group(
        interface_address,
        [multicast_endpoints],
        use_ibv,
        stream_config,
        max_chunks,
        0,
        chunk_place.ctypes,
        lambda stream: katgpucbf.recv.Chunk(
            present=np.empty(HEAPS_PER_CHUNK, np.uint8),
            data=np.empty((n_chans, n_bls, COMPLEX), dtype=np.dtype(f"int{n_bits_per_sample}")),
            sink=stream,
        ),
    )


def create_tied_array_channelised_voltage_receive_stream(
    interface_address: str,
    multicast_endpoints: list[list[tuple[str, int]]],
    n_chans: int,
    n_chans_per_substream: int,
    n_bits_per_sample: int,
    n_spectra_per_heap: int,
    n_samples_between_spectra: int,
    use_ibv: bool = False,
) -> spead2.recv.ChunkStreamRingGroup:
    """Create a spead2 recv stream for ingesting tied array channelised voltage data."""
    n_substreams = n_chans // n_chans_per_substream
    n_beams = len(multicast_endpoints)
    expected_payload_size = n_chans_per_substream * n_spectra_per_heap * COMPLEX * n_bits_per_sample // 8
    timestamp_step = n_spectra_per_heap * n_samples_between_spectra
    beam_ants_dtype = np.dtype(np.uint16)
    beam_ants_itemsize = beam_ants_dtype.itemsize

    @numba.cfunc(types.void(types.CPointer(chunk_place_data), types.size_t, types.CPointer(types.int64)), nopython=True)
    def chunk_place(data_ptr, data_size, user_data_ptr):
        data = numba.carray(data_ptr, 1)
        user_data = numba.carray(user_data_ptr, 1)  # Contains the beam index
        items = numba.carray(intp_to_voidptr(data[0].items), 4, dtype=np.int64)
        extra = numba.carray(intp_to_voidptr(data[0].extra), 1, dtype=beam_ants_dtype)
        channel_offset = items[0]
        timestamp = items[1]
        payload_size = items[3]
        beam = user_data[0]
        # If the payload size doesn't match, discard the heap (could be descriptors etc).
        if payload_size == expected_payload_size and timestamp >= 0:
            data[0].chunk_id = timestamp // timestamp_step
            data[0].heap_index = channel_offset // n_chans_per_substream + beam * n_substreams
            data[0].heap_offset = data[0].heap_index * payload_size
            extra[0] = items[2]  # beam_ants
            data[0].extra_size = beam_ants_itemsize
            data[0].extra_offset = data[0].heap_index * beam_ants_itemsize

    stream_config = spead2.recv.StreamConfig(substreams=n_substreams, explicit_start=True)

    # Allow about 1 GiB for resynchronising the B-engines.
    chunk_size = expected_payload_size * n_substreams * n_beams
    max_chunks = math.ceil(1024**3 / chunk_size)

    return _create_receive_stream_group(
        interface_address,
        multicast_endpoints,
        use_ibv,
        stream_config,
        max_chunks,
        beam_ants_dtype.itemsize,
        chunk_place.ctypes,
        lambda stream: katgpucbf.recv.Chunk(
            present=np.empty((n_beams, n_substreams), np.uint8),
            extra=np.zeros((n_beams, n_substreams), beam_ants_dtype),
            data=np.empty((n_beams, n_chans, n_spectra_per_heap, COMPLEX), dtype=np.dtype(f"int{n_bits_per_sample}")),
            sink=stream,
        ),
    )
