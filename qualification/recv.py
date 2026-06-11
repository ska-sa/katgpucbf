################################################################################
# Copyright (c) 2022-2026, National Research Foundation (SARAO)
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

"""Classes for receiving SPEAD data as part of the qualification tests."""

import ast
import asyncio
import ctypes
import io
import logging
import math
import os
import socket
import struct
from collections.abc import AsyncGenerator, Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal, overload

import numba
import numpy as np
import scipy
import spead2
import spead2.recv
import spead2.recv.asyncio
from baseband.vdif import VDIFFrame, VDIFFrameSet
from katsdptelstate.endpoint import endpoint_list_parser, endpoint_parser
from numba import types
from numpy.typing import NDArray
from spead2.numba import intp_to_voidptr
from spead2.recv.numba import chunk_place_data

import katgpucbf.recv
from katgpucbf import COMPLEX, DEFAULT_RECV_BUFFER_SIZE, DIG_SAMPLE_BITS
from katgpucbf.spead import BEAM_ANTS_ID, DEFAULT_PORT, FREQUENCY_ID, TIMESTAMP_ID
from katgpucbf.utils import TimeConverter

from .cbf import CBFRemoteControl

DEFAULT_MAX_DELAY = 1000000

DEFAULT_TIMEOUT = 10.0
logger = logging.getLogger(__name__)


class XBReceiver:
    """Base for :class:`BaselineCorrelationProductsReceiver` and :class:`TiedArrayChannelisedVoltageReceiver`."""

    # Attributes instantiated by the derived classes
    stream_group: spead2.recv.ChunkStreamRingGroup
    timestamp_step: int  # Step (in ADC samples) between chunk timestamps

    def __init__(self, cbf: CBFRemoteControl, stream_names: Sequence[str]) -> None:
        # Some metadata we know already from the config.
        acv_name: str = cbf.config["outputs"][stream_names[0]]["src_streams"][0]
        acv_config: dict[str, Any] = cbf.config["outputs"][acv_name]
        self.stream_names = list(stream_names)
        self.n_inputs = len(acv_config["src_streams"])
        self.n_ants = self.n_inputs // 2
        self.n_chans: int = acv_config["n_chans"]
        self.input_labels: list[str] = acv_config["input_labels"]
        self.decimation_factor: int = 1
        if "narrowband" in acv_config:
            self.decimation_factor = acv_config["narrowband"]["decimation_factor"]
        self.adc_sample_rate: float = cbf.config["outputs"][acv_config["src_streams"][0]]["adc_sample_rate"]

        # But some we don't. Note: these could be properties. But copying them up
        # front ensures we get an exception early if the sensor is missing.
        # We assume the streams all have the same information except for addresses.
        self.n_chans_per_substream: int = cbf.init_sensors[f"{stream_names[0]}.n-chans-per-substream"].value
        self.n_spectra_per_heap: int = cbf.init_sensors[f"{acv_name}.spectra-per-heap"].value
        self.n_samples_between_spectra: int = cbf.init_sensors[f"{acv_name}.n-samples-between-spectra"].value
        self.sync_time: float = cbf.init_sensors[f"{acv_name}.sync-time"].value
        self.scale_factor_timestamp: float = cbf.init_sensors[f"{acv_name}.scale-factor-timestamp"].value
        self.bandwidth: float = cbf.init_sensors[f"{acv_name}.bandwidth"].value
        self.center_freq: float = cbf.init_sensors[f"{acv_name}.center-freq"].value
        self.multicast_endpoints: list[list[tuple[str, int]]] = [
            [
                (endpoint.host, endpoint.port)
                for endpoint in endpoint_list_parser(DEFAULT_PORT)(
                    cbf.init_sensors[f"{stream_name}.destination"].value.decode()
                )
            ]
            for stream_name in stream_names
        ]

        self.time_converter = TimeConverter(self.sync_time, self.scale_factor_timestamp)
        self.cbf = cbf
        self._acv_name = acv_name

    def is_complete_chunk(self, chunk: katgpucbf.recv.Chunk) -> bool:
        """Check whether this chunk is complete (no missing data)."""
        return bool(np.all(chunk.present))

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
    ) -> AsyncGenerator[tuple[int, katgpucbf.recv.Chunk], None]:
        yield ...  # type: ignore

    @overload
    async def complete_chunks(
        self,
        min_timestamp: int | None = None,
        *,
        all_timestamps: bool = False,
        max_delay: int = DEFAULT_MAX_DELAY,
        time_limit: float | None = None,
    ) -> AsyncGenerator[tuple[int, katgpucbf.recv.Chunk | None], None]:
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

        data_ringbuffer = self.stream_group.data_ringbuffer
        assert isinstance(data_ringbuffer, spead2.recv.asyncio.ChunkRingbuffer)
        try:
            async with asyncio.timeout(time_limit) as timer:
                async for chunk in data_ringbuffer:
                    assert isinstance(chunk, katgpucbf.recv.Chunk)  # keeps mypy happy
                    timestamp = chunk.chunk_id * self.timestamp_step
                    if min_timestamp is not None and timestamp < min_timestamp:
                        logger.debug("Skipping chunk with timestamp %d (< %d)", timestamp, min_timestamp)
                    elif not self.is_complete_chunk(chunk):
                        logger.debug("Incomplete chunk %d", chunk.chunk_id)
                    else:
                        yield timestamp, chunk
                        continue
                    # If we get here, the chunk is ignored
                    chunk.recycle()
                    if all_timestamps:
                        yield timestamp, None
        except TimeoutError:
            if not timer.expired():
                raise  # The TimeoutError came from something else

    async def next_complete_chunk(
        self,
        min_timestamp: int | None = None,
        *,
        max_delay: int = DEFAULT_MAX_DELAY,
        timeout: float | None = DEFAULT_TIMEOUT,
    ) -> tuple[int, np.ndarray]:
        """Return the data from the next complete chunk from the stream.

        The return value includes the timestamp.

        Parameters
        ----------
        min_timestamp, max_delay
            See :meth:`complete_chunks`
        timeout
            Maximum time to wait

        Raises
        ------
        TimeoutError
            If a complete chunk is not received in time
        RuntimeError
            If the stream is stopped before a complete chunk is received
        """
        async with asyncio.timeout(timeout):
            async for timestamp, chunk in self.complete_chunks(min_timestamp=min_timestamp, max_delay=max_delay):
                with chunk:
                    return timestamp, np.array(chunk.data)  # Makes a copy before we return the chunk
        raise RuntimeError("stream was shut down before we received a complete chunk")

    async def wait_complete_chunk(
        self,
        min_timestamp: int | None = None,
        *,
        max_delay: int = DEFAULT_MAX_DELAY,
        timeout: float | None = DEFAULT_TIMEOUT,
    ) -> int:
        """Wait for a complete chunk, but do not return it.

        Only the timestamp is returned. This is more efficient than
        :meth:`next_complete_chunk` because it does not need to copy the data
        from the chunk.

        Parameters
        ----------
        min_timestamp, max_delay
            See :meth:`complete_chunks`
        timeout
            Maximum time to wait

        Raises
        ------
        TimeoutError
            If a complete chunk is not received in time
        RuntimeError
            If the stream is stopped before a complete chunk is received
        """
        async with asyncio.timeout(timeout):
            async for timestamp, chunk in self.complete_chunks(min_timestamp=min_timestamp, max_delay=max_delay):
                chunk.recycle()
                return timestamp
        raise RuntimeError("stream was shut down before we received a complete chunk")

    async def consecutive_chunks(
        self,
        n: int,
        min_timestamp: int | None = None,
        *,
        max_delay: int = DEFAULT_MAX_DELAY,
        timeout: float | None = DEFAULT_TIMEOUT,
    ) -> list[tuple[int, katgpucbf.recv.Chunk]]:
        """Obtain `n` consecutive complete chunks from the stream.

        .. warning::

           This is not safe to use with large values of `n`, because the chunks
           are removed from the stream's pool. If you plan to use any value
           larger than 2 you should check that the free ring is initialised
           with enough chunks and update this comment.
        """
        chunks: list[tuple[int, katgpucbf.recv.Chunk]] = []
        async with asyncio.timeout(timeout):
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
    def channel_frequency(self, channel: float) -> float: ...

    @overload
    def channel_frequency(self, channel: np.ndarray) -> np.ndarray: ...

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

    def __init__(
        self,
        cbf: CBFRemoteControl,
        stream_name: str,
        cores: Sequence[int],
        interface_address: str,
        use_ibv: bool = False,
    ) -> None:
        super().__init__(cbf, [stream_name])

        # Fill in extra sensors specific to baseline-correlation-products
        self.n_bls: int = cbf.init_sensors[f"{stream_name}.n-bls"].value
        self.n_bits_per_sample: int = cbf.init_sensors[f"{stream_name}.xeng-out-bits-per-sample"].value
        self.n_spectra_per_acc: int = cbf.init_sensors[f"{stream_name}.n-accs"].value
        self.int_time: float = cbf.init_sensors[f"{stream_name}.int-time"].value
        self.bls_ordering: list[tuple[str, str]] = ast.literal_eval(
            cbf.init_sensors[f"{stream_name}.bls-ordering"].value.decode()
        )
        self.n_xengs: int = cbf.init_sensors[f"{stream_name}.n-xengs"].value

        self.timestamp_step = self.n_samples_between_spectra * self.n_spectra_per_acc
        self.stream_group = create_baseline_correlation_product_receive_stream_group(
            interface_address,
            multicast_endpoints=self.multicast_endpoints[0],
            cores=cores,
            n_bls=self.n_bls,
            n_chans=self.n_chans,
            n_chans_per_substream=self.n_chans_per_substream,
            n_bits_per_sample=self.n_bits_per_sample,
            n_spectra_per_acc=self.n_spectra_per_acc,
            int_time=self.int_time,
            n_samples_between_spectra=self.n_samples_between_spectra,
            use_ibv=use_ibv,
        )

    def is_complete_chunk(self, chunk: katgpucbf.recv.Chunk) -> bool:  # noqa: D102
        if not super().is_complete_chunk(chunk):
            return False
        # Elements affected by missing antennas are marked with a real part
        # equal to -2^31. It's expensive to check every element for this
        # marker, but we know it will apply to all channels in a heap, so we
        # only need to check the first channel.
        n_channels_per_substream = chunk.data.shape[0] // chunk.present.shape[0]
        return np.min(chunk.data[::n_channels_per_substream]) > -(2**31)

    def compute_tone_gain(self, amplitude: float, target_voltage: float) -> float:  # noqa: D102
        # We need to avoid saturating the signed 32-bit X-engine accumulation as
        # well (2e9 is comfortably less than 2^31).
        target_voltage = min(target_voltage, np.sqrt(2e9 / self.n_spectra_per_acc))
        return super().compute_tone_gain(amplitude, target_voltage)

    if TYPE_CHECKING:
        # Just refine the return type, without any run-time implementation
        async def next_complete_chunk(  # noqa: D102
            self,
            min_timestamp: int | None = None,
            *,
            max_delay: int = DEFAULT_MAX_DELAY,
            timeout: float | None = DEFAULT_TIMEOUT,
        ) -> tuple[int, NDArray[np.int32]]: ...


class TiedArrayChannelisedVoltageReceiver(XBReceiver):
    """Wrap a tied-array-channelised-voltage stream with helper functions."""

    def __init__(
        self,
        cbf: CBFRemoteControl,
        stream_names: Sequence[str],
        cores: Sequence[int],
        interface_address: str,
        use_ibv: bool = False,
    ) -> None:
        super().__init__(cbf, stream_names)

        self.n_bits_per_sample = cbf.init_sensors[f"{stream_names[0]}.beng-out-bits-per-sample"].value
        self.timestamp_step = self.n_samples_between_spectra * self.n_spectra_per_heap
        self.source_indices: list[list[int]] = [
            ast.literal_eval(cbf.init_sensors[f"{stream_name}.source-indices"].value.decode())
            for stream_name in stream_names
        ]
        self.n_bengs = cbf.init_sensors[f"{stream_names[0]}.n-bengs"].value

        self.stream_group = create_tied_array_channelised_voltage_receive_stream_group(
            interface_address,
            multicast_endpoints=self.multicast_endpoints,
            cores=cores,
            n_chans=self.n_chans,
            n_chans_per_substream=self.n_chans_per_substream,
            n_bits_per_sample=self.n_bits_per_sample,
            n_spectra_per_heap=self.n_spectra_per_heap,
            n_samples_between_spectra=self.n_samples_between_spectra,
            decimation_factor=self.decimation_factor,
            use_ibv=use_ibv,
        )

    def is_complete_chunk(self, chunk: katgpucbf.recv.Chunk) -> bool:  # noqa: D102
        return super().is_complete_chunk(chunk) and (chunk.extra is None or np.min(chunk.extra) == self.n_ants)


def _create_receive_stream_group(
    interface_address: str,
    multicast_endpoints: Sequence[list[tuple[str, int]]],
    cores: Sequence[int],
    use_ibv: bool,
    stream_config: spead2.recv.StreamConfig,
    max_chunks: int,
    max_heap_extra: int,
    chunk_place: Callable,  # Actual type comes from ctypes.CFUNCTYPE, but it doesn't have a static name
    chunk_factory: Callable[[spead2.recv.ChunkRingPair], katgpucbf.recv.Chunk],
) -> spead2.recv.ChunkStreamRingGroup:
    """Create and start a stream group to receive data from an engine.

    Parameters
    ----------
    interface_address
        IP address of receiving interface
    multicast_endpoints
        List of list of (group, port) pairs. Each list corresponds to a single
        stream.
    cores
        CPU cores to bind to, of the same length as `multicast_endpoints`.
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
    for i, (endpoints, core) in enumerate(zip(multicast_endpoints, cores, strict=True)):
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
        # Opt in to real-time scheduling just while creating the thread pool,
        # so that the thread inherits it.
        scheduling_set = False
        try:
            old_policy = os.sched_getscheduler(0)
            old_param = os.sched_getparam(0)
            os.sched_setscheduler(0, os.SCHED_RR, os.sched_param(1))
            scheduling_set = True
        except PermissionError:
            logger.warning("Real-time scheduling could not be enabled (permission denied)")
        stream = group.emplace_back(spead2.ThreadPool(1, [core]), stream_config, chunk_stream_config)
        if scheduling_set:
            os.sched_setscheduler(0, old_policy, old_param)

        if use_ibv:
            config = spead2.recv.UdpIbvConfig(
                endpoints=endpoints,
                interface_address=interface_address,
                buffer_size=DEFAULT_RECV_BUFFER_SIZE,
                comp_vector=core,
            )
            stream.add_udp_ibv_reader(config)
        else:
            for ep in endpoints:
                stream.add_udp_reader(*ep, interface_address=interface_address)
    for stream in group:
        stream.start()
    return group


def create_baseline_correlation_product_receive_stream_group(
    interface_address: str,
    multicast_endpoints: list[tuple[str, int]],
    cores: Sequence[int],
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

    # Use one stream per core, and partition the endpoints between them.
    multicast_sets: list[list[tuple[str, int]]] = [[] for _ in cores]
    for i, endpoint in enumerate(multicast_endpoints):
        multicast_sets[i % len(cores)].append(endpoint)

    return _create_receive_stream_group(
        interface_address,
        multicast_sets,
        cores,
        use_ibv,
        stream_config,
        max_chunks,
        0,
        chunk_place.ctypes,
        # np.ones is used to ensure that the memory is paged in
        lambda stream_group: katgpucbf.recv.Chunk(
            present=np.ones(HEAPS_PER_CHUNK, np.uint8),
            data=np.ones((n_chans, n_bls, COMPLEX), dtype=np.dtype(f"int{n_bits_per_sample}")),
            sink=stream_group,
        ),
    )


def create_tied_array_channelised_voltage_receive_stream_group(
    interface_address: str,
    multicast_endpoints: Sequence[list[tuple[str, int]]],
    cores: Sequence[int],
    n_chans: int,
    n_chans_per_substream: int,
    n_bits_per_sample: int,
    n_spectra_per_heap: int,
    n_samples_between_spectra: int,
    decimation_factor: int,
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

    # Allow about 1 GiB for resynchronising the B-engines in wideband.
    # In narrowband, reduce it by the decimation factor to avoid creating
    # an excessively high latency which slows down the test.
    chunk_size = expected_payload_size * n_substreams * n_beams
    max_chunks = math.ceil(1024**3 / chunk_size / decimation_factor)

    return _create_receive_stream_group(
        interface_address,
        multicast_endpoints,
        cores,
        use_ibv,
        stream_config,
        max_chunks,
        beam_ants_dtype.itemsize,
        chunk_place.ctypes,
        # np.ones is used to ensure that the memory is paged in
        lambda stream_group: katgpucbf.recv.Chunk(
            present=np.ones((n_beams, n_substreams), np.uint8),
            extra=np.ones((n_beams, n_substreams), beam_ants_dtype),
            data=np.ones((n_beams, n_chans, n_spectra_per_heap, COMPLEX), dtype=np.dtype(f"int{n_bits_per_sample}")),
            sink=stream_group,
        ),
    )


class VTPBuffer:
    """Buffer for storing VTP packets, and decoding them into VDIF framesets."""

    def __init__(self, n_threads: int) -> None:
        self.data = list[bytes]()
        self.incomplete_framesets = list[VDIFFrameSet]()
        self.n_threads = n_threads

    def add_packet(self, packet: bytes) -> None:
        """Add a packet to the buffer without decoding it."""
        self.data.append(packet)

    async def decode_vtp(self) -> AsyncGenerator[tuple[int, bytes], None]:
        """Decode the VTP packets in the buffer."""
        for packet in self.data:
            new_seq_id = struct.unpack("<Q", packet[:8])[0]
            yield new_seq_id, packet[8:]

    async def decode_vdif_framesets(self) -> AsyncGenerator[tuple[list[int], VDIFFrameSet], None]:
        """Slow computation: order and decode all the data captured so far.

        The seq_ids might include frames whose complete set is not present.
        These frames will be ignored.
        """
        vtp_packets = dict[int, bytes]()
        async for seq_id, packet in self.decode_vtp():
            vtp_packets[seq_id] = packet

        seq_ids = sorted(vtp_packets.keys())
        previous_frames = list[VDIFFrame]()
        set_seq_ids = list[int]()
        for seq_id in seq_ids:
            frame = VDIFFrame.fromfile(io.BytesIO(vtp_packets[seq_id]))
            if frame.header.frame_nr == 0:
                if len(previous_frames) > 0:
                    frameset = VDIFFrameSet(previous_frames, previous_frames[0].header)
                    if frameset.get_thread_ids() != self.n_threads:
                        self.incomplete_framesets.append(frameset)
                    else:
                        yield set_seq_ids, frameset
                previous_frames = [frame]
                set_seq_ids.clear()
            previous_frames.append(frame)
            set_seq_ids.append(seq_id)

        if len(previous_frames) > 0:
            frameset = VDIFFrameSet(previous_frames, previous_frames[0].header)
            if frameset.get_thread_ids() != self.n_threads:
                self.incomplete_framesets.append(frameset)
            else:
                yield set_seq_ids, frameset

    def close(self) -> None:
        """Close the buffer."""
        self.data.clear()


class TiedArrayResampledVoltageReceiver:
    """Receive tied-array-resampled-voltage streams from the v engines."""

    max_packet_size = 65535

    def __init__(
        self,
        cbf: CBFRemoteControl,
        interface_address: str,
    ) -> None:
        self.stream_names = list(["tied-array-resampled-voltage"])
        self.multicast_group = endpoint_parser(DEFAULT_PORT)(
            cbf.init_sensors[f"{self.stream_names[0]}.destination"].value.decode()
        )
        self.scale_factor_timestamp = cbf.init_sensors[f"{self.stream_names[0]}.scale-factor-timestamp"].value
        self.power_int_time = cbf.init_sensors[f"{self.stream_names[0]}.power-int-time"].value
        self.bandwidth = cbf.init_sensors[f"{self.stream_names[0]}.bandwidth"].value
        tacv_name: str = cbf.config["outputs"][self.stream_names[0]]["src_streams"][0]
        acv_name: str = cbf.config["outputs"][tacv_name]["src_streams"][0]
        acv_config: dict[str, Any] = cbf.config["outputs"][acv_name]
        self.n_inputs = len(acv_config["src_streams"])
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.multicast_group.host, self.multicast_group.port))
        mreq = struct.pack("=4s4s", socket.inet_aton(self.multicast_group.host), socket.inet_aton(interface_address))
        self.socket.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        self.socket.setblocking(False)
        self.vtp_buffer = VTPBuffer(4)  # TODO: calculate number of threads

    async def _read(self) -> bytes:
        loop = asyncio.get_running_loop()
        return await loop.sock_recv(self.socket, self.max_packet_size)

    async def listen(self) -> AsyncGenerator[tuple[int, bytes], None]:
        """Listen for packets from the v engine and store them in the VTPBuffer."""
        while True:
            self.vtp_buffer.add_packet(await self._read())

    async def decode_vdif_framesets(self) -> AsyncGenerator[tuple[list[int], VDIFFrameSet], None]:
        """Decode the VDIF framesets in the buffer."""
        async for seq_ids, frameset in self.vtp_buffer.decode_vdif_framesets():
            yield seq_ids, frameset

    async def get_vdif_frameset(self) -> tuple[list[int], VDIFFrameSet]:
        """Listen until a single complete VDIF frameset is available, then return it."""
        while True:
            self.vtp_buffer.add_packet(await self._read())
            try:
                seq_ids, frameset = await anext(self.vtp_buffer.decode_vdif_framesets())
            except StopAsyncIteration:
                continue
            return seq_ids, frameset

    def get_incomplete_framesets(self) -> list[VDIFFrameSet]:
        """Get the incomplete VDIF framesets in the buffer."""
        return self.vtp_buffer.incomplete_framesets

    def close(self) -> None:
        """Close the socket."""
        self.socket.close()
        self.vtp_buffer.close()
