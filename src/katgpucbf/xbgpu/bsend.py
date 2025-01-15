################################################################################
# Copyright (c) 2023-2024, National Research Foundation (SARAO)
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

"""Module for sending tied array channelised voltage products onto the network."""

import asyncio
import functools
import logging
from collections.abc import Callable, Sequence
from typing import Final

import katsdpsigproc.accel as accel
import numpy as np
import spead2
import spead2.send.asyncio
from aiokatcp import SensorSet
from katsdpsigproc.abc import AbstractContext
from katsdptelstate.endpoint import Endpoint
from prometheus_client import Counter

from .. import COMPLEX, DEFAULT_PACKET_PAYLOAD_BYTES
from ..send import send_rate
from ..spead import BEAM_ANTS_ID, BF_RAW_ID, FLAVOUR, FREQUENCY_ID, IMMEDIATE_DTYPE, IMMEDIATE_FORMAT, TIMESTAMP_ID
from ..utils import TimeConverter
from . import METRIC_NAMESPACE
from .output import BOutput
from .send import Send

output_heaps_counter = Counter(
    "output_b_heaps", "number of B-engine heaps transmitted", ["stream"], namespace=METRIC_NAMESPACE
)
output_bytes_counter = Counter(
    "output_b_bytes", "number of B-engine payload bytes transmitted", ["stream"], namespace=METRIC_NAMESPACE
)
output_samples_counter = Counter(
    "output_b_samples", "number of complex beam samples transmitted", ["stream"], namespace=METRIC_NAMESPACE
)
output_clip_counter = Counter(
    "output_b_clipped_samples", "number of beam samples that were saturated", ["stream"], namespace=METRIC_NAMESPACE
)

logger = logging.getLogger(__name__)
# NOTE: ICD suggests `beng_out_bits_per_sample`,
# MK correlator doesn't make this configurable.
SEND_DTYPE = np.dtype(np.int8)


def make_item_group(bf_raw_shape: tuple[int, ...]) -> spead2.send.ItemGroup:
    """Create an item group (with no values)."""
    item_group = spead2.send.ItemGroup(flavour=FLAVOUR)
    item_group.add_item(
        TIMESTAMP_ID,
        "timestamp",
        "Timestamp provided by the MeerKAT digitisers and scaled to the digitiser sampling rate.",
        shape=[],
        format=IMMEDIATE_FORMAT,
    )
    item_group.add_item(
        FREQUENCY_ID,
        "frequency",  # Misleading name, but it's what the ICD specifies
        "Value of the first channel in collections stored here.",
        shape=[],
        format=IMMEDIATE_FORMAT,
    )
    item_group.add_item(
        BEAM_ANTS_ID,
        "beam_ants",
        "Count of antennas included in the beam sum.",
        shape=[],
        format=IMMEDIATE_FORMAT,
    )
    item_group.add_item(
        BF_RAW_ID,
        "bf_raw",
        "Beamformer output for frequency-domain beam.",
        shape=bf_raw_shape,
        dtype=SEND_DTYPE,
    )
    return item_group


class Batch:
    """Hold all data for heaps with a single timestamp.

    It does not own its memory - the backing store is in :class:`Chunk`. It keeps
    a cached :class:`spead2.send.HeapReferenceList` with the heaps of the enabled
    beams, along with a tuple of the enabled beams.

    Parameters
    ----------
    timestamp
        Zero-dimensional array of dtype ``>u8`` holding the timestamp
    data
        Payload data for the batch with shape (n_beams,
        n_channels_per_substream, spectra_per_heap, COMPLEX).
    channel_offset
        The first frequency channel processed.
    present_ants
        Zero-dimensional array of dtype ``>u8`` holding the number of antennas
        present in the Batch's input data.
    """

    def __init__(
        self,
        timestamp: np.ndarray,
        data: np.ndarray,
        *,
        channel_offset: int,
        present_ants: np.ndarray,
    ) -> None:
        self.heaps: list[spead2.send.Heap] = []
        self.data = data
        n_beams = data.shape[0]

        item_group = make_item_group(data.shape[1:])  # Get rid of the 'beam' dimension
        item_group[FREQUENCY_ID].value = channel_offset
        item_group[TIMESTAMP_ID].value = timestamp
        item_group[BEAM_ANTS_ID].value = present_ants
        for i in range(n_beams):
            item_group[BF_RAW_ID].value = self.data[i, ...]
            heap = item_group.get_heap(descriptors="none", data="all")
            heap.repeat_pointers = True
            self.heaps.append(heap)
        self.send_enabled = (False,) * n_beams
        self.send_heaps = spead2.send.HeapReferenceList([])


class Chunk:
    r"""
    An array of :class:`Batch`\ es.

    Parameters
    ----------
    data
        Storage for tied-array-channelised-voltage data, with shape (n_batches,
        n_beams, n_channels_per_substream, n_spectra_per_heap, COMPLEX) and
        dtype :const:`SEND_DTYPE`.
    saturated
        Storage for saturation counts, with shape (n_beams,) and dtype
        uint32.
    channel_offset
        The first frequency channel processed.
    timestamp_step
        Timestamp step between successive :class:`Batch`\ es in a chunk.
    """

    def __init__(
        self,
        data: np.ndarray,
        saturated: np.ndarray,
        *,
        channel_offset: int,
        timestamp_step: int,
    ) -> None:
        n_batches = data.shape[0]
        self.data = data
        self.saturated = saturated

        self._timestamp = 0
        self._timestamp_step = timestamp_step
        self._timestamps = (np.arange(n_batches) * self._timestamp_step).astype(IMMEDIATE_DTYPE)

        self._present_ants = np.zeros(shape=(n_batches,), dtype=IMMEDIATE_DTYPE)
        # NOTE: The future indicates when it is safe to modify the chunk,
        # i.e. it is not being transmitted. At construction there is nothing to
        # wait for, so we mark it ready.
        self.future = asyncio.get_running_loop().create_future()
        self.future.set_result(None)

        self._batches = [
            Batch(
                self._timestamps[i, ...],
                data[i],
                channel_offset=channel_offset,
                present_ants=self._present_ants[i, ...],
            )
            for i in range(n_batches)
        ]

    @property
    def present_ants(self) -> np.ndarray:
        """
        Number of antennas present in the current beam sums.

        This is a count for each :class:`Batch` in the chunk. Setting this
        property updates the immediate SPEAD items in the heaps. Much like
        :attr:`timestamp`, this should only be done when :attr:`future`
        is done.
        """
        return self._present_ants

    @present_ants.setter
    def present_ants(self, value: np.ndarray) -> None:
        self._present_ants[:] = value

    @property
    def timestamp(self) -> int:
        """
        Timestamp of the first heap.

        Setting this property updates the timestamps stored in all the heaps.
        This should only be done when :attr:`future` is done.
        """
        return self._timestamp

    @timestamp.setter
    def timestamp(self, value: int) -> None:
        delta = value - self.timestamp
        self._timestamps += delta
        self._timestamp = value

    @staticmethod
    def _inc_counters(
        n_batches_sent: int,
        data_shape: tuple[int, int, int],
        data_dtype: np.dtype,
        enabled: Sequence[bool],
        output_names: Sequence[str],
        saturated: np.ndarray,
        sensors: SensorSet,
        sensor_timestamp: float,
        future: asyncio.Future,
    ) -> None:
        """Increment beam stream Prometheus counters.

        Intended to be used on a gathered set of futures as it is
        computationally expensive to increment Prometheus counters for each
        call to async_send_heaps.

        Parameters
        ----------
        n_batches_sent
            The number of batches transmitted.
        data_shape
            The shape of the beam data being transmitted. Expected in the
            format of (n_channels_per_substream, samples_per_spectra, COMPLEX).
        data_dtype
            The `np.dtype` of the beam data transmitted.
        enabled
            Boolean flag array indicating which streams are enabled for transmission.
        output_names
            List of beam stream names
        saturated
            Saturation count for the chunk for each stream in `output_names`.
        sensors
            Server's katcp sensors.
        sensor_timestamp
            Timestamp (UNIX time) to use for sensor update.
        future
            Future returned by the spead2 stream's `async_send_heaps`.
        """
        if future.cancelled() or future.exception() is not None:
            # Don't update output counters if we didn't successfully transmit the data.
            n_batches_sent = 0
        # int casts are because np.prod returns np.int64 which is
        # incompatible with the type annotations for Prometheus.
        # Multiply across dimensions to get total bytes
        byte_count = int(np.prod(data_shape)) * data_dtype.itemsize * n_batches_sent
        # Multiply across the first two dimensions to get complex sample count
        sample_count = int(np.prod(data_shape[:-1])) * n_batches_sent
        for i, output_name in enumerate(output_names):
            clipped = int(saturated[i])
            sensor = sensors[f"{output_name}.beng-clip-cnt"]
            sensor.set_value(sensor.value + clipped, timestamp=sensor_timestamp)
            if enabled[i] and n_batches_sent != 0:
                output_heaps_counter.labels(output_name).inc(n_batches_sent)
                output_bytes_counter.labels(output_name).inc(byte_count)
                output_samples_counter.labels(output_name).inc(sample_count)
                output_clip_counter.labels(output_name).inc(clipped)

    def send(
        self,
        send_stream: "BSend",
        time_converter: TimeConverter,
        sensors: SensorSet,
    ) -> asyncio.Future:
        """
        Transmit a chunk's heaps over a SPEAD stream.

        This method returns immediately and sends the data asynchronously. Before
        modifying the chunk, first await :attr:`future`.
        """
        send_enabled = tuple(
            enabled and self.timestamp >= timestamp
            for enabled, timestamp in zip(send_stream.send_enabled, send_stream.send_enabled_timestamp)
        )
        n_enabled = sum(send_enabled)
        rate = send_stream.bytes_per_second_per_beam * n_enabled
        send_futures: list[asyncio.Future] = []
        if n_enabled > 0:
            for batch, antenna_presence in zip(self._batches, self._present_ants):
                if antenna_presence == 0:
                    # No antennas were present in the received batch of heaps
                    # This check takes priority as we do not transmit batches
                    # that did not have any input data. The updating of the
                    # batch's :class:`HeapReferenceList` is not time-critical.
                    continue
                if batch.send_enabled != send_enabled:
                    batch.send_heaps = spead2.send.HeapReferenceList(
                        [
                            spead2.send.HeapReference(heap, substream_index=i, rate=rate)
                            for i, (heap, enabled) in enumerate(zip(batch.heaps, send_enabled))
                            if enabled
                        ]
                    )
                    batch.send_enabled = send_enabled
                send_futures.append(
                    send_stream.stream.async_send_heaps(batch.send_heaps, mode=spead2.send.GroupMode.ROUND_ROBIN)
                )

            self.future = asyncio.gather(*send_futures)
        else:
            # TODO: Is it necessary to handle this case?
            self.future = asyncio.create_task(send_stream.stream.async_flush())

        end_timestamp_adc = self._timestamp + self._timestamp_step * len(self._batches)
        end_timestamp_unix = time_converter.adc_to_unix(end_timestamp_adc)
        self.future.add_done_callback(
            functools.partial(
                self._inc_counters,
                len(send_futures),  # Increment counters for as many calls to async_send_heaps
                # Get rid of 'batch' and 'beam' dimensions (ignore is because
                # mypy doesn't know the number of dimensions)
                self.data.shape[2:],  # type: ignore[arg-type]
                self.data.dtype,
                send_enabled,
                send_stream.output_names,
                self.saturated.copy(),  # Copy since the original may get overwritten
                sensors,
                end_timestamp_unix,
            )
        )
        return self.future


class BSend(Send):
    r"""
    Class for turning tied array channelised voltage products into SPEAD heaps.

    This class creates a queue of chunks that can be sent out onto the network.
    To obtain a chunk, call :meth:`get_free_chunk` - which will return a
    :class:`Chunk`. This object will create a limited number of transmit
    buffers and keep recycling them, avoiding any memory allocation at runtime.

    The transmission of a chunk's data is abstracted by :meth:`send_chunk`. This
    invokes transmission and immediately returns the :class:`Chunk` back to the
    queue for reuse.

    This object keeps track of each tied-array-channelised-voltage data stream by
    means of substreams in :class:`spead2.send.asyncio.AsyncStream`, allowing
    for individual enabling and disabling of the data product.

    To allow this class to be used with multiple transports, the constructor
    takes a factory function to create the stream.

    Parameters
    ----------
    outputs
        Sequence of :class:`.output.BOutput`.
    batches_per_chunk
        Number of :class:`Batch`\ es in each transmitted :class:`Chunk`.
    n_chunks
        Number of :class:`Chunk`\ s to create.
    adc_sample_rate, n_channels, n_channels_per_substream, spectra_per_heap, channel_offset
        See :class:`.XBEngine` for further information.
    timestamp_step
        The timestamp step between successive heaps.
    send_rate_factor
        Factor dictating how fast the send-stream should transmit data.
    context
        Device context to create buffers.
    stream_factory
        Callback function to create the spead2 send stream. It is passed the
        stream configuration and memory buffers.
    packet_payload
        Size, in bytes, for the output packets (tied array channelised voltage
        payload only; headers and padding are added to this).
    send_enabled
        Enable/Disable transmission.
    """

    descriptor_heap: spead2.send.Heap
    preamble_size: Final[int] = 72

    def __init__(
        self,
        outputs: Sequence[BOutput],
        batches_per_chunk: int,
        n_chunks: int,
        n_channels: int,
        n_channels_per_substream: int,
        spectra_per_heap: int,
        adc_sample_rate: float,
        timestamp_step: int,
        send_rate_factor: float,
        channel_offset: int,
        context: AbstractContext,
        stream_factory: Callable[[spead2.send.StreamConfig, Sequence[np.ndarray]], "spead2.send.asyncio.AsyncStream"],
        packet_payload: int = DEFAULT_PACKET_PAYLOAD_BYTES,
        send_enabled: bool = False,
    ) -> None:
        self.send_enabled = [send_enabled] * len(outputs)
        self.send_enabled_timestamp = [0] * len(outputs)
        n_beams = len(outputs)
        self.output_names = [output.name for output in outputs]

        self._chunks_queue: asyncio.Queue[Chunk] = asyncio.Queue()
        buffers: list[np.ndarray] = []

        send_shape = (batches_per_chunk, n_beams, n_channels_per_substream, spectra_per_heap, COMPLEX)
        for _ in range(n_chunks):
            chunk = Chunk(
                accel.HostArray(send_shape, SEND_DTYPE, context=context),
                accel.HostArray((n_beams,), np.uint32, context=context),
                channel_offset=channel_offset,
                timestamp_step=timestamp_step,
            )
            self._chunks_queue.put_nowait(chunk)
            buffers.append(chunk.data)

        heap_payload_size_bytes = n_channels_per_substream * spectra_per_heap * COMPLEX * SEND_DTYPE.itemsize
        self.bytes_per_second_per_beam = send_rate(
            packet_header=BSend.preamble_size,
            packet_payload=packet_payload,
            heap_payload=heap_payload_size_bytes,
            heap_interval=timestamp_step / adc_sample_rate,
            send_rate_factor=send_rate_factor,
        )

        stream_config = spead2.send.StreamConfig(
            max_packet_size=packet_payload + BSend.preamble_size,
            # + 1 below for the descriptor per beam
            max_heaps=(n_chunks * batches_per_chunk + 1) * n_beams,
            rate_method=spead2.send.RateMethod.AUTO,
        )

        item_group = make_item_group(buffers[0].shape[2:])
        super().__init__(
            n_channels=n_channels,
            n_channels_per_substream=n_channels_per_substream,
            channel_offset=channel_offset,
            stream=stream_factory(stream_config, buffers),
            descriptor_heap=item_group.get_heap(descriptors="all", data="none"),
        )

    def enable_beam(self, beam_id: int, enable: bool = True, timestamp: int = 0) -> None:
        """Enable/Disable a beam's data transmission.

        :class:`.BSend` operates as a single, large stream with multiple
        substreams. Each substream (beam) is its own data product and is
        required to be enabled/disabled independently.

        Parameters
        ----------
        beam_id
            Index of the beam's data product.
        enable
            Boolean indicating whether the `beam_id` should be enabled or
            disabled.
        timestamp
            Minimum timestamp to transmit when enabled.
        """
        self.send_enabled[beam_id] = enable
        self.send_enabled_timestamp[beam_id] = timestamp

    async def get_free_chunk(self) -> Chunk:
        """Obtain a :class:`.Chunk` for transmission.

        We await the chunk's :attr:`future` to be sure we are not overwriting
        data that is still being transmitted. If sending failed, it is no
        longer being transmitted, and therefore safe to return the chunk.

        Raises
        ------
        asyncio.CancelledError
            If the chunk's send future is cancelled.
        """
        chunk = await self._chunks_queue.get()
        try:
            await chunk.future
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Error sending chunk")
        return chunk

    def send_chunk(self, chunk: Chunk, time_converter: TimeConverter, sensors: SensorSet) -> None:
        """Send a chunk's data and put it on the :attr:`_chunks_queue`."""
        chunk.send(self, time_converter, sensors)
        self._chunks_queue.put_nowait(chunk)

    async def send_stop_heap(self) -> None:
        """Send a Stop Heap over the spead2 transport."""
        stop_heap = spead2.send.Heap(FLAVOUR)
        stop_heap.add_end()
        # Flush just to ensure that we don't overflow the stream's queue.
        # It's a heavy-handed approach, but we don't care about performance
        # during shutdown.
        await self.stream.async_flush()
        for i in range(len(self.output_names)):
            await self.stream.async_send_heap(stop_heap, substream_index=i)


def make_stream(
    *,
    output_names: list[str],
    endpoints: list[Endpoint],
    interface: str,
    ttl: int,
    use_ibv: bool,
    affinity: int,
    comp_vector: int,
    stream_config: spead2.send.StreamConfig,
    buffers: Sequence[np.ndarray],
) -> "spead2.send.asyncio.AsyncStream":
    """Create asynchronous SPEAD stream for transmission.

    This is architected to be a single send stream with multiple substreams,
    each corresponding to a tied-array-channelised-voltage output data product.
    The `endpoints` need not be a contiguous list of multicast addresses.
    """
    stream: spead2.send.asyncio.AsyncStream
    thread_pool = spead2.ThreadPool(1, [] if affinity < 0 else [affinity])

    if use_ibv:
        stream = spead2.send.asyncio.UdpIbvStream(
            thread_pool,
            stream_config,
            spead2.send.UdpIbvConfig(
                endpoints=[(ep.host, ep.port) for ep in endpoints],
                interface_address=interface,
                ttl=ttl,
                comp_vector=comp_vector,
                memory_regions=list(buffers),
            ),
        )
    else:
        stream = spead2.send.asyncio.UdpStream(
            thread_pool,
            [(ep.host, ep.port) for ep in endpoints],
            stream_config,
            interface_address=interface,
            ttl=ttl,
        )
    # Referencing the labels causes them to be created, in advance of data
    # actually being transmitted.
    for output_name in output_names:
        output_heaps_counter.labels(output_name)
        output_bytes_counter.labels(output_name)
        output_samples_counter.labels(output_name)
        output_clip_counter.labels(output_name)

    return stream
