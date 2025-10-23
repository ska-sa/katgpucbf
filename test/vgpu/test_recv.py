################################################################################
# Copyright (c) 2025, National Research Foundation (SARAO)
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

"""Unit tests for :mod:`katgpucbf.vgpu.recv`."""

from collections.abc import Generator
from unittest import mock

import numpy as np
import pytest
import spead2.recv.asyncio
import spead2.send.asyncio
from aiokatcp import DeviceStatus, Reading, Sensor, SensorSet

from katgpucbf import COMPLEX, N_POLS
from katgpucbf.recv import Chunk, make_sensors
from katgpucbf.spead import BEAM_ANTS_ID, BF_RAW_ID, FLAVOUR, FREQUENCY_ID, IMMEDIATE_FORMAT, TIMESTAMP_ID
from katgpucbf.utils import TimeConverter
from katgpucbf.vgpu import recv
from katgpucbf.vgpu.recv import METRIC_NAMESPACE, Layout, iter_chunks

from .. import PromDiff

POL_LABELS = ("x", "y")


@pytest.fixture
def layout() -> Layout:
    """Layout fixture."""
    return Layout(
        sample_bits=8,
        n_channels=32768,
        n_channels_per_substream=1024,
        n_batches_per_chunk=3,
        n_spectra_per_heap=32,
        heap_timestamp_step=2**24,
    )


class TestLayout:
    """Test :class:`~katgpucbf.vgpu.recv.Layout`."""

    def test_properties(self, layout: Layout) -> None:
        """Test the properties of :class:`~katgpucbf.vgpu.recv.Layout`."""
        assert layout.heap_bytes == 65536
        assert layout.n_pol_substreams == 32
        assert layout.chunk_batches == 3
        assert layout.batch_heaps == 64
        assert layout.chunk_timestamp_step == 3 * 2**24
        assert layout.heap_sample_count == 32768
        assert layout.chunk_heaps == 192
        assert layout.chunk_bytes == 192 * 65536


@pytest.fixture
def queues() -> list[spead2.InprocQueue]:
    """Create in-process queues."""
    return [spead2.InprocQueue() for _ in range(N_POLS)]


@pytest.fixture
def data_ringbuffer(layout) -> spead2.recv.asyncio.ChunkRingbuffer:
    """Create an asynchronous data chunk ringbuffer, to be shared by the receive streams."""
    # Size is mostly arbitrary i.e. large enough to not block
    return spead2.recv.asyncio.ChunkRingbuffer(100)


@pytest.fixture
def free_ringbuffer(layout) -> spead2.recv.ChunkRingbuffer:
    """Create asynchronous free chunk ringbuffer, to be used by the receive streams."""
    # Size is mostly arbitrary i.e. large enough to not block.
    # It also determined memory usage so should not be *too* large.
    return spead2.recv.ChunkRingbuffer(20)


@pytest.fixture
def stream_group(
    layout: Layout,
    data_ringbuffer: spead2.recv.asyncio.ChunkRingbuffer,
    free_ringbuffer: spead2.recv.ChunkRingbuffer,
    queues: list[spead2.InprocQueue],
) -> Generator[spead2.recv.ChunkStreamRingGroup, None, None]:
    """Create a receive stream group.

    It is connected to the :func:`queues` fixture for input and
    :func:`data_ringbuffer` for output.
    """
    stream_group = recv.make_stream_group(layout, data_ringbuffer, free_ringbuffer, -1, POL_LABELS)
    for _ in range(free_ringbuffer.maxsize):
        data = np.empty(
            (N_POLS, layout.n_batches_per_chunk, layout.n_channels, layout.n_spectra_per_heap, COMPLEX), np.int8
        )
        # Use np.ones to make sure the bits get zeroed out
        present = np.ones((N_POLS, layout.n_batches_per_chunk, layout.n_pol_substreams), np.uint8)
        chunk = Chunk(data=data, present=present, sink=stream_group)
        chunk.recycle()
    for stream, queue in zip(stream_group, queues, strict=True):
        stream.add_inproc_reader(queue)
    for stream in stream_group:
        stream.start()

    yield stream_group
    stream_group.stop()


@pytest.fixture
def send_stream(queues: list[spead2.InprocQueue]) -> "spead2.send.asyncio.AsyncStream":
    """Create a stream that feeds into :func:`stream_group`."""
    config = spead2.send.StreamConfig()
    return spead2.send.asyncio.InprocStream(spead2.ThreadPool(), queues, config)


def make_heap(timestamp: int, frequency: int, beam_ants: int, data: np.ndarray) -> spead2.send.Heap:
    """Build a heap with the standard items for tied-array-channelised-voltage."""
    heap = spead2.send.Heap(FLAVOUR)
    heap.add_item(spead2.Item(TIMESTAMP_ID, "", "", shape=(), format=IMMEDIATE_FORMAT, value=timestamp))
    heap.add_item(spead2.Item(FREQUENCY_ID, "", "", shape=(), format=IMMEDIATE_FORMAT, value=frequency))
    heap.add_item(spead2.Item(BEAM_ANTS_ID, "", "", shape=(), format=IMMEDIATE_FORMAT, value=beam_ants))
    heap.add_item(spead2.Item(BF_RAW_ID, "", "", shape=data.shape, dtype=data.dtype, value=data))
    return heap


def mask_present(
    data: np.ndarray[tuple[int, int, int, int, int], np.dtype[np.int8]],
    present: np.ndarray[tuple[int, int, int], np.dtype[np.uint8]],
) -> None:
    """Zero out elements of `data` corresponding to zero elements of `present`."""
    # Split the channel axis into substreams so that the leading dimensions match.
    view = data.reshape((*present.shape, -1, *data.shape[-2:]))
    view[~present.astype(bool)] = 0


def gen_heaps(
    layout: Layout,
    data: np.ndarray[tuple[int, int, int, int, int], np.dtype[np.int8]],
    first_timestamp: int,
) -> Generator[spead2.send.HeapReference, None, None]:
    """Generate heaps from an array of data.

    The `data` must have shape (pols, batches, channels, spectra_per_heap, COMPLEX)
    """
    n_batches = data.shape[1]
    assert data.shape == (N_POLS, n_batches, layout.n_channels, layout.n_spectra_per_heap, COMPLEX)
    timestamp = first_timestamp
    for t in range(n_batches):
        for s in range(layout.n_pol_substreams):
            ch0 = s * layout.n_channels_per_substream
            ch1 = (s + 1) * layout.n_channels_per_substream
            for p in range(N_POLS):
                heap_data = data[p, t, ch0:ch1]
                heap = make_heap(
                    timestamp=timestamp,
                    frequency=s * layout.n_channels_per_substream,
                    beam_ants=1,  # Value doesn't matter for now
                    data=heap_data,
                )
                yield spead2.send.HeapReference(heap, substream_index=p)
        timestamp += layout.heap_timestamp_step


class TestStreamGroup:
    """Test the stream group built by :func:`katgpucbf.fgpu.recv.make_stream_group`."""

    @pytest.fixture
    async def sensors(self) -> SensorSet:
        """Receiver sensors."""
        # This is an async fixture because make_sensors requires a running event loop
        # Large timeout so that it doesn't affect the test
        return make_sensors(sensor_timeout=1e6, prefixes=[f"{pol}." for pol in POL_LABELS])

    @pytest.fixture(autouse=True)
    def patch_max_chunks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Replace MAX_CHUNKS with a large number.

        This prevents heaps being evicted if one of the receiving threads runs
        faster than the other.
        """
        monkeypatch.setattr("katgpucbf.vgpu.recv.MAX_CHUNKS", 10)

    @pytest.mark.parametrize("missing", [pytest.param(False, id="nomissing"), pytest.param(True, id="missing")])
    @pytest.mark.parametrize("reorder", [pytest.param(False, id="noreorder"), pytest.param(True, id="reorder")])
    async def test_recv(
        self,
        layout: Layout,
        send_stream: "spead2.send.asyncio.AsyncStream",
        stream_group: spead2.recv.ChunkStreamRingGroup,
        queues: list[spead2.InprocQueue],
        data_ringbuffer: spead2.recv.asyncio.ChunkRingbuffer,
        time_converter: TimeConverter,
        sensors: SensorSet,
        reorder: bool,
        missing: bool,
    ) -> None:
        """Send heaps and check that they arrive and that metrics are correct.

        There are two parametrisations:

        missing
            If true, do not send any heaps from batches 4:10 (covering the
            tail of one chunk, the whole of the next, and the start of the
            third) plus one more heap.
        reorder
            If true, shuffle the heaps randomly before sending them.
        """
        n_batches = 25
        rng = np.random.default_rng(seed=1)
        data = rng.integers(
            -127, 127, size=(N_POLS, n_batches, layout.n_channels, layout.n_spectra_per_heap, COMPLEX), dtype=np.int8
        )
        first_chunk_id = 123
        first_timestamp = first_chunk_id * layout.chunk_timestamp_step
        heaps = list(gen_heaps(layout, data, first_timestamp))
        if missing:
            # Remove a contiguous range of heaps. This is not the most thorough
            # test, but katgpucbf.recv.iter_chunks already gets cover from fgpu
            # and xbgpu tests.
            del heaps[4 * layout.batch_heaps : 10 * layout.batch_heaps + 1]
        if reorder:
            # Ignore is needed due to https://github.com/numpy/numpy/issues/29974
            rng.shuffle(heaps)  # type: ignore

        with PromDiff(namespace=METRIC_NAMESPACE) as prom_diff:
            # Heap with no payload - representing any sort of metadata heap such as descriptors
            await send_stream.async_send_heap(spead2.send.Heap(FLAVOUR))
            for heap in heaps:
                await send_stream.async_send_heap(heap.heap, substream_index=heap.substream_index)
            for queue in queues:
                queue.stop()  # Flushes out the receive stream

            start_batch = 0
            expected_timestamp = first_timestamp
            i = 0
            async for chunk in iter_chunks(data_ringbuffer, layout, sensors, time_converter, POL_LABELS):
                with chunk:
                    if missing and i == 2:
                        # Previous chunk is entirely missing
                        start_batch += layout.n_batches_per_chunk
                        expected_timestamp += layout.chunk_timestamp_step
                    n_chunk_batches = min(layout.n_batches_per_chunk, n_batches - start_batch)
                    expected_present = np.zeros_like(chunk.present)
                    expected_present[:, :n_chunk_batches] = 1
                    expected_data = np.zeros_like(chunk.data)
                    expected_data[:, :n_chunk_batches] = data[:, start_batch : start_batch + n_chunk_batches]
                    if missing:
                        match i:
                            case 1:
                                expected_present[:, 1:] = 0
                            case 2:
                                expected_present[:, :1] = 0
                                expected_present[0, 1, 0] = 0
                    np.testing.assert_equal(chunk.present, expected_present)
                    # Mask out missing data from the comparison
                    mask_present(chunk.data, expected_present)
                    mask_present(expected_data, expected_present)
                    np.testing.assert_equal(chunk.data, expected_data)
                    start_batch += n_chunk_batches
                    expected_timestamp += layout.chunk_timestamp_step
                    if start_batch == n_batches:
                        break  # Received all the heaps we expected to
                i += 1

        assert prom_diff.diff("input_metadata_heaps_total") == 1
        assert prom_diff.diff("input_incomplete_heaps_total") == 0
        assert prom_diff.diff("input_too_old_heaps_total") == 0
        assert prom_diff.diff("input_bad_timestamp_heaps_total") == 0
        assert prom_diff.diff("input_bad_frequency_heaps_total") == 0
        total_chunks = n_batches // layout.n_batches_per_chunk + 1
        expected_chunks = total_chunks
        if missing:
            expected_chunks -= 1  # One whole chunk is missing
        assert prom_diff.diff("input_chunks_total") == expected_chunks
        for pol in POL_LABELS:
            expected_heaps = n_batches * layout.n_pol_substreams
            if missing:
                expected_heaps -= 6 * layout.n_pol_substreams
                if pol == "x":
                    expected_heaps -= 1
            assert prom_diff.diff("input_heaps_total", {"pol": pol}) == expected_heaps
            assert prom_diff.diff("input_bytes_total", {"pol": pol}) == expected_heaps * layout.heap_bytes
            assert prom_diff.diff("input_samples_total", {"pol": pol}) == expected_heaps * layout.heap_sample_count
            # The stream doesn't fill the last chunk
            expected_missing_heaps = total_chunks * layout.chunk_heaps // N_POLS - expected_heaps
            assert prom_diff.diff("input_missing_heaps_total", {"pol": pol}) == expected_missing_heaps

        last_chunk_id = first_chunk_id + total_chunks - 1
        last_chunk_timestamp = last_chunk_id * layout.chunk_timestamp_step
        last_chunk_timestamp_unix = time_converter.adc_to_unix(last_chunk_timestamp)
        for pol in POL_LABELS:
            assert sensors[f"{pol}.rx.timestamp"].reading == Reading(
                last_chunk_timestamp_unix, Sensor.Status.NOMINAL, last_chunk_timestamp
            )
            assert sensors[f"{pol}.rx.unixtime"].reading == Reading(
                last_chunk_timestamp_unix, Sensor.Status.NOMINAL, last_chunk_timestamp_unix
            )
            assert sensors[f"{pol}.rx.missing-unixtime"].reading == Reading(
                last_chunk_timestamp_unix, Sensor.Status.ERROR, last_chunk_timestamp_unix
            )
        assert sensors["rx.device-status"].reading == Reading(mock.ANY, Sensor.Status.WARN, DeviceStatus.DEGRADED)

    @pytest.mark.parametrize(
        "timestamp,frequency,bad_timestamps,bad_frequencies",
        [
            (12345, 0, 1, 0),  # Timestamp not a multiple of step
            (0, 13, 0, 1),  # Frequency not a multiple of n_channels_per_substream
            (0, 2**20, 0, 1),  # Frequency too large
        ],
    )
    async def test_bad_timestamp_frequency(
        self,
        layout: Layout,
        send_stream: "spead2.send.asyncio.AsyncStream",
        stream_group: spead2.recv.ChunkStreamRingGroup,
        queues: list[spead2.InprocQueue],
        data_ringbuffer: spead2.recv.asyncio.ChunkRingbuffer,
        time_converter: TimeConverter,
        sensors: SensorSet,
        timestamp: int,
        frequency: int,
        bad_timestamps: int,
        bad_frequencies: int,
    ) -> None:
        """Test flagging of a heap with an invalid timestamp or frequency.

        It sends a single heap with bad data, then checks the Prometheus
        counters.

        Parameters
        ----------
        timestamp
            Timestamp to store in the heap
        frequency
            Frequency (channel number) to store in the heap
        bad_timestamps
            Expected value for ``input_bad_timestamp_heaps`` counter
        bad_frequencies
            Expected value for ``input_bad_frequency_heaps`` counter
        """
        heap_data = np.zeros((layout.n_channels_per_substream, layout.n_spectra_per_heap, COMPLEX), np.int8)
        # Value of beam_ants is arbitrary for now.
        heap = make_heap(timestamp, frequency, 1, heap_data)
        with PromDiff(namespace=METRIC_NAMESPACE) as prom_diff:
            await send_stream.async_send_heap(heap, substream_index=0)
            for queue in queues:
                queue.stop()  # Flushes out the receive stream
            # We don't expect to receive any chunks, but waiting for the iterator
            # to run out ensures we process all the data.
            async for chunk in iter_chunks(data_ringbuffer, layout, sensors, time_converter, POL_LABELS):
                chunk.recycle()
        assert prom_diff.diff("input_bad_timestamp_heaps_total") == bad_timestamps
        assert prom_diff.diff("input_bad_frequency_heaps_total") == bad_frequencies
