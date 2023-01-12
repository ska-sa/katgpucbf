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

"""Unit tests for :mod:`katgpucbf.xbgpu.recv`."""

import itertools
import logging
import random
from collections.abc import Generator, Iterator

import numpy as np
import pytest
import spead2.recv.asyncio
from aiokatcp import DeviceStatus, Sensor, SensorSet
from numpy.typing import ArrayLike

from katgpucbf.spead import FENG_ID_ID, FENG_RAW_ID, FLAVOUR, FREQUENCY_ID, IMMEDIATE_FORMAT, TIMESTAMP_ID
from katgpucbf.utils import TimeConverter
from katgpucbf.xbgpu import METRIC_NAMESPACE, recv
from katgpucbf.xbgpu.recv import Chunk, Layout, make_sensors, recv_chunks

from .. import PromDiff


@pytest.fixture
def layout() -> Layout:
    """Return an example layout."""
    return Layout(
        n_ants=4,
        n_channels_per_stream=1024 // 4,
        n_spectra_per_heap=32,
        timestamp_step=2 * 1024 * 32,
        sample_bits=8,
        heaps_per_fengine_per_chunk=10,
    )


class TestLayout:
    """Test :class:`.Layout`."""

    def test_properties(self, layout: Layout) -> None:
        """Test the properties of :class:`.Layout`."""
        assert layout.heap_bytes == 32768
        assert layout.chunk_heaps == 4 * 10
        assert layout.chunk_bytes == 1310720


@pytest.fixture
def queue() -> spead2.InprocQueue:
    """Create a in-process queue."""
    return spead2.InprocQueue()


@pytest.fixture
def stream(layout, queue) -> Generator[spead2.recv.ChunkRingStream, None, None]:
    """Create a receive stream.

    It is connected to the :func:`queue` fixture for input and
    :func:`data_ringbuffer` for output.
    """
    max_active_chunks = 5
    data_ringbuffer = spead2.recv.asyncio.ChunkRingbuffer(max_active_chunks)
    n_chunks_total = max_active_chunks + 8  # 8 is just a few more.
    free_ringbuffer = spead2.recv.ChunkRingbuffer(n_chunks_total)
    stream = recv.make_stream(layout, data_ringbuffer, free_ringbuffer, -1, max_active_chunks)
    for _ in range(n_chunks_total):
        data = np.empty(layout.chunk_bytes, np.int8)
        # Use np.ones to make sure the bits get zeroed out
        present = np.ones(layout.chunk_heaps, np.uint8)
        chunk = Chunk(data=data, present=present, stream=stream)
        chunk.recycle()
    stream.add_inproc_reader(queue)

    yield stream

    stream.stop()


@pytest.fixture
def send_stream(queue) -> "spead2.send.asyncio.AsyncStream":
    """Create a stream that feeds into :func:`stream`."""
    config = spead2.send.StreamConfig(max_packet_size=9000)
    return spead2.send.asyncio.InprocStream(
        spead2.ThreadPool(),
        [queue],
        config,
    )


def gen_heap(timestamp: int, feng_id: int, frequency: int, feng_raw: np.ndarray) -> spead2.send.Heap:
    """Create a single heap."""
    heap = spead2.send.Heap(FLAVOUR)
    heap.add_item(spead2.Item(TIMESTAMP_ID, "", "", shape=(), format=IMMEDIATE_FORMAT, value=timestamp))
    heap.add_item(spead2.Item(FENG_ID_ID, "", "", shape=(), format=IMMEDIATE_FORMAT, value=feng_id))
    heap.add_item(spead2.Item(FREQUENCY_ID, "", "", shape=(), format=IMMEDIATE_FORMAT, value=frequency))
    heap.add_item(spead2.Item(FENG_RAW_ID, "", "", shape=feng_raw.shape, dtype=feng_raw.dtype, value=feng_raw))
    heap.repeat_pointers = True
    return heap


def gen_heaps(layout: Layout, data: ArrayLike, first_timestamp: int) -> Generator[spead2.send.Heap, None, None]:
    """Generate heaps from an array of data.

    The data must be a 1D array of bytes, evenly divisible by the heap size.
    """
    data_arr = np.require(data, dtype=np.int8)
    assert data_arr.ndim == 1
    data_arr = data_arr.reshape(-1, layout.n_ants, layout.heap_bytes)  # One row per heap
    # The term "batch" here takes the same meaning as in the help text of
    # the `--heaps-per-fengine-per-chunk` parser argument in main.py.
    for batch_id, batch in enumerate(data_arr):
        for feng_id, feng_data in enumerate(batch):
            timestamp = first_timestamp + batch_id * layout.timestamp_step
            heap = gen_heap(timestamp, feng_id, 0, feng_data)
            yield heap


class TestStream:
    """Test the stream built by :func:`katgpucbf.recv.make_stream`."""

    @pytest.fixture
    async def sensors(self) -> SensorSet:
        """Receiver sensors."""
        # This is an async fixture because make_sensors requires a running event loop
        return make_sensors(sensor_timeout=1e6)  # Large timeout so that it doesn't affect the test

    @pytest.mark.parametrize("reorder", [False, True])
    @pytest.mark.parametrize("timestamps", ["good", "bad"])
    async def test_basic(
        self,
        layout: Layout,
        send_stream: "spead2.send.asyncio.AsyncStream",
        stream: spead2.recv.ChunkRingStream,
        queue: spead2.InprocQueue,
        reorder: bool,
        timestamps: str,
        sensors: SensorSet,
        time_converter: TimeConverter,
    ) -> None:
        """Send heaps and check that they arrive.

        The parameters modify the test as follows:

        reorder
            Introduce a slight reordering into the heaps, to check that this
            is handled correctly.

        timestamps
            One of

            good
                All timestamps are valid.

            bad
                Valid heaps are interleaved with heaps with invalid timestamps.
        """
        rng = np.random.default_rng(seed=1)
        data = rng.integers(-127, 127, size=5 * layout.chunk_bytes, dtype=np.int8)
        expected_chunk_id = 123
        first_timestamp = expected_chunk_id * layout.timestamp_step * layout.heaps_per_fengine_per_chunk

        heaps: Iterator[spead2.send.Heap] = gen_heaps(layout, data, first_timestamp)
        if reorder:
            # We don't shuffle the first few heaps, this just makes sure
            # that we get chunk 123 first, as expected.
            heap_list: list[spead2.send.Heap] = []
            for _ in range(2):
                heap_list.append(next(heaps))
            # The rest are going to be pretty well shuffled. Heaps from a given
            # f-engine are unlikely to arrive out-of-order, we anticipate that
            # they'll be out-of-sync with each other, but it's possibly not
            # worth the effort to emulate that so I just shuffle them all.
            r = random.Random(123)
            temp_heap_list = list(heaps)
            r.shuffle(temp_heap_list)
            heap_list.extend(temp_heap_list)
            heaps = iter(heap_list)
        if timestamps == "bad":
            bad_heaps = gen_heaps(layout, ~data, first_timestamp + 1234567)
            # Interleave the sequences
            heaps = itertools.chain.from_iterable(zip(heaps, bad_heaps))

        with PromDiff(namespace=METRIC_NAMESPACE) as prom_diff:
            # Heap with no payload - representing any sort of metadata heap such as descriptors
            heap = spead2.send.Heap(FLAVOUR)
            await send_stream.async_send_heap(heap)

            # Send a heap with a bad F-engine ID
            heap = gen_heap(first_timestamp, 5, 0, np.zeros((layout.heap_bytes,), dtype=np.int8))
            await send_stream.async_send_heap(heap)

            # Now the bulk of the actual data
            for heap in heaps:
                await send_stream.async_send_heap(heap)

            # Finally a heap from the distant past
            heap = gen_heap(
                first_timestamp - 8 * layout.timestamp_step, 0, 0, np.zeros((layout.heap_bytes,), dtype=np.int8)
            )
            await send_stream.async_send_heap(heap)

            queue.stop()  # Flushes out the receive stream
            seen = 0
            empty_chunks = 0
            async for chunk in recv_chunks(stream, layout=layout, sensors=sensors, time_converter=time_converter):
                assert isinstance(chunk, Chunk)
                with chunk:
                    if not np.any(chunk.present):
                        # It's a chunk with no data. Currently spead2 may generate
                        # these due to the way it allocates chunks to keep the window
                        # full.
                        empty_chunks += 1
                        continue
                    assert chunk.chunk_id == expected_chunk_id
                    assert np.all(chunk.present)
                    np.testing.assert_array_equal(chunk.data, data[: layout.chunk_bytes])
                    data = data[layout.chunk_bytes :]  # Throw away the samples we've checked
                    seen += 1
                    expected_chunk_id += 1
        assert seen == 5
        expected_bad_timestamps = seen * layout.chunk_heaps if timestamps == "bad" else 0

        assert prom_diff.get_sample_diff("input_chunks_total") == seen + empty_chunks
        assert prom_diff.get_sample_diff("input_heaps_total") == (seen + empty_chunks) * layout.chunk_heaps
        assert prom_diff.get_sample_diff("input_bytes_total") == layout.chunk_bytes * seen
        assert prom_diff.get_sample_diff("input_bad_timestamp_heaps_total") == expected_bad_timestamps
        assert prom_diff.get_sample_diff("input_bad_feng_id_heaps_total") == 1
        assert prom_diff.get_sample_diff("input_metadata_heaps_total") == 1
        assert prom_diff.get_sample_diff("input_too_old_heaps_total") == 1

    async def test_missing_heaps(
        self,
        layout: Layout,
        send_stream: "spead2.send.asyncio.AsyncStream",
        stream: spead2.recv.ChunkRingStream,
        queue: spead2.InprocQueue,
        sensors: SensorSet,
        time_converter: TimeConverter,
        caplog,
    ) -> None:
        """Test that the receiver handles missing heaps and Chunks.

        Start from a non-zero Chunk ID to check the receiving logic doesn't
        assume we missed Chunks up to that point.
        """
        rng = np.random.default_rng(seed=1)
        data = rng.integers(-127, 127, size=layout.chunk_bytes, dtype=np.int8)
        start_chunk_id = 123
        n_chunks_to_send = 20
        n_single_heaps_to_delete = 7
        n_chunks_to_delete = 6
        heaps_to_send = []

        for i in range(n_chunks_to_send):
            timestamp = (start_chunk_id + i) * layout.timestamp_step * layout.heaps_per_fengine_per_chunk
            heaps_to_send += list(gen_heaps(layout, data, timestamp))

        expected_chunk_presence_flat = np.ones(
            shape=((n_chunks_to_send - n_chunks_to_delete) * layout.chunk_heaps,), dtype=np.uint8
        )

        # Create a gap in the heaps to send, enough to make the receiver
        # 'fast forward' in the Chunk IDs received, but also test the
        # case of a single heap missing every now and then.
        missing_heap_ids = [7 + i * layout.chunk_heaps for i in range(n_single_heaps_to_delete)]
        # Update these heap indices now, as the following removes entire Chunks
        for missing_heap_id in missing_heap_ids:
            expected_chunk_presence_flat[missing_heap_id] = 0

        missing_chunk_ids = [i for i in range(9, 9 + n_chunks_to_delete)]
        for missing_chunk_id in missing_chunk_ids:
            start_heap_id = missing_chunk_id * layout.chunk_heaps
            end_heap_id = (missing_chunk_id + 1) * layout.chunk_heaps
            missing_heap_ids += [i for i in range(start_heap_id, end_heap_id)]
        missing_heap_ids.sort()  # Just to be sure
        # This is done in reverse as the indices would otherwise shift up
        for heap_idx in reversed(missing_heap_ids):
            del heaps_to_send[heap_idx]

        for heap in heaps_to_send:
            await send_stream.async_send_heap(heap)
        queue.stop()  # Flushes out the receive stream

        # Need to compare present arrays during the handling of Chunks
        # - Initialise to zeroes to automatically catch mismatches,
        #   but this array should be completely overwritten anyway.
        received_chunk_presence = np.zeros(
            shape=(n_chunks_to_send - n_chunks_to_delete, layout.chunk_heaps), dtype=np.uint8
        )
        received_chunk_ids = []
        with caplog.at_level(logging.WARNING, logger="katgpucbf.xbgpu.recv"), PromDiff(
            namespace=METRIC_NAMESPACE
        ) as prom_diff:
            # Manually register the receiving stream with the StatsCollector,
            # as we haven't used the `recv.make_stream` utility.
            recv.stats_collector.add_stream(stream)

            # NOTE: We have to use a 'manual' counter as there is a jump in
            # received Chunk IDs - due to the deletions earlier.
            seen = 0
            async for chunk in recv_chunks(stream, layout=layout, sensors=sensors, time_converter=time_converter):
                with chunk:
                    # recv_chunks should filter out the phantom chunks created by
                    # spead2.
                    assert np.any(chunk.present)
                    received_chunk_ids.append(chunk.chunk_id)
                    received_chunk_presence[seen, :] = chunk.present
                    seen += 1

        absolute_missing_chunk_id = start_chunk_id + missing_chunk_ids[0] + n_chunks_to_delete
        assert caplog.record_tuples == [
            (
                "katgpucbf.xbgpu.recv",
                logging.WARNING,
                f"Receiver missed {n_chunks_to_delete} chunks. Expected ID: {start_chunk_id + missing_chunk_ids[0]}, "
                f"received ID: {absolute_missing_chunk_id}.",
            )
        ]

        # Have to expand this list as the `range` generator doesn't support item deletion
        expected_chunk_ids = [i for i in range(start_chunk_id, start_chunk_id + n_chunks_to_send)]
        for missing_chunk_id in reversed(missing_chunk_ids):
            del expected_chunk_ids[missing_chunk_id]
        np.testing.assert_equal(received_chunk_ids, expected_chunk_ids)
        np.testing.assert_array_equal(received_chunk_presence.flatten(), expected_chunk_presence_flat)

        # Check StatsCollector's values
        assert prom_diff.get_sample_diff("input_heaps_total") == seen * layout.chunk_heaps - n_single_heaps_to_delete
        assert (
            prom_diff.get_sample_diff("input_missing_heaps_total")
            == n_chunks_to_delete * layout.chunk_heaps + n_single_heaps_to_delete
        )

        # Check sensors
        sensor = sensors["input.rx-timestamp"]
        # sensor.value should be of the last chunk sent
        absolute_present_timestamp = expected_chunk_ids[-1] * layout.timestamp_step * layout.heaps_per_fengine_per_chunk
        assert sensor.value == absolute_present_timestamp
        assert sensor.status == Sensor.Status.NOMINAL
        assert sensor.timestamp == time_converter.adc_to_unix(sensor.value)
        sensor = sensors["input.rx-unixtime"]
        # Should be the same value as the previous sensor, but in UNIX time
        assert sensor.value == time_converter.adc_to_unix(absolute_present_timestamp)
        assert sensor.status == Sensor.Status.NOMINAL
        assert sensor.timestamp == sensor.value
        sensor = sensors["input.rx-missing-unixtime"]
        # sensor.value should be of the last chunk to go missing
        absolute_missing_timestamp = (
            absolute_missing_chunk_id * layout.timestamp_step * layout.heaps_per_fengine_per_chunk
        )
        assert sensor.value == time_converter.adc_to_unix(absolute_missing_timestamp)
        assert sensor.status == Sensor.Status.ERROR
        assert sensor.timestamp == sensor.value
        ds_sensor = sensors["rx.device-status"]
        assert ds_sensor.value == DeviceStatus.DEGRADED
        assert ds_sensor.status == Sensor.Status.WARN
