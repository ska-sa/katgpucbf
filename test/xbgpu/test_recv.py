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
from typing import Generator, Iterator, List
from unittest.mock import Mock

import numpy as np
import pytest
import spead2.recv.asyncio
from numpy.typing import ArrayLike

from katgpucbf.spead import FENG_ID_ID, FENG_RAW_ID, FLAVOUR, FREQUENCY_ID, IMMEDIATE_FORMAT, TIMESTAMP_ID
from katgpucbf.xbgpu import METRIC_NAMESPACE, recv
from katgpucbf.xbgpu.recv import Chunk, Layout, recv_chunks

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
        heaps_per_fengine_per_chunk=5,
    )


class TestLayout:
    """Test :class:`.Layout`."""

    def test_properties(self, layout: Layout) -> None:
        """Test the properties of :class:`.Layout`."""
        assert layout.heap_bytes == 32768
        assert layout.chunk_heaps == 20
        assert layout.chunk_bytes == 655360


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
        chunk = Chunk(data=data, present=present)
        stream.add_free_chunk(chunk)
    stream.add_inproc_reader(queue)

    yield stream

    stream.stop()


@pytest.fixture
def send_stream(queue) -> "spead2.send.asyncio.AsyncStream":
    """Create a stream that feeds into :func:`stream`."""
    config = spead2.send.StreamConfig(max_packet_size=9000)
    return spead2.send.asyncio.InprocStream(
        spead2.ThreadPool(),
        [
            queue,
        ],
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
            heap_list: List[spead2.send.Heap] = []
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
            async for chunk in recv_chunks(stream):
                assert isinstance(chunk, Chunk)
                if not np.any(chunk.present):
                    # It's a chunk with no data. Currently spead2 may generate
                    # these due to the way it allocates chunks to keep the window
                    # full.
                    stream.add_free_chunk(chunk)
                    empty_chunks += 1
                    continue
                assert chunk.chunk_id == expected_chunk_id
                assert np.all(chunk.present)
                np.testing.assert_array_equal(chunk.data, data[: layout.chunk_bytes])
                data = data[layout.chunk_bytes :]  # Throw away the samples we've checked
                stream.add_free_chunk(chunk)
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
    ) -> None:
        """Test that the chunk placement sets heap indices correctly."""
        rng = np.random.default_rng(seed=1)
        data = rng.integers(-127, 127, size=layout.chunk_bytes, dtype=np.int8)
        expected_chunk_id = 123
        first_timestamp = expected_chunk_id * layout.timestamp_step * layout.heaps_per_fengine_per_chunk
        heaps = list(gen_heaps(layout, data, first_timestamp))
        # Create some gaps in the heaps
        missing = [0, 5, 7]
        for idx in reversed(missing):  # Have to go backwards, otherwise indices shift up
            del heaps[idx]

        for heap in heaps:
            await send_stream.async_send_heap(heap)
        queue.stop()  # Flushes out the receive streams
        # Get just the chunks that actually have some data. We needn't worry
        # about returning chunks to the free ring as we don't expect to deplete
        # it.
        chunks = [chunk async for chunk in recv_chunks(stream) if np.any(chunk.present)]
        assert len(chunks) == 1
        chunk = chunks[0]
        assert isinstance(chunk, Chunk)
        assert chunk.chunk_id == expected_chunk_id
        expected = np.ones_like(chunk.present)
        expected[missing] = 0
        np.testing.assert_equal(chunk.present, expected)

    async def test_missing_chunks(
        self,
        layout: Layout,
        caplog,
    ) -> None:
        """Test that that the receiver accounts for missing chunks.

        Heavily inspired by fgpu/test_recv.

        .. todo::
            This doesn't necessarily test the adding of free chunks back to
            the Ringbuffer. That logic lives in the XBEngine's gpu_proc_loop.
            We might want to bring that functionality to parity with fgpu's
            receiver.
        """
        mock_stream = Mock()  # We don't need a full-blown spead2 stream

        # Reality can be whatever we want
        config = spead2.recv.StreamConfig()
        # config.add_stat("incomplete_heaps_evicted")  # Odd, this one doesn't need to be added
        config.add_stat("too_old_heaps")
        config.add_stat("katgpucbf.metadata_heaps")
        config.add_stat("katgpucbf.bad_timestamp_heaps")
        config.add_stat("katgpucbf.bad_feng_id_heaps")
        mock_stream.config = config

        mock_stream.stats = {}
        mock_stream.stats[config.get_stat_index("too_old_heaps")] = 1010
        mock_stream.stats[config.get_stat_index("incomplete_heaps_evicted")] = 2020
        mock_stream.stats[config.get_stat_index("katgpucbf.metadata_heaps")] = 3030
        mock_stream.stats[config.get_stat_index("katgpucbf.bad_timestamp_heaps")] = 4040
        mock_stream.stats[config.get_stat_index("katgpucbf.bad_feng_id_heaps")] = 5050

        ringbuffer = spead2.recv.asyncio.ChunkRingbuffer(100)
        mock_stream.data_ringbuffer = ringbuffer

        rng = np.random.default_rng(seed=1)

        def add_chunk(chunk_id: int) -> None:
            data = rng.integers(-127, 127, size=layout.chunk_bytes, dtype=np.int8)
            # We're not testing missing heaps specifically
            present = np.ones(layout.chunk_heaps, np.uint8)
            chunk = Chunk(data=data, present=present, chunk_id=chunk_id)
            ringbuffer.put_nowait(chunk)

        # Receiver will naturally expect Chunk ID 0 first
        add_chunk(10)
        add_chunk(11)
        add_chunk(12)
        # Skip a few more
        add_chunk(20)
        ringbuffer.stop()

        with caplog.at_level(logging.WARNING, logger="katgpucbf.xbgpu.recv"), PromDiff(
            namespace=METRIC_NAMESPACE
        ) as prom_diff:
            # Register the stream with the StatsCollector manually
            recv.stats_collector.add_stream(mock_stream)

            # Need to actually grab the chunks
            chunks = [chunk async for chunk in recv_chunks(mock_stream)]

        # Double check the Chunk IDs received
        assert chunks[0].chunk_id == 10
        assert chunks[1].chunk_id == 11
        assert chunks[2].chunk_id == 12
        assert chunks[3].chunk_id == 20

        # TODO: Perhaps wrap this into a for-loop (somehow)
        assert caplog.record_tuples[0] == (
            "katgpucbf.xbgpu.recv",
            logging.WARNING,
            f"Receiver missed 10 chunks. Expected ID: 0, received ID: {chunks[0].chunk_id}.",
        )
        assert caplog.record_tuples[1] == (
            "katgpucbf.xbgpu.recv",
            logging.WARNING,
            f"Receiver missed 7 chunks. Expected ID: {chunks[2].chunk_id + 1}, received ID: {chunks[3].chunk_id}.",
        )

        # TODO: There must be a better way to phrase this calculation
        expected_missing_chunks = chunks[0].chunk_id - 0 + chunks[3].chunk_id - (chunks[2].chunk_id + 1)

        assert prom_diff.get_sample_diff("input_too_old_heaps_total") == 1010
        assert prom_diff.get_sample_diff("input_incomplete_heaps_total") == 2020
        assert prom_diff.get_sample_diff("input_metadata_heaps_total") == 3030
        assert prom_diff.get_sample_diff("input_bad_timestamp_heaps_total") == 4040
        assert prom_diff.get_sample_diff("input_bad_feng_id_heaps_total") == 5050
        assert prom_diff.get_sample_diff("input_missing_heaps_total") == expected_missing_chunks * layout.chunk_heaps
