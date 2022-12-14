################################################################################
# Copyright (c) 2020-2022, National Research Foundation (SARAO)
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

"""Unit tests for :mod:`katgpucbf.fgpu.recv`."""

import dataclasses
import itertools
import logging
from collections.abc import Generator, Iterable, Sequence
from typing import cast
from unittest.mock import Mock

import aiokatcp
import numpy as np
import pytest
import spead2.recv.asyncio
from aiokatcp import DeviceStatus
from numpy.typing import ArrayLike

from katgpucbf import N_POLS
from katgpucbf.fgpu import METRIC_NAMESPACE, recv
from katgpucbf.fgpu.recv import Chunk, Layout, make_sensors
from katgpucbf.spead import (
    ADC_SAMPLES_ID,
    DIGITISER_ID_ID,
    DIGITISER_STATUS_ID,
    DIGITISER_STATUS_SATURATION_COUNT_SHIFT,
    DIGITISER_STATUS_SATURATION_FLAG_BIT,
    FLAVOUR,
    TIMESTAMP_ID,
)
from katgpucbf.utils import TimeConverter

from .. import PromDiff


@pytest.fixture
def layout(request) -> Layout:
    """Return an example layout.

    The test may be decorated with ``pytest.mark.mask_timestamp`` to request
    timestamp masking.
    """
    mask_timestamp = request.node.get_closest_marker("mask_timestamp") is not None
    return Layout(sample_bits=10, heap_samples=4096, chunk_samples=65536, mask_timestamp=mask_timestamp)


class TestLayout:
    """Test :class:`.Layout`."""

    def test_properties(self, layout: Layout) -> None:
        """Test the properties of :class:`.Layout`."""
        assert layout.heap_bytes == 5120
        assert layout.chunk_bytes == 81920
        assert layout.chunk_heaps == 16
        assert layout.timestamp_mask == 2**64 - 1
        masked = dataclasses.replace(layout, mask_timestamp=True)
        assert masked.timestamp_mask == 2**64 - 4096


@pytest.fixture
def queues() -> list[spead2.InprocQueue]:
    """Create a in-process queue per polarization."""
    return [spead2.InprocQueue() for _ in range(N_POLS)]


@pytest.fixture
def data_ringbuffer(layout) -> spead2.recv.asyncio.ChunkRingbuffer:
    """Create an asynchronous data chunk ringbuffer, to be shared by the receive streams."""
    return spead2.recv.asyncio.ChunkRingbuffer(1)


@pytest.fixture
def free_ringbuffers(layout) -> list[spead2.recv.asyncio.ChunkRingbuffer]:
    """Create asynchronous free chunk ringbuffers, to be used by the receive streams."""
    return [spead2.recv.asyncio.ChunkRingbuffer(4) for _ in range(N_POLS)]


@pytest.fixture
def streams(
    layout, data_ringbuffer, free_ringbuffers, queues
) -> Generator[list[spead2.recv.ChunkRingStream], None, None]:
    """Create a receive stream per polarization.

    They are connected to the :func:`queues` fixture for input and
    :func:`data_ringbuffer` for output.
    """
    streams = recv.make_streams(layout, data_ringbuffer, free_ringbuffers, [-1, -1])
    for stream, queue, free_ringbuffer in zip(streams, queues, free_ringbuffers):
        for _ in range(free_ringbuffer.maxsize):
            data = np.empty(layout.chunk_bytes, np.uint8)
            # Use np.ones to make sure the bits get zeroed out
            present = np.ones(layout.chunk_heaps, np.uint8)
            extra = np.zeros(layout.chunk_heaps, np.uint16)
            chunk = Chunk(data=data, present=present, extra=extra, stream=stream)
            chunk.recycle()
        stream.add_inproc_reader(queue)

    yield streams

    for stream in streams:
        stream.stop()


@pytest.fixture
def send_stream(queues) -> "spead2.send.asyncio.AsyncStream":
    """Create a stream that feeds into :func:`streams`.

    It has one substream per polarisation.
    """
    config = spead2.send.StreamConfig(max_packet_size=9000)  # Just needs to be bigger than heaps
    return spead2.send.asyncio.InprocStream(spead2.ThreadPool(), queues, config)


def gen_heaps(
    layout: Layout,
    data: ArrayLike,
    first_timestamp: int,
    pol: int,
    present: Sequence[bool] | None = None,
    saturated: Sequence[int] | None = None,
) -> Generator[spead2.send.Heap, None, None]:
    """Generate heaps from an array of data.

    The data must be a 1D array of bytes, evenly divisible by the heap size.
    The heaps do not exactly match the real digitiser packet format, but
    contain all the relevant items to test the receiver code.

    If `present` is specified, it must contain one boolean per heap. If the
    boolean is false, the heap is skipped.

    If `present` is specified, it contains the saturation count for each heap
    (used to compute `digitiser_status`). If not specified, all samples are
    assumed to be unsaturated.
    """
    data_arr = np.require(data, dtype=np.uint8)
    assert data_arr.ndim == 1
    data_arr = data_arr.reshape(-1, layout.heap_bytes)  # One row per heap
    assert present is None or len(present) == data_arr.shape[0]
    timestamp = first_timestamp
    imm_format = [("u", FLAVOUR.heap_address_bits)]
    for i, row in enumerate(data_arr):
        if present is None or present[i]:
            if saturated is not None and saturated[i] > 0:
                status = (saturated[i] << DIGITISER_STATUS_SATURATION_COUNT_SHIFT) | (
                    1 << DIGITISER_STATUS_SATURATION_FLAG_BIT
                )
            else:
                status = 0
            heap = spead2.send.Heap(FLAVOUR)
            heap.add_item(spead2.Item(TIMESTAMP_ID, "", "", shape=(), format=imm_format, value=timestamp))
            heap.add_item(spead2.Item(DIGITISER_ID_ID, "", "", shape=(), format=imm_format, value=pol))
            heap.add_item(spead2.Item(DIGITISER_STATUS_ID, "", "", shape=(), format=imm_format, value=status))
            heap.add_item(spead2.Item(ADC_SAMPLES_ID, "", "", shape=row.shape, dtype=row.dtype, value=row))
            yield heap
        timestamp += layout.heap_samples


class TestStream:
    """Test the stream built by :func:`katgpucbf.recv.make_stream`."""

    @pytest.mark.parametrize("reorder", [True, False])
    @pytest.mark.parametrize("timestamps", ["good", "bad", pytest.param("mask", marks=[pytest.mark.mask_timestamp])])
    async def test_basic(
        self,
        layout: Layout,
        send_stream: "spead2.send.asyncio.AsyncStream",
        streams: list[spead2.recv.ChunkRingStream],
        queues: list[spead2.InprocQueue],
        data_ringbuffer: spead2.recv.asyncio.ChunkRingbuffer,
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

            mask
                Like "good" but all timestamps have some garbage in the low-order bits.
        """
        POL = 1  # Only test one of the pols # noqa: N806
        rng = np.random.default_rng(seed=1)
        data = rng.integers(0, 255, size=5 * layout.chunk_bytes, dtype=np.uint8)
        expected_chunk_id = 123
        first_timestamp = expected_chunk_id * layout.chunk_samples
        if timestamps == "mask":
            first_timestamp += 1234  # Invalid bits that will be masked off
        heaps: Iterable[spead2.send.Heap] = gen_heaps(layout, data, first_timestamp, POL)
        if reorder:
            heap_list = list(heaps)
            # In each group of 12 heaps, swap the first and the last. The
            # stride is non-power-of-2 to ensure that this will swap heaps
            # across chunk boundaries.
            for i in range(11, len(heap_list), 12):
                heap_list[i - 11], heap_list[i] = heap_list[i], heap_list[i - 11]
            heaps = heap_list
        if timestamps == "bad":
            bad_heaps = gen_heaps(layout, ~data, first_timestamp + 1234567, POL)
            # Interleave the sequences
            heaps = itertools.chain.from_iterable(zip(heaps, bad_heaps))

        # Heap with no payload - representing any sort of metadata heap such as descriptors
        heap = spead2.send.Heap(FLAVOUR)
        await send_stream.async_send_heap(heap, substream_index=POL)
        for heap in heaps:
            await send_stream.async_send_heap(heap, substream_index=POL)
        for queue in queues:
            queue.stop()  # Flushes out the receive streams
        seen = 0
        async for chunk in data_ringbuffer:
            assert isinstance(chunk, Chunk)
            with chunk:
                if not np.any(chunk.present):
                    # It's a chunk with no data. Currently spead2 may generate
                    # these due to the way it allocates chunks to keep the window
                    # full.
                    continue
                assert chunk.stream_id == POL
                assert chunk.chunk_id == expected_chunk_id
                assert np.all(chunk.present)
                np.testing.assert_array_equal(chunk.data, data[: layout.chunk_bytes])
                data = data[layout.chunk_bytes :]  # Throw away the samples we've checked
                seen += 1
                expected_chunk_id += 1
        assert seen == 5
        expected_bad_timestamps = seen * layout.chunk_heaps if timestamps == "bad" else 0
        assert streams[POL].stats["katgpucbf.metadata_heaps"] == 1
        assert streams[POL].stats["katgpucbf.bad_timestamp_heaps"] == expected_bad_timestamps

    async def test_missing_heaps(
        self,
        layout: Layout,
        send_stream: "spead2.send.asyncio.AsyncStream",
        streams: list[spead2.recv.ChunkRingStream],
        queues: list[spead2.InprocQueue],
        data_ringbuffer: spead2.recv.asyncio.ChunkRingbuffer,
    ) -> None:
        """Test that the chunk placement sets heap indices correctly."""
        POL = 1  # Only test one of the pols # noqa: N806
        rng = np.random.default_rng(seed=1)
        data = rng.integers(0, 255, size=layout.chunk_bytes, dtype=np.uint8)
        expected_chunk_id = 123
        first_timestamp = expected_chunk_id * layout.chunk_samples
        heaps = list(gen_heaps(layout, data, first_timestamp, POL))
        # Create some gaps in the heaps
        missing = [0, 5, 7]
        for idx in reversed(missing):  # Have to go backwards, otherwise indices shift up
            del heaps[idx]

        for heap in heaps:
            await send_stream.async_send_heap(heap, substream_index=POL)
        for queue in queues:
            queue.stop()  # Flushes out the receive streams
        # Get just the chunks that actually have some data. We needn't worry
        # about returning chunks to the free ring as we don't expect to deplete
        # it.
        chunks = [chunk async for chunk in data_ringbuffer if np.any(chunk.present)]  # type: ignore
        assert len(chunks) == 1
        chunk = chunks[0]
        assert isinstance(chunk, Chunk)
        assert chunk.chunk_id == expected_chunk_id
        assert chunk.stream_id == POL
        expected = np.ones_like(chunk.present)
        expected[missing] = 0
        np.testing.assert_equal(chunk.present, expected)


class TestChunkSets:
    """Test :func:`.chunk_sets`."""

    @pytest.fixture
    async def sensors(self) -> aiokatcp.SensorSet:
        """Receiver sensors."""
        # This is an async fixture because make_sensors requires a running event loop
        return make_sensors(sensor_timeout=1e6)  # Large timeout so that it doesn't affect the test

    async def test(
        self, layout: Layout, sensors: aiokatcp.SensorSet, time_converter: TimeConverter, caplog
    ) -> None:  # noqa: D102
        streams = [Mock() for _ in range(N_POLS)]
        # Fake up stream stats
        config = spead2.recv.StreamConfig()
        config.add_stat("too_old_heaps")
        config.add_stat("katgpucbf.bad_timestamp_heaps")
        config.add_stat("katgpucbf.metadata_heaps")
        for pol, stream in enumerate(streams):
            stream.config = config
            stream.stats = {}
            stream.stats[config.get_stat_index("too_old_heaps")] = 111 + 1000 * pol
            stream.stats[config.get_stat_index("katgpucbf.bad_timestamp_heaps")] = 123 + 1000 * pol
            stream.stats[config.get_stat_index("katgpucbf.metadata_heaps")] = 321 + 1000 * pol

        ringbuffer = spead2.recv.asyncio.ChunkRingbuffer(100)  # Big enough not to worry about
        for stream in streams:
            stream.data_ringbuffer = ringbuffer
        rng = np.random.default_rng(1)
        expected_clip: dict[tuple[int, int], int] = {}  # Maps (chunk_id, pol) to total clip count

        def add_chunk(chunk_id: int, pol: int, missing: int = 0) -> None:
            data = rng.integers(0, 255, size=layout.chunk_bytes, dtype=np.uint8)
            present = np.ones(layout.chunk_heaps, np.uint8)
            present[:missing] = 0  # Mark some leading heaps as missing
            extra = rng.integers(0, layout.heap_samples - 1, size=layout.chunk_heaps, dtype=np.uint16)
            expected_clip[chunk_id, pol] = int(np.sum(extra[missing:], dtype=np.int64))
            chunk = Chunk(
                data=data, present=present, extra=extra, chunk_id=chunk_id, stream_id=pol, stream=streams[pol]
            )
            ringbuffer.put_nowait(chunk)

        for i in range(10):
            # Throw in some empty chunks, to match what spead2 does
            for pol in range(N_POLS):
                add_chunk(i, pol, missing=layout.chunk_heaps)
        add_chunk(10, 0)
        add_chunk(10, 1)
        add_chunk(11, 1)
        add_chunk(12, 1, 1)
        add_chunk(20, 1)
        add_chunk(12, 0, 2)
        add_chunk(20, 0)
        add_chunk(21, 0)
        ringbuffer.stop()

        with caplog.at_level(logging.WARNING, logger="katgpucbf.fgpu.recv"), PromDiff(
            namespace=METRIC_NAMESPACE
        ) as prom_diff:
            # We're not using make_stream, so we have to register the stream with
            # the stats collector manually.
            for pol, stream in enumerate(streams):
                recv.stats_collector.add_stream(stream, labels=[str(pol)])
            sets = [
                chunk_set
                async for chunk_set in recv.chunk_sets(
                    cast(list[spead2.recv.ChunkRingStream], streams), layout, sensors, time_converter
                )
            ]
        assert caplog.record_tuples == [
            ("katgpucbf.fgpu.recv", logging.WARNING, "Chunk not matched: timestamp=0xb0000 pol=1")
        ]
        assert len(sets) == 3
        # Check that all the pairs are consistent
        for s in sets:
            for pol in range(N_POLS):
                assert s[pol].stream_id == pol
                assert s[pol].chunk_id == s[0].chunk_id
                assert s[pol].timestamp == s[0].chunk_id * layout.chunk_samples
        expected_ids = [10, 12, 20]
        assert [s[0].chunk_id for s in sets] == expected_ids
        assert sets[0][0].chunk_id == 10
        assert sets[1][0].chunk_id == 12
        assert sets[2][0].chunk_id == 20
        # Check that the mismatched / empty chunks were returned to the free ring
        assert streams[0].add_free_chunk.call_count == 11
        assert streams[0].add_free_chunk.call_args[0][0].chunk_id == 21
        assert streams[1].add_free_chunk.call_count == 11
        assert streams[1].add_free_chunk.call_args[0][0].chunk_id == 11

        # Check metrics
        def get_sample_diffs(name: str) -> list[float | None]:
            return [prom_diff.get_sample_diff(name, {"pol": str(pol)}) for pol in range(N_POLS)]

        assert get_sample_diffs("input_heaps_total") == [46, 47]
        assert get_sample_diffs("input_chunks_total") == [3, 3]
        assert get_sample_diffs("input_samples_total") == [46 * 4096, 47 * 4096]
        assert get_sample_diffs("input_bytes_total") == [46 * 5120, 47 * 5120]
        assert get_sample_diffs("input_too_old_heaps_total") == [111, 1111]
        assert get_sample_diffs("input_missing_heaps_total") == [11 * 16 - 46, 11 * 16 - 47]
        assert get_sample_diffs("input_bad_timestamp_heaps_total") == [123, 1123]
        assert get_sample_diffs("input_metadata_heaps_total") == [321, 1321]
        expected_clip_total = [sum(expected_clip[chunk_id, pol] for chunk_id in expected_ids) for pol in range(N_POLS)]
        assert get_sample_diffs("input_clipped_samples_total") == expected_clip_total

        # Check sensors
        sensor = sensors["input0-rx-timestamp"]
        assert sensor.value == 21 * layout.chunk_samples
        assert sensor.status == aiokatcp.Sensor.Status.NOMINAL
        assert sensor.timestamp == time_converter.adc_to_unix(sensor.value)
        sensor = sensors["input1-rx-timestamp"]
        assert sensor.value == 20 * layout.chunk_samples
        assert sensor.status == aiokatcp.Sensor.Status.NOMINAL
        assert sensor.timestamp == time_converter.adc_to_unix(sensor.value)
        sensor = sensors["input0-rx-unixtime"]
        assert sensor.value == time_converter.adc_to_unix(21 * layout.chunk_samples)
        assert sensor.status == aiokatcp.Sensor.Status.NOMINAL
        assert sensor.timestamp == sensor.value
        sensor = sensors["input1-rx-unixtime"]
        assert sensor.value == time_converter.adc_to_unix(20 * layout.chunk_samples)
        assert sensor.status == aiokatcp.Sensor.Status.NOMINAL
        assert sensor.timestamp == sensor.value
        for pol in range(N_POLS):
            sensor = sensors[f"input{pol}-rx-missing-unixtime"]
            assert sensor.value == time_converter.adc_to_unix(20 * layout.chunk_samples)
            assert sensor.status == aiokatcp.Sensor.Status.ERROR
            assert sensor.timestamp == sensor.value
        ds_sensor = sensors["rx.device-status"]
        assert ds_sensor.value == DeviceStatus.DEGRADED
        assert ds_sensor.status == aiokatcp.Sensor.Status.WARN
