################################################################################
# Copyright (c) 2023, National Research Foundation (SARAO)
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

"""Unit tests for :mod:`katgpucbf.fgpu.send`."""

import asyncio
from typing import Sequence
from unittest import mock

import aiokatcp
import numpy as np
import pytest
import spead2.recv.asyncio
import spead2.send.asyncio
from katsdptelstate.endpoint import endpoint_list_parser

from katgpucbf import BYTE_BITS, COMPLEX, N_POLS
from katgpucbf.fgpu import METRIC_NAMESPACE
from katgpucbf.fgpu.send import Chunk, make_descriptor_heap, make_streams
from katgpucbf.utils import TimeConverter, gaussian_dtype

from .. import PromDiff, unpack_complex

pytest_mark = pytest.mark.parametrize("bits", [4, 8])
ADC_SAMPLE_RATE = 1e9
N_SUBSTREAMS = 16
N_CHUNKS = 5
N_BATCHES = 7  # batches per chunk
N_CHANNELS = 1024
N_SPECTRA_PER_HEAP = 32  # Small to make the test fast
SPECTRA_SAMPLES = 2 * N_CHANNELS
FENG_ID = 3
NAME = "foo"


@pytest.fixture(params=[4, 8])
def sample_bits(request) -> int:
    return request.param


@pytest.fixture(params=[1, 2])
def interfaces(request) -> Sequence[str]:
    # request.param gives number of interfaces to use
    return ["10.0.0.1", "10.0.0.2"][: request.param]


@pytest.fixture
def time_converter() -> TimeConverter:
    return TimeConverter(1234567890.0, ADC_SAMPLE_RATE)


@pytest.fixture
def queues(interfaces: Sequence[str]) -> dict[str, list[spead2.InprocQueue]]:
    """Create in-process queues to connect the test code to the test."""
    return {iface: [spead2.InprocQueue() for _ in range(N_SUBSTREAMS)] for iface in interfaces}


@pytest.fixture
def chunks(sample_bits) -> list[Chunk]:
    dtype = gaussian_dtype(sample_bits)
    return [
        Chunk(
            np.zeros((N_BATCHES, N_CHANNELS, N_SPECTRA_PER_HEAP, N_POLS), dtype),
            np.zeros((N_BATCHES, N_POLS), np.uint32),
            n_substreams=N_SUBSTREAMS,
            feng_id=FENG_ID,
            spectra_samples=SPECTRA_SAMPLES,
        )
        for _ in range(N_CHUNKS)
    ]


@pytest.fixture
def send_streams(
    queues: dict[str, list[spead2.InprocQueue]], chunks: list[Chunk]
) -> list["spead2.send.asyncio.AsyncStream"]:
    """Create the send streams.

    The actual stream constructors are mocked so that we use in-process
    streams.
    """
    expected_endpoints = endpoint_list_parser(None)(f"239.102.1.0+{N_SUBSTREAMS - 1}:7148")

    # Note: this function signature is somewhat fragile. It's built to match the
    # call in send.py.
    def make_inproc_stream(
        thread_pool: spead2.ThreadPool,
        endpoints: list[tuple[str, int]],
        config: spead2.send.StreamConfig,
        interface_address: str,
        **kwargs,
    ) -> spead2.send.asyncio.InprocStream:
        assert endpoints == [tuple(x) for x in expected_endpoints]
        return spead2.send.asyncio.InprocStream(thread_pool, queues[interface_address], config)

    with mock.patch("spead2.send.asyncio.UdpStream", make_inproc_stream):
        # These are somewhat typical values and generally match the defaults in
        # katgpucbf.fgpu.main. Most of them don't actually matter, because they
        # control stream creation and that's been mocked out, or they control
        # transmission rate.
        return make_streams(
            output_name=NAME,
            thread_pool=spead2.ThreadPool(1),
            endpoints=expected_endpoints,
            interfaces=list(queues.keys()),
            ttl=4,
            ibv=False,
            packet_payload=8192,
            comp_vector=0,
            buffer=65536,
            bandwidth=0.5 * ADC_SAMPLE_RATE,
            send_rate_factor=0.0,  # Just send as fast as possible
            feng_id=FENG_ID,
            num_ants=64,
            n_data_heaps=N_CHUNKS * N_BATCHES * N_SUBSTREAMS,
            chunks=chunks,
        )


@pytest.fixture
def recv_streams(queues: dict[str, list[spead2.InprocQueue]]) -> dict[str, list[spead2.recv.asyncio.Stream]]:
    """Create streams for receiving the transmitted data."""
    config = spead2.recv.StreamConfig()
    streams: dict[str, list[spead2.recv.asyncio.Stream]] = {}
    for iface in queues.keys():
        streams[iface] = []
        for queue in queues[iface]:
            tp = spead2.ThreadPool(1)
            stream = spead2.recv.asyncio.Stream(tp, config)
            stream.add_inproc_reader(queue)
            streams[iface].append(stream)
    return streams


@pytest.fixture
def sensors() -> aiokatcp.SensorSet:
    """Create sensors that the send code updates."""
    sensors = aiokatcp.SensorSet()
    for pol in range(N_POLS):
        sensors.add(
            aiokatcp.Sensor(
                int,
                f"{NAME}.input{pol}.feng-clip-cnt",
                "Number of output samples that are saturated",
            )
        )
    return sensors


def _fill_random(data: np.ndarray, rng: np.random.Generator) -> None:
    """Fill an array with random bytes.

    The underlying bytes are filled, rather than assuming any particular dtype.
    """
    view = data.view(np.uint8)
    view[:] = rng.integers(0, 256, view.shape, dtype=np.uint8)


def test_bad_substreams():
    """Test that :class:`Chunk` raises an exception if n_substreams doesn't divide n_channels.

    This is really just for code coverage; nothing really depends on it.
    """
    dtype = gaussian_dtype(8)
    with pytest.raises(ValueError):
        Chunk(
            np.zeros((N_BATCHES, N_CHANNELS, N_SPECTRA_PER_HEAP, N_POLS), dtype),
            np.zeros((N_BATCHES, N_POLS), np.uint32),
            n_substreams=5,
            feng_id=FENG_ID,
            spectra_samples=SPECTRA_SAMPLES,
        )


async def test_send(
    sample_bits: int,
    queues: dict[str, list[spead2.InprocQueue]],
    send_streams: list["spead2.send.asyncio.AsyncStream"],
    recv_streams: dict[str, list[spead2.recv.asyncio.Stream]],
    chunks: list[Chunk],
    sensors: aiokatcp.SensorSet,
    time_converter: TimeConverter,
) -> None:
    """Test sending data via the :class:`katgpucbf.fgpu.send.Chunk` interface."""
    # Send descriptors to all the streams
    descriptor_heap = make_descriptor_heap(
        channels_per_substream=N_CHANNELS // N_SUBSTREAMS,
        spectra_per_heap=N_SPECTRA_PER_HEAP,
        sample_bits=sample_bits,
    )
    for send_stream in send_streams:
        for substream in range(N_SUBSTREAMS):
            await send_stream.async_send_heap(descriptor_heap, substream_index=substream)

    rng = np.random.default_rng(seed=1)
    first_timestamp = 0x123456780000
    skip_batches = 3
    for i, chunk in enumerate(chunks):
        _fill_random(chunk.data, rng)
        # Note: this doesn't correspond in any way to the values in data.
        # That isn't necessary for this test.
        _fill_random(chunk.saturated, rng)
        chunk.present[:] = True
        timestamp = first_timestamp + i * SPECTRA_SAMPLES * N_SPECTRA_PER_HEAP * N_BATCHES
        chunk.timestamp = timestamp
        # Check that the property works as expected
        assert chunk.timestamp == timestamp
    # Knock out the first few batches, to test partial transmission
    chunks[0].present[:skip_batches] = False
    data = np.concatenate([chunk.data for chunk in chunks])
    saturated = np.concatenate([chunk.saturated for chunk in chunks])
    saturated = np.sum(saturated[skip_batches:], axis=0, dtype=np.uint64)

    with PromDiff(namespace=METRIC_NAMESPACE) as prom_diff:
        # Send all the chunks, without waiting for the first one to complete
        # transmission (to ensure that the send streams have sufficient
        # capacity).
        futures = [
            asyncio.create_task(chunk.send(send_streams, N_BATCHES, time_converter, sensors, NAME)) for chunk in chunks
        ]
        await asyncio.gather(*futures)

    for queue_list in queues.values():
        for queue in queue_list:
            queue.stop()

    n_channels_per_substream = N_CHANNELS // N_SUBSTREAMS
    heaps_per_iface = {iface: 0 for iface in queues.keys()}
    for i in range(N_SUBSTREAMS):
        seen_batches = np.zeros(N_CHUNKS * N_BATCHES, bool)
        # Only one of the two interfaces should be receiving any data for
        # this substream, but it's simpler to just iterate over both and
        # find the data wherever it happened to fall.
        for iface in queues.keys():
            recv_stream = recv_streams[iface][i]
            ig = spead2.ItemGroup()
            batch = skip_batches
            async for heap in recv_stream:
                updated = ig.update(heap)
                if not updated:
                    continue  # it's a stream control or descriptor heap
                expected_timestamp = first_timestamp + batch * SPECTRA_SAMPLES * N_SPECTRA_PER_HEAP
                assert updated["feng_id"].value == FENG_ID
                assert updated["frequency"].value == i * n_channels_per_substream
                assert updated["timestamp"].value == expected_timestamp
                raw = updated["feng_raw"].value
                raw_complex = raw[..., 0] + 1j * raw[..., 1]
                expected = data[batch, i * n_channels_per_substream : (i + 1) * n_channels_per_substream]
                np.testing.assert_equal(raw_complex, unpack_complex(expected))
                assert not seen_batches[batch]
                seen_batches[batch] = True
                batch += 1
                heaps_per_iface[iface] += 1
        assert np.all(seen_batches[skip_batches:])  # Check that we received all the data we expected
    # Check the load balancing
    good_batches = N_CHUNKS * N_BATCHES - skip_batches
    for iface in queues.keys():
        assert heaps_per_iface[iface] == N_SUBSTREAMS * good_batches // len(queues)

    # Check the sensors and Prometheus metrics
    labels = {"stream": NAME}
    assert prom_diff.get_sample_diff("output_heaps_total", labels) == good_batches * N_SUBSTREAMS
    expected_samples = good_batches * N_SPECTRA_PER_HEAP * N_CHANNELS * N_POLS
    assert prom_diff.get_sample_diff("output_samples_total", labels) == expected_samples
    expected_bytes = expected_samples * COMPLEX * sample_bits // BYTE_BITS
    assert prom_diff.get_sample_diff("output_bytes_total", labels) == expected_bytes
    assert prom_diff.get_sample_diff("output_skipped_heaps_total", labels) == skip_batches * N_SUBSTREAMS
    for pol in range(N_POLS):
        pol_labels = {"stream": NAME, "pol": str(pol)}
        assert prom_diff.get_sample_diff("output_clipped_samples_total", pol_labels) == saturated[pol]
        assert sensors[f"{NAME}.input{pol}.feng-clip-cnt"].value == saturated[pol]
