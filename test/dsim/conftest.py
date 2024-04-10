################################################################################
# Copyright (c) 2021-2022, National Research Foundation (SARAO)
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

"""Common fixtures for dsim tests."""

from collections.abc import Generator, Sequence
from unittest import mock

import numpy as np
import pytest
import spead2.recv.asyncio
import spead2.send.asyncio

from katgpucbf import BYTE_BITS, DEFAULT_TTL, DIG_HEAP_SAMPLES, DIG_SAMPLE_BITS, SPEAD_DESCRIPTOR_INTERVAL_S, spead
from katgpucbf.dsim import descriptors, send
from katgpucbf.send import DescriptorSender

N_POLS = 2
N_ENDPOINTS_PER_POL = 4
SIGNAL_HEAPS = 1024
ADC_SAMPLE_RATE = 1712e6


@pytest.fixture
def inproc_queues() -> Sequence[spead2.InprocQueue]:  # noqa: D401
    """An in-process queue for data per multicast destination."""
    return [spead2.InprocQueue() for _ in range(N_ENDPOINTS_PER_POL * N_POLS)]


@pytest.fixture
def recv_streams(
    inproc_queues: Sequence[spead2.InprocQueue],
) -> Generator[Sequence[spead2.recv.asyncio.Stream], None, None]:
    """Streams that receive data from :func:`inproc_queues`."""
    streams = []
    for queue in inproc_queues:
        stream = spead2.recv.asyncio.Stream(spead2.ThreadPool())
        stream.add_inproc_reader(queue)
        streams.append(stream)
    yield streams
    for stream in streams:
        stream.stop()


@pytest.fixture
def send_stream(inproc_queues: Sequence[spead2.InprocQueue]) -> "spead2.send.asyncio.AsyncStream":
    """Stream that feeds data to the :func:`inproc_queues`."""

    def mock_udp_stream(thread_pool, endpoints, config, **kwargs):
        assert len(endpoints) == len(inproc_queues)
        config.rate = 0  # Just send as fast as possible
        return spead2.send.asyncio.InprocStream(thread_pool, inproc_queues, config)

    with mock.patch("spead2.send.asyncio.UdpStream", side_effect=mock_udp_stream):
        return send.make_stream(
            endpoints=[("invalid", -1) for _ in inproc_queues],
            heap_sets=[],  # Only needed for UdpIbvStream, which we're not using
            n_pols=N_POLS,
            adc_sample_rate=ADC_SAMPLE_RATE,
            heap_samples=DIG_HEAP_SAMPLES,
            sample_bits=DIG_SAMPLE_BITS,
            max_heaps=SIGNAL_HEAPS * N_POLS,
            ttl=DEFAULT_TTL,
            interface_address="",
            ibv=False,
            affinity=-1,
        )


@pytest.fixture
def timestamps() -> np.ndarray:
    """Timestamp array for building heap sets."""
    return np.zeros(SIGNAL_HEAPS, spead.IMMEDIATE_DTYPE)


@pytest.fixture
def heap_sets(timestamps: np.ndarray) -> Sequence[send.HeapSet]:  # noqa: D401
    """Two instances of :class:`~katgpucbf.dsim.send.HeapSet` with random payload bytes."""
    heap_sets = [
        send.HeapSet.create(
            timestamps, [N_ENDPOINTS_PER_POL] * N_POLS, DIG_HEAP_SAMPLES * DIG_SAMPLE_BITS // BYTE_BITS, range(N_POLS)
        )
        for _ in range(2)
    ]
    rng = np.random.default_rng(1)
    for heap_set in heap_sets:
        heap_set.data["payload"][:] = rng.integers(0, 256, size=heap_set.data["payload"].shape, dtype=np.uint8)
    return heap_sets


@pytest.fixture
def sender(
    send_stream: "spead2.send.asyncio.AsyncStream", heap_sets: Sequence[send.HeapSet]
) -> send.Sender:  # noqa: D401
    """A :class:`~katgpucbf.dsim.Sender` using the first of :func:`heaps_sets`."""
    return send.Sender(send_stream, heap_sets[0], DIG_HEAP_SAMPLES)


@pytest.fixture
def descriptor_inproc_queues() -> Sequence[spead2.InprocQueue]:  # noqa: D401
    """An in-process queue for descriptors. Only one queue needed."""
    return [spead2.InprocQueue()]


@pytest.fixture
def descriptor_recv_streams(
    descriptor_inproc_queues: Sequence[spead2.InprocQueue],
) -> Generator[Sequence[spead2.recv.asyncio.Stream], None, None]:
    """Streams that receive data from :func:`descriptor_inproc_queue`."""
    streams = []
    for queue in descriptor_inproc_queues:
        stream = spead2.recv.asyncio.Stream(spead2.ThreadPool())
        stream.add_inproc_reader(queue)
        streams.append(stream)
    yield streams
    for stream in streams:
        stream.stop()


@pytest.fixture
def descriptor_send_stream(descriptor_inproc_queues: Sequence[spead2.InprocQueue]) -> "spead2.send.asyncio.AsyncStream":
    """Stream that feeds data to the :func:`descriptor_inproc_queues`."""

    def mock_udp_stream(thread_pool, endpoints, config, **kwargs):
        return spead2.send.asyncio.InprocStream(thread_pool, descriptor_inproc_queues, config)

    config = descriptors.create_config()
    with mock.patch("spead2.send.asyncio.UdpStream", side_effect=mock_udp_stream):
        return send.make_stream_base(
            endpoints=[("invalid", -1) for _ in descriptor_inproc_queues],
            config=config,
            ttl=4,
            interface_address="",
        )


@pytest.fixture
def descriptor_heap() -> spead2.send.Heap:  # noqa: D401
    """One instances of :class:`~katgpucbf.dsim.descriptors.spead2.send.Heap`."""
    return descriptors.create_descriptors_heap()


@pytest.fixture
def descriptor_sender(
    descriptor_send_stream: "spead2.send.asyncio.AsyncStream", descriptor_heap: spead2.send.Heap
) -> DescriptorSender:  # noqa: D401
    """A :class:`~katgpucbf.dsim.descriptors.DescriptorSender`."""
    return DescriptorSender(
        descriptor_send_stream,
        descriptor_heap,
        SPEAD_DESCRIPTOR_INTERVAL_S,
    )
