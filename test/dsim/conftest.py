################################################################################
# Copyright (c) 2021, National Research Foundation (SARAO)
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

import time
from typing import Generator, Sequence
from unittest import mock

import numpy as np
import pytest
import spead2.recv.asyncio
import spead2.send.asyncio

from katgpucbf import BYTE_BITS, DEFAULT_TTL
from katgpucbf.dsim import send

N_POLS = 2
N_ENDPOINTS_PER_POL = 4
SIGNAL_HEAPS = 1024
HEAP_SAMPLES = 4096
SAMPLE_BITS = 10
ADC_SAMPLE_RATE = 1712e6

MULTICAST_ADDRESS = "239.103.0.64"
PORT = 7148


@pytest.fixture
def inproc_queues() -> Sequence[spead2.InprocQueue]:  # noqa: D401
    """An in-process queue per multicast destination."""
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
            heap_samples=HEAP_SAMPLES,
            sample_bits=SAMPLE_BITS,
            max_heaps=SIGNAL_HEAPS * N_POLS,
            ttl=DEFAULT_TTL,
            interface_address="",
            ibv=False,
            affinity=-1,
        )


@pytest.fixture
def timestamps() -> np.ndarray:
    """Timestamp array for building heap sets."""
    return np.zeros(SIGNAL_HEAPS, ">u8")


@pytest.fixture
def heap_sets(timestamps: np.ndarray) -> Sequence[send.HeapSet]:  # noqa: D401
    """Two instances of :class:`~katgpucbf.dsim.send.HeapSet` with random payload bytes."""
    heap_sets = [
        send.HeapSet.create(
            timestamps, [N_ENDPOINTS_PER_POL] * N_POLS, HEAP_SAMPLES * SAMPLE_BITS // BYTE_BITS, range(N_POLS)
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
    return send.Sender(send_stream, heap_sets[0], 0, HEAP_SAMPLES, time.time(), ADC_SAMPLE_RATE)
