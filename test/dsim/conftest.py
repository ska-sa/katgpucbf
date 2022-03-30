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
from typing import Generator, Sequence, List, Tuple
from unittest import mock
import logging
from venv import create
import netifaces
import itertools

import numpy as np
import pytest
import spead2
import spead2.recv
import spead2.recv.asyncio
import spead2.send.asyncio

from katgpucbf import BYTE_BITS, DEFAULT_TTL
from katgpucbf.dsim import send
from katgpucbf.dsim import descriptors
from katgpucbf.dsim.descriptors import create_descriptor_stream, create_config, create_descriptors_heap
import asyncio

N_POLS = 2
N_ENDPOINTS_PER_POL = 4
SIGNAL_HEAPS = 1024
HEAP_SAMPLES = 4096
SAMPLE_BITS = 10
ADC_SAMPLE_RATE = 1712e6

MULTICAST_ADDRESS = "239.103.0.64"
PORT = 7148
interface_address = netifaces.ifaddresses('lo')[netifaces.AF_INET][0]["addr"]


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


# ----------------------------------------------------------------------------
@pytest.fixture
def descriptor_inproc_queue() -> Sequence[spead2.InprocQueue]:  # noqa: D401
    """An in-process queue for DSim descriptors."""
    # return [spead2.InprocQueue() for _ in range(N_ENDPOINTS_PER_POL * N_POLS)]
    return [spead2.InprocQueue()]

@pytest.fixture   
def descriptor_recv_streams(
    descriptor_inproc_queue: Sequence[spead2.InprocQueue]
) -> Generator[Sequence[spead2.recv.Stream], None, None]:
    """Streams that receive data from :func:`inproc_queues`."""
    streams = []
    for queue in descriptor_inproc_queue:
        stream = spead2.recv.asyncio.Stream(spead2.ThreadPool())
        stream.add_inproc_reader(queue)
        streams.append(stream)
    return streams
    # yield streams
    # for stream in streams:
    #     stream.stop()

@pytest.fixture
def descriptor_send_stream(descriptor_inproc_queue: Sequence[spead2.InprocQueue]) -> "spead2.send.asyncio.AsyncStream":
    """Stream that feeds data to the :func:`inproc_queues`."""

    # config = create_config()
    # return create_descriptor_stream(
    #     endpoints=[("invalid", -1) for _ in descriptor_inproc_queue],
    #     config=config,
    #     ttl=4,
    #     interface_address="",    
    #     queues= descriptor_inproc_queue, 
    # )

    def mock_udp_stream(thread_pool, endpoints, config, **kwargs):
        # assert len(endpoints) == len(descriptor_inproc_queue)
        # config.rate = 0  # Just send as fast as possible
        return spead2.send.asyncio.InprocStream(thread_pool, descriptor_inproc_queue, config)

    config = create_config()
    with mock.patch("spead2.send.asyncio.UdpStream", side_effect=mock_udp_stream):
        return create_descriptor_stream(
            endpoints=[("invalid", -1) for _ in descriptor_inproc_queue],
            config=config,
            ttl=4,
            interface_address="",      
        )

@pytest.fixture
def descriptor_heap() -> descriptors:  # noqa: D401
    """One instances of :class:`~katgpucbf.dsim.descriptors.heap_to_send`."""    
    return create_descriptors_heap(HEAP_SAMPLES)
    # return descriptor_generator.heap_to_send

@pytest.fixture
def descriptor_sender(
    descriptor_send_stream: "spead2.send.asyncio.AsyncStream", descriptor_heap: descriptors
) -> send.Sender:  # noqa: D401
    """A :class:`~katgpucbf.dsim.Sender` using the first of :func:`heaps_sets`."""
    timestamp = 0
    return descriptors.DescriptorSender(descriptor_send_stream, timestamp, descriptor_heap)
    # return descriptors.DescriptorSender(descriptor_send_stream, 'lo', HEAP_SAMPLES, DEFAULT_TTL, timestamp, descriptor_heap)
    # return send.Sender(send_stream, heap_sets[0], 0, HEAP_SAMPLES, time.time(), ADC_SAMPLE_RATE)



# # # TO BE REMOVED
# if __name__ == "__main__":

#     async def dgb_rx(recv_stream):
#         print('rx')
#         await asyncio.sleep(0)
#         ig = spead2.ItemGroup()
#         for stream in recv_stream:
#             for i, stream in enumerate(itertools.cycle(stream)):
#                 try:
#                     for heap in stream:
#                         items = ig.update(heap)
#                         # heap = await stream.get()
#                         # updated = ig.update(heap)
#                 except spead2.Stopped:
#                     break

#     async def main():
#         heap = descriptor_heap()
#         ig = spead2.ItemGroup()
#         items = ig.update(heap)
#         for item in items.values():
#             print(heap.cnt, item.name, item.value)
#             if item.name == "ADC Samples":
#                 print('Captured Data')


#         # for raw_descriptor in heap.get_descriptors():
#         #     descriptor = spead2.Descriptor.from_raw(raw_descriptor, heap.flavour)


#         # recv = descriptor_recv_streams(descriptor_inproc_queue())
#         ds = descriptor_sender(descriptor_send_stream(descriptor_inproc_queue()), descriptor_heap())
#         task = asyncio.create_task(ds.run())
#         task_rx = asyncio.create_task(dgb_rx(descriptor_recv_streams(descriptor_inproc_queue())))
#         await asyncio.gather(task, task_rx)

#     asyncio.run(main())
