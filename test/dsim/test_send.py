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

"""Test data transmission."""

import asyncio
import itertools
from typing import Generator, Optional, Sequence
from unittest import mock

import numpy as np
import pytest
import spead2.recv.asyncio
import spead2.send.asyncio

from katgpucbf import BYTE_BITS, DEFAULT_TTL, spead
from katgpucbf.dsim import send

from .. import PromDiff

N_POLS = 2
N_ENDPOINTS_PER_POL = 4
SIGNAL_HEAPS = 1024
HEAP_SAMPLES = 4096
SAMPLE_BITS = 10

pytestmark = [pytest.mark.asyncio]


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
            adc_sample_rate=1712e6,
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
    return send.Sender(send_stream, heap_sets[0], 0, HEAP_SAMPLES)


async def test_sender(
    recv_streams: Sequence[spead2.recv.asyncio.Stream],
    inproc_queues: Sequence[spead2.InprocQueue],
    sender: send.Sender,
    heap_sets: Sequence[send.HeapSet],
    mocker,
) -> None:
    """Send random data via a :class:`~katgpucbf.dsim.send.Sender` and check it."""
    # Tweak the sending so that we can interrupt the sender after sending a
    # certain number of heaps
    orig_send_heaps = spead2.send.asyncio.InprocStream.async_send_heaps
    repeats = 5
    remaining_calls = repeats * 2  # Each repetition sends a HeapSet in two halves
    # The last 3 half-heapsets are from after the switch
    switch_heap = SIGNAL_HEAPS * repeats - 3 * (SIGNAL_HEAPS // 2)
    switch_task: Optional[asyncio.Future[int]] = None
    orig_payload = [heap_set.data["payload"].copy() for heap_set in heap_sets]

    async def switch_heap_sets() -> int:
        timestamp = await sender.set_heaps(heap_sets[1])
        # The above is supposed to wait until the original heap set is no
        # longer in use. Fill it with zeros to verify.
        heap_sets[0].data["payload"].data.fill(0)
        return timestamp

    def wrapped_send_heaps(*args, **kwargs):
        nonlocal remaining_calls, switch_task
        ret = orig_send_heaps(*args, **kwargs)
        remaining_calls -= 1
        if remaining_calls == 3:
            switch_task = asyncio.create_task(switch_heap_sets())
        if remaining_calls == 0:
            sender.halt()
        return ret

    mocker.patch.object(
        spead2.send.asyncio.InprocStream, "async_send_heaps", side_effect=wrapped_send_heaps, autospec=True
    )

    with PromDiff(namespace=send.METRIC_NAMESPACE) as prom_diff:
        await sender.run()
    for queue in inproc_queues:
        queue.stop()

    # We don't have descriptors (yet), so we have to set the items manually
    ig = spead2.ItemGroup()
    immediate_format = [("u", spead.FLAVOUR.heap_address_bits)]
    ig.add_item(spead.TIMESTAMP_ID, "timestamp", "", shape=(), format=immediate_format)
    ig.add_item(spead.DIGITISER_ID_ID, "digitiser_id", "", shape=(), format=immediate_format)
    ig.add_item(spead.DIGITISER_STATUS_ID, "digitiser_status", "", shape=(), format=immediate_format)
    # Just treat it as raw bytes so that we can directly compare to the packed data
    ig.add_item(spead.RAW_DATA_ID, "raw_data", "", dtype=np.uint8, shape=(HEAP_SAMPLES * SAMPLE_BITS // BYTE_BITS,))

    for pol in range(N_POLS):
        pol_streams = recv_streams[N_ENDPOINTS_PER_POL * pol : N_ENDPOINTS_PER_POL * (pol + 1)]
        for i, stream in enumerate(itertools.cycle(pol_streams)):
            try:
                heap = await stream.get()
                updated = ig.update(heap)
            except spead2.Stopped:
                break
            assert updated["timestamp"].value == i * HEAP_SAMPLES
            assert updated["digitiser_id"].value == pol
            assert updated["digitiser_status"].value == 0
            side = int(i >= switch_heap)
            expected = orig_payload[side].isel(time=i % SIGNAL_HEAPS, pol=pol)
            np.testing.assert_equal(updated["raw_data"].value, expected)
        assert i == SIGNAL_HEAPS * repeats  # Check that all the data arrived
    assert switch_task is not None
    assert (await switch_task) == switch_heap * HEAP_SAMPLES

    # Check the Prometheus counters
    assert prom_diff.get_sample_diff("output_heaps_total") == SIGNAL_HEAPS * repeats * N_POLS
    assert prom_diff.get_sample_diff("output_bytes_total") == orig_payload[0].nbytes * repeats
