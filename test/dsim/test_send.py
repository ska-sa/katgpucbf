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
from typing import Optional, Sequence

import numba
import numpy as np
import pytest
import spead2.recv.asyncio
import spead2.send.asyncio
from spead2 import ItemGroup

from katgpucbf import DIG_HEAP_SAMPLES
from katgpucbf.dsim import send
from katgpucbf.dsim.descriptors import DescriptorSender

from .. import PromDiff
from .conftest import N_ENDPOINTS_PER_POL, N_POLS, SIGNAL_HEAPS

pytestmark = [pytest.mark.asyncio]


@numba.njit
def unpackbits(packed_data: np.ndarray) -> np.ndarray:
    """Unpack 8b data words to 10b data words.

    Parameters
    ----------
    packed_data
        A numpy ndarray of packed 8b data words.
    """
    unpacked_data = np.zeros((DIG_HEAP_SAMPLES,), dtype=np.int16)
    data_sample = np.int16(0)
    pack_idx = 0
    unpack_idx = 0

    for _ in range(len(packed_data) // 5):
        tmp_40b_word = np.uint64(
            packed_data[pack_idx] << (8 * 4)
            | packed_data[pack_idx + 1] << (8 * 3)
            | packed_data[pack_idx + 2] << (8 * 2)
            | packed_data[pack_idx + 3] << 8
            | packed_data[pack_idx + 4]
        )
        for data_idx in range(4):
            data_sample = np.int16((tmp_40b_word & np.uint64(1098437885952)) >> np.uint64(30))
            if data_sample > 511:
                data_sample = data_sample - 1024
            unpacked_data[unpack_idx + data_idx] = np.int16(data_sample)
            tmp_40b_word = tmp_40b_word << np.uint8(10)
        unpack_idx += 4
        pack_idx += 5
    return unpacked_data


async def descriptor_recv(rec_streams, descriptor_sender) -> ItemGroup:
    """Create receiver for unpacking of spead descriptors.

    Parameters
    ----------
    rec_streams
        Descriptor receiver stream.
    descriptor_sender
        Descriptor sender object. Used to halt the sender method.
    """
    for stream in rec_streams:
        async for heap in stream:
            ig = spead2.ItemGroup()
            for raw_descriptor in heap.get_descriptors():
                descriptor = spead2.Descriptor.from_raw(raw_descriptor, heap.flavour)
                ig.add_item(
                    id=descriptor.id,
                    name=descriptor.name,
                    dtype=descriptor.dtype,
                    description=descriptor.description,
                    shape=descriptor.shape,
                    format=descriptor.format,
                )
            if ig:
                break
    if ig:
        # Halt the descriptor sender as a descriptor set has been received.
        descriptor_sender.halt()
    return ig


async def test_sender(
    descriptor_recv_streams: spead2.recv.asyncio.Stream,
    descriptor_inproc_queues: Sequence[spead2.InprocQueue],
    descriptor_sender: DescriptorSender,
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

    # Start descriptor sender and wait for descriptors before awaiting for DSim data
    descriptor_sender_task = asyncio.create_task(descriptor_sender.run())
    descriptor_recv_streams_task = asyncio.create_task(descriptor_recv(descriptor_recv_streams, descriptor_sender))
    _, ig = await asyncio.gather(descriptor_sender_task, descriptor_recv_streams_task)

    # Stop the descriptor queue
    for queue in descriptor_inproc_queues:
        queue.stop()

    # Check that the descriptors received make sense.
    assert set(ig.keys()) == {"timestamp", "digitiser_id", "digitiser_status", "adc_samples"}

    # Now proceed with DSim data using received descriptors (in ItemGroup)(ig)
    with PromDiff(namespace=send.METRIC_NAMESPACE) as prom_diff:
        await sender.run()
    for queue in inproc_queues:
        queue.stop()

    for pol in range(N_POLS):
        pol_streams = recv_streams[N_ENDPOINTS_PER_POL * pol : N_ENDPOINTS_PER_POL * (pol + 1)]
        for i, stream in enumerate(itertools.cycle(pol_streams)):
            try:
                heap = await stream.get()
                updated = ig.update(heap)
            except spead2.Stopped:
                break
            assert updated["timestamp"].value == i * DIG_HEAP_SAMPLES
            assert updated["digitiser_id"].value == pol
            assert updated["digitiser_status"].value == 0
            side = int(i >= switch_heap)
            expected = orig_payload[side].isel(time=i % SIGNAL_HEAPS, pol=pol)
            expected = unpackbits(expected.to_numpy())
            np.testing.assert_equal(updated["adc_samples"].value, expected)
        assert i == SIGNAL_HEAPS * repeats  # Check that all the data arrived
    assert switch_task is not None
    assert (await switch_task) == switch_heap * DIG_HEAP_SAMPLES

    # Check the Prometheus counters
    assert prom_diff.get_sample_diff("output_heaps_total") == SIGNAL_HEAPS * repeats * N_POLS
    assert prom_diff.get_sample_diff("output_bytes_total") == orig_payload[0].nbytes * repeats
