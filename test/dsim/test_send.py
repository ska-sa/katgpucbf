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

import numpy as np
import pytest
from pytest_mock import mocker
import spead2.recv.asyncio
import spead2.send.asyncio
import logging
logging.basicConfig()

from katgpucbf import BYTE_BITS, spead
from katgpucbf.dsim import send
from katgpucbf.dsim import descriptors

from .. import PromDiff
from .conftest import HEAP_SAMPLES, N_ENDPOINTS_PER_POL, N_POLS, SAMPLE_BITS, SIGNAL_HEAPS

# from conftest import HEAP_SAMPLES, N_ENDPOINTS_PER_POL, N_POLS, SAMPLE_BITS, SIGNAL_HEAPS, recv_streams
# from conftest import descriptor_heap, descriptor_sender, descriptor_send_stream, descriptor_recv_streams, descriptor_inproc_queue
# from conftest import inproc_queues, sender, send_stream, heap_sets, timestamps

pytestmark = [pytest.mark.asyncio]

# def descriptor_unpack(rec_streams):
#     ig = spead2.ItemGroup(rec_streams)
#     try:
#         for s in rec_streams:
#             for heap in s:
#                 for raw_descriptor in heap.get_descriptors():
#                     descriptor = spead2.Descriptor.from_raw(raw_descriptor, heap.flavour)
#                     print('''\
#                         Descriptor for {0.name} ({0.id:#x})
#                         description: {0.description}
#                         format:      {0.format}
#                         dtype:       {0.dtype}
#                         shape:       {0.shape}'''.format(descriptor))

#     except asyncio.TimeoutError:
#         pass

async def descriptor_recv(rec_streams, descriptor_sender):
    while True:
        await asyncio.sleep(0)      
        for s in rec_streams:
            async for heap in s:
                ig = spead2.ItemGroup()

                for raw_descriptor in heap.get_descriptors():
                    descriptor = spead2.Descriptor.from_raw(raw_descriptor, heap.flavour)
                    print('''\
                        Descriptor for {0.name} ({0.id:#x})
                        description: {0.description}
                        format:      {0.format}
                        dtype:       {0.dtype}
                        shape:       {0.shape}'''.format(descriptor))
                    
                    ig.add_item(id=descriptor.id,
                                    name=descriptor.name, 
                                    dtype=descriptor.dtype,
                                    description=descriptor.description,
                                    shape=descriptor.shape,
                                    format= descriptor.format)
                if ig:
                    break
        if ig:
            descriptor_sender.halt()
            break
    return ig

# async def test_sender_dbg(
#     descriptor_recv_streams: spead2.recv.asyncio.Stream,
#     descriptor_inproc_queue: Sequence[spead2.InprocQueue],
#     descriptor_sender,
#     descriptor_heap: descriptors,
# ) -> None:
#     """Send random data via a :class:`~katgpucbf.dsim.send.Sender` and check it."""

#     # Start descriptor sender and wait for descriptors before awaiting for DSim data
#     await descriptor_sender.run()

#     # descriptor_task = asyncio.create_task(descriptor_sender.run())
#     # other_task = asyncio.create_task(other(descriptor_recv_streams))
#     # await asyncio.gather(descriptor_task, other_task)
    

#     # stop the descriptor sender and queue
#     descriptor_sender.halt()
#     for queue in descriptor_inproc_queue:
#         queue.stop()

#     descriptors = await descriptor_recv(descriptor_recv_streams)


async def test_sender(
    descriptor_recv_streams: spead2.recv.asyncio.Stream,
    descriptor_inproc_queue: Sequence[spead2.InprocQueue],
    descriptor_sender,
    # descriptor_heap: descriptors,
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
    desc_ig = await asyncio.gather(descriptor_sender_task, descriptor_recv_streams_task)
    ig = desc_ig[1]

    # stop the descriptor sender and queue
    # ig_ref = spead2.ItemGroup()
    # for raw_descriptor in descriptor_heap.get_descriptors():
    #     descriptor = spead2.Descriptor.from_raw(raw_descriptor, heap.flavour)
    

    for queue in descriptor_inproc_queue:
        queue.stop()

    # Check that the descriptors received make sense.
    assert ig['timestamp'].name == "timestamp"
    assert ig['digitiser_id'].name == "digitiser_id"
    assert ig['digitiser_status'].name == "digitiser_status"
    assert ig['raw_data'].name == "raw_data"

    # TODO:   fix extra entry in ig
    # TODO:   clean up code
    # TODO:   Add descriptor assert

    with PromDiff(namespace=send.METRIC_NAMESPACE) as prom_diff:
        await sender.run()
    for queue in inproc_queues:
        queue.stop()

    # We don't have descriptors (yet), so we have to set the items manually
    # ig = spead2.ItemGroup()
    # immediate_format = [("u", spead.FLAVOUR.heap_address_bits)]
    # ig.add_item(spead.TIMESTAMP_ID, "timestamp", "", shape=(), format=immediate_format)
    # ig.add_item(spead.DIGITISER_ID_ID, "digitiser_id", "", shape=(), format=immediate_format)
    # ig.add_item(spead.DIGITISER_STATUS_ID, "digitiser_status", "", shape=(), format=immediate_format)
    # # Just treat it as raw bytes so that we can directly compare to the packed data
    # ig.add_item(spead.RAW_DATA_ID, "raw_data", "", dtype=np.uint8, shape=(HEAP_SAMPLES * SAMPLE_BITS // BYTE_BITS,))

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

# # TO BE REMOVED
if __name__ == "__main__":
    async def main():
        # Qs = [spead2.InprocQueue()]
        Qs = descriptor_inproc_queue()
        Qr = inproc_queues()

        await test_sender(descriptor_recv_streams(Qs), 
        Qs, 
        descriptor_sender(descriptor_send_stream(Qs), descriptor_heap()), 
        descriptor_heap(),
        recv_streams(Qr),
        Qr,
        sender(send_stream(Qr),heap_sets(timestamps())),
        heap_sets(timestamps()),
        mocker,
        )

        await test_sender_dbg(descriptor_recv_streams(Qs), Qs, descriptor_sender(descriptor_send_stream(Qs), 
        descriptor_heap()), descriptor_heap())

    asyncio.run(main())