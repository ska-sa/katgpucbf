################################################################################
# Copyright (c) 2021-2022, 2024, National Research Foundation (SARAO)
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
import time
from collections.abc import Sequence

import numpy as np
import pytest
import spead2.recv.asyncio
import spead2.send.asyncio
from spead2 import ItemGroup

from katgpucbf import DIG_HEAP_SAMPLES
from katgpucbf.dsim import send
from katgpucbf.send import DescriptorSender
from katgpucbf.utils import TimeConverter

from .. import PromDiff, unpackbits
from .conftest import ADC_SAMPLE_RATE, N_ENDPOINTS_PER_POL, N_POLS, SIGNAL_HEAPS


async def descriptor_recv(
    recv_streams: Sequence[spead2.recv.asyncio.Stream], descriptor_sender: DescriptorSender
) -> ItemGroup:
    """Create receiver for unpacking of spead descriptors.

    Parameters
    ----------
    recv_streams
        Descriptor receiver stream.
    descriptor_sender
        Descriptor sender object. Used to halt the sender method.
    """
    for stream in recv_streams:
        async for heap in stream:
            ig = spead2.ItemGroup()
            ig.update(heap)
            if ig:
                break

    assert ig
    descriptor_sender.halt()  # Halt the descriptor sender as a descriptor set has been received.
    return ig


async def test_sender(
    descriptor_recv_streams: Sequence[spead2.recv.asyncio.Stream],
    descriptor_inproc_queues: Sequence[spead2.InprocQueue],
    descriptor_sender: DescriptorSender,
    recv_streams: Sequence[spead2.recv.asyncio.Stream],
    inproc_queues: Sequence[spead2.InprocQueue],
    sender: send.Sender,
    heap_sets: Sequence[send.HeapSet],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Send random data via a :class:`~katgpucbf.dsim.send.Sender` and check it."""
    # Tweak the sending so that we can interrupt the sender after sending a
    # certain number of heaps
    orig_send_heaps = spead2.send.asyncio.InprocStream.async_send_heaps
    repeats = 5
    remaining_calls = repeats * 2  # Each repetition sends a HeapSet in two halves
    # The last 3 half-heapsets are from after the switch
    switch_heap = SIGNAL_HEAPS * repeats - 3 * (SIGNAL_HEAPS // 2)
    switch_task: asyncio.Future[int] | None = None
    # The copy below fails on xarray >= 2022.9.0 because it tries to deep-copy
    # SharedArray, which doesn't support that. The attribute isn't needed for
    # this test, so just delete it.
    for heap_set in heap_sets:
        del heap_set.data["payload"].attrs["shared_array"]
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

    # Start descriptor sender and wait for descriptors before awaiting for DSim data
    async with asyncio.TaskGroup() as tg:
        tg.create_task(descriptor_sender.run())
        descriptor_recv_streams_task = tg.create_task(descriptor_recv(descriptor_recv_streams, descriptor_sender))
    ig = descriptor_recv_streams_task.result()

    # Note: only do this after dealing with the descriptors, as otherwise
    # they interfere with the countdown.
    monkeypatch.setattr(spead2.send.asyncio.InprocStream, "async_send_heaps", wrapped_send_heaps)

    # Stop the descriptor queue
    for queue in descriptor_inproc_queues:
        queue.stop()

    # Check that the descriptors received make sense.
    assert set(ig.keys()) == {"timestamp", "digitiser_id", "digitiser_status", "adc_samples"}
    assert ig["adc_samples"].format == [("i", 10)]
    assert ig["adc_samples"].shape == (DIG_HEAP_SAMPLES,)

    # Now proceed with DSim data using received descriptors (in ItemGroup)(ig)
    with PromDiff(namespace=send.METRIC_NAMESPACE) as prom_diff:
        await sender.run(0, TimeConverter(time.time(), ADC_SAMPLE_RATE))
        # sender.run can return with some future callbacks (which update
        # Prometheus counters) still scheduled for the next event loop
        # iteration. Ensure they get a chance to run before exiting PromDiff.
        await asyncio.sleep(0)
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
            expected_unpacked = unpackbits(expected.to_numpy())
            np.testing.assert_equal(updated["adc_samples"].value, expected_unpacked)
        assert i == SIGNAL_HEAPS * repeats  # Check that all the data arrived
    assert switch_task is not None
    assert (await switch_task) == switch_heap * DIG_HEAP_SAMPLES

    # Check the Prometheus counters
    assert prom_diff.diff("output_heaps_total") == SIGNAL_HEAPS * repeats * N_POLS
    assert prom_diff.diff("output_bytes_total") == orig_payload[0].nbytes * repeats
