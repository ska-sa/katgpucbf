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
import spead2.recv.asyncio
import spead2.send.asyncio

from katgpucbf import BYTE_BITS, spead
from katgpucbf.dsim import send

from .. import PromDiff
from .conftest import HEAP_SAMPLES, N_ENDPOINTS_PER_POL, N_POLS, SAMPLE_BITS, SIGNAL_HEAPS

pytestmark = [pytest.mark.asyncio]


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
    ig.add_item(spead.TIMESTAMP_ID, "timestamp", "", shape=(), format=spead.IMMEDIATE_FORMAT)
    ig.add_item(spead.DIGITISER_ID_ID, "digitiser_id", "", shape=(), format=spead.IMMEDIATE_FORMAT)
    ig.add_item(spead.DIGITISER_STATUS_ID, "digitiser_status", "", shape=(), format=spead.IMMEDIATE_FORMAT)
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
