################################################################################
# Copyright (c) 2020-2021, National Research Foundation (SARAO)
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

"""
Module for performing unit tests on the XSend class in the xsend.py module.

Testing network code is difficult to do on a single thread. SPEAD2 has the
concept of transports. A transport generally receives data from a network.
SPEAD2 provides two other transports that can receive simulated network data -
these can be used for testing the receiver in once process. The two transports
are an inproc and a buffer transport. The inproc transport is more flexible and
as such is used here.

NOTE: A downside of this test is that it does not check that the packet formats
are exactly correct. This test will ensure that the packets are transmitted in
a way that they are able to be assembled into a heap by any SPEAD2 receiver or
a full implementation of the SPEAD protocol. However, the exact packet size and
the presence of repeat pointers within the a packet are not checked. Some sort
of external test should be done to check this. See the
display_xengine_multicast_packets.py script in the scratch folder of this repo
as a starting point to check packet formats. UPDATE: If we wanted to actually
process real network packets, we could use the SPEAD2 PCAP transport. This
allows SPEAD2 to read a PCAP file. We could generate network data in a manner
that duplicates real traffic and then store that in a PCAP file to be used in
unit tests. This data could be interleaved and out of order if necessary for a
more robust test. The downside to a PCAP test is that the PCAP file could get
quite large quite quickly. This could lead to the repo getting very large which
is to be avoided. I think the best use a single PCAP file of a single array
configuration and run a single unit test and then use the inproc readers as
done below for the remaining configurations.  This will ensure that we run a
test on "real" data while still testing all configurations without the repo
growing too large.

TODO: Implement a pcap test as described in the above note.
TODO: Review the xsend.py class to see if some functionality has not been
covered in all of these tests.
"""

from asyncio import create_task, gather
from typing import Final

import numpy as np
import pytest
import spead2
import spead2.recv.asyncio

from katgpucbf import COMPLEX
from katgpucbf.spead import FREQUENCY_ID, TIMESTAMP_ID, XENG_RAW_ID
from katgpucbf.xbgpu.xsend import XSend

from . import test_parameters

TOTAL_HEAPS: Final[int] = 20
DUMP_INTERVAL_S: Final[int] = 0
SEND_RATE_FACTOR: Final[float] = 1.1
SAMPLE_BITWIDTH: Final[int] = 32


class TestSend:
    """Test the spead2 send stream in :class:`xbgpu.xsend.XSend`."""

    @staticmethod
    async def _send_data(send_stream: XSend) -> None:
        """Send a fixed number of heaps."""
        num_sent = 0

        # 4.1 Send the descriptor as the recv_stream object needs it to
        # interpret the received heaps correctly.
        await send_stream.send_descriptor_heap()

        # 4.2 Run until a set number of heaps have been transferred.
        while num_sent < TOTAL_HEAPS:
            # 4.2.1 Get a free buffer to store the next heap - there is not
            # always a free buffer available. This function yields until one it
            # available.
            buffer_wrapper = await send_stream.get_free_heap()

            # 4.2.2 Populate the buffer with dummy data - notice how we copy
            # new values into the buffer, we dont overwrite the buffer.
            # Attempts to overwrite the buffer will throw an error. This is
            # intended behavour as the memory regions in the buffer have been
            # configured for zero-copy sends.
            # The [:] syntax forces the data to be copied to the buffer.
            # Without this, the buffer_wrapper variable would point to the new
            # created numpy array which is located in new location in memory
            # and the old buffer would be garbage collected.
            buffer_wrapper.buffer[:] = np.full(buffer_wrapper.buffer.shape, num_sent, np.uint8)

            # 4.2.3 Give the buffer back to the send_stream to transmit out
            # onto the network. The timestamp is multiplied by 0x1000000 so
            # that its value is different from the values in the buffer_wrapper
            # array.
            send_stream.send_heap(num_sent * 0x1000000, buffer_wrapper)
            num_sent += 1

    @staticmethod
    async def _recv_data(
        recv_stream: spead2.recv.asyncio.Stream,
        n_engines: int,
        n_channels_per_stream: int,
        n_baselines: int,
    ) -> None:
        """Receive data transmitted from :func:`_send_data`.

        Error-check data here as well.

        .. todo::
            Remove error-checking, to be placed elsewhere in the unit test, and
            Consolidate _send and _recv.
        """
        num_received = 0
        ig = spead2.ItemGroup()

        # 5.1 Wait for the first packet to arrive - it is expected to be the
        # SPEAD descriptor. Without the descriptor the recv_stream cannot
        # interpret the heaps correctly.
        heap = await recv_stream.get()
        assert heap.cnt % n_engines == 4 % n_engines, "The heap IDs are not correctly strided"
        items = ig.update(heap)
        assert len(list(items.values())) == 0, "This heap contains item values not just the expected descriptors."

        # 5.2 Wait for the rest of the heaps to arrive and then end the function.
        while num_received < TOTAL_HEAPS:
            # 5.2.1 The next heap may not be available immediatly. This
            # function waits asynchronously until a
            # heap arrives.
            heap = await recv_stream.get()

            items = ig.update(heap)
            has_timestamp = False
            has_channel_offset = False
            has_xeng_raw = False

            for item in items.values():
                # 5.2.2 Check that the received heap has a timestamp item with
                # the correct expected value.
                if item.id == 0x1600:
                    has_timestamp = True
                    assert (
                        item.value == num_received * 0x1000000
                    ), f"Timestamp value incorrect. Expected: {hex(num_received)}, actual {hex(item.value)}"

                # 5.2.2 Check that the received heap has a channel offset item
                # with the correct expected value.
                if item.id == 0x4103:
                    has_channel_offset = True
                    assert item.value == n_channels_per_stream * 4, (
                        "Channel offset incorrect. "
                        f"Expected: {hex(n_channels_per_stream * 4)}, actual: {hex(item.value)}"
                    )

                # 5.2.2 Check that the received heap has an xeng_raw data
                # buffer item. Check that the buffer is the correct size and
                # that the values are all the expected value.
                if item.id == 0x1800:
                    has_xeng_raw = True
                    data_length_bytes = n_baselines * n_channels_per_stream * COMPLEX * SAMPLE_BITWIDTH // 8
                    assert item.value.nbytes == data_length_bytes, (
                        "xeng_raw data not correct size. "
                        f"Expected: {data_length_bytes} bytes, actual: {item.value.size} bytes."
                    )
                    assert (
                        item.value.dtype == np.int32
                    ), f"xeng_raw dtype is {(item.value.dtype)}, dtype of uint64 expected."
                    assert np.all(item.value == num_received)

            assert has_timestamp, f"Received heap is missing timestamp item with ID {hex(TIMESTAMP_ID)}"
            assert has_channel_offset, f"Received heap is missing channel offset item with ID {hex(FREQUENCY_ID)}"
            assert has_xeng_raw, f"Received heap is missing xeng_raw data buffer item with ID {hex(XENG_RAW_ID)}"
            num_received += 1

    @pytest.mark.combinations(
        "num_ants, num_channels",
        test_parameters.array_size,
        test_parameters.num_channels,
    )
    async def test_send_simple(self, context, num_ants, num_channels) -> None:
        """
        Tests the XSend class in the xsend.py module.

        This test transmits a number of heaps from a XSend object over a SPEAD2
        inproc transport. A SPEAD2 receiver object then examines the transmitted
        heaps to ensure that the correct fields are transmitted with the correct
        values.

        This test does not generate random data as it will take much more compute
        to check that the random data is received correctly. I do not think that
        random data is necessary, as that would be checking that SPEAD2 is
        assembling heaps correctly which is a function of SPEAD2 not the XSend
        class. Mangled heaps should be picked up in the SPEAD2 unit tests.

        Parameters
        ----------
        num_ants: int
            The number of antennas that have been correlated.
        num_channels: int
            The number of frequency channels out of the FFT. NB: This is not the
            number of FFT channels per stream. The number of channels per stream is
            calculated from this value.
        """
        # Get a realistic number of engines: round n_ants*4 up to the next power of 2.
        n_engines = 1
        while n_engines < num_ants * 4:
            n_engines *= 2
        n_channels_per_stream = num_channels // n_engines
        n_baselines = (num_ants + 1) * (num_ants) * 2

        # 3. Initialise SPEAD2 sender and receiver objects and link them
        # together.
        # 3.1 Create the queue that will link the sender and receiver together.
        queue = spead2.InprocQueue()

        send_stream = XSend(
            n_ants=num_ants,
            n_channels=num_channels,
            n_channels_per_stream=n_channels_per_stream,
            dump_interval_s=DUMP_INTERVAL_S,
            send_rate_factor=SEND_RATE_FACTOR,
            channel_offset=n_channels_per_stream * 4,  # Arbitrary for now
            context=context,
            stream_factory=lambda stream_config, buffers: spead2.send.asyncio.InprocStream(
                spead2.ThreadPool(), [queue], stream_config
            ),
            tx_enabled=True,
        )

        # 3.3 Create a generic SPEAD2 receiver that will receive heaps from the
        # XSend over the queue.
        thread_pool = spead2.ThreadPool()
        recv_stream = spead2.recv.asyncio.Stream(thread_pool, spead2.recv.StreamConfig(max_heaps=100))
        recv_stream.add_inproc_reader(queue)

        # TODO: These need to be consolidated into one task
        await gather(
            create_task(self._send_data(send_stream)),
            create_task(self._recv_data(recv_stream, n_engines, n_channels_per_stream, n_baselines)),
        )
