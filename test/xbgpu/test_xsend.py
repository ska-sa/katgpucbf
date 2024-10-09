################################################################################
# Copyright (c) 2020-2024, National Research Foundation (SARAO)
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

"""Unit tests for the :mod:`katgpucbf.xbgpu.xsend` module."""

from typing import Final

import numpy as np
import pytest
import spead2
import spead2.recv.asyncio
from katsdpsigproc.abc import AbstractContext

from katgpucbf import COMPLEX
from katgpucbf.spead import FREQUENCY_ID, TIMESTAMP_ID, XENG_RAW_ID
from katgpucbf.xbgpu.xsend import XSend

from . import test_parameters

TOTAL_HEAPS: Final[int] = 20
TIMESTAMP_SCALE: Final[int] = 0x1000000


class TestXSend:
    """Test :class:`katgpucbf.xbgpu.xsend.XSend`."""

    @staticmethod
    async def _send_data(send_stream: XSend) -> None:
        """Send a fixed number of heaps."""
        # Send the descriptors as the recv_stream object needs it to
        # interpret the received heaps correctly.
        await send_stream.stream.async_send_heap(send_stream.descriptor_heap)

        for i in range(TOTAL_HEAPS):
            # Get a free heap - there is not always a free one available. This
            # function yields until one it available.
            heap = await send_stream.get_free_heap()

            # Populate the buffer with dummy data.
            heap.buffer.fill(i)

            # Give the heap back to the send_stream to transmit out
            # onto the network. The timestamp is multiplied by TIMESTAMP_SCALE
            # so that its value is different from the values in the
            # buffer array.
            heap.timestamp = i * TIMESTAMP_SCALE
            send_stream.send_heap(heap)
        # send_heap just queues data for sending but is non-blocking.
        # Flush to ensure that the data all gets sent before we return.
        await send_stream.stream.async_flush()

    @staticmethod
    async def _recv_data(
        recv_stream: spead2.recv.asyncio.Stream,
        n_engines: int,
        n_channels_per_substream: int,
        n_baselines: int,
        first_heap: int,
    ) -> None:
        """Receive data transmitted from :func:`_send_data`.

        Error-check data here as well.
        """
        ig = spead2.ItemGroup()

        # Wait for the first packet to arrive - it is expected to be the
        # SPEAD descriptor.
        heap = await recv_stream.get()
        assert heap.cnt % n_engines == 4 % n_engines, "The heap IDs are not correctly strided"
        items = ig.update(heap)
        assert items == {}, "This heap contains item values not just the expected descriptors."

        # Check the data heaps
        for i in range(first_heap, TOTAL_HEAPS):
            heap = await recv_stream.get()
            items = ig.update(heap)
            assert set(items.keys()) == {"timestamp", "frequency", "xeng_raw"}
            assert items["timestamp"].id == TIMESTAMP_ID
            assert items["timestamp"].value == i * TIMESTAMP_SCALE
            assert items["frequency"].id == FREQUENCY_ID
            assert items["frequency"].value == n_channels_per_substream * 4
            assert items["xeng_raw"].id == XENG_RAW_ID
            assert items["xeng_raw"].value.shape == (n_channels_per_substream, n_baselines, COMPLEX)
            assert items["xeng_raw"].value.dtype == np.int32
            np.testing.assert_equal(items["xeng_raw"].value, i)

    @pytest.mark.combinations(
        "n_ants, n_channels",
        test_parameters.array_size,
        test_parameters.n_channels,
    )
    async def test_send_simple(self, context: AbstractContext, n_ants: int, n_channels: int) -> None:
        """
        Test :class:`katgpucbf.xbgpu.xsend.XSend`.

        This test transmits a number of heaps from a XSend object over a spead2
        inproc transport. The received heaps are then checked.

        This test does not generate random data as it will take much more compute
        to check that the random data is received correctly. I do not think that
        random data is necessary, as that would be checking that SPEAD2 is
        assembling heaps correctly which is a function of SPEAD2 not the XSend
        class. Mangled heaps should be picked up in the SPEAD2 unit tests.

        .. note::

            This test only tests at the level of heaps, and does not verify anything
            about the layout of individual packets, which is left up to spead2. In
            particular, it does not verify that

            - packets have the requested payload size;
            - every packet has a copy of the immediate items.

        Parameters
        ----------
        context
            Device context for allocating buffers.
        n_ants
            The number of antennas that have been correlated.
        n_channels
            The number of frequency channels out of the FFT. NB: This is not the
            number of FFT channels per stream. The number of channels per stream is
            calculated from this value.
        """
        # Get a realistic number of engines: round n_ants*4 up to the next power of 2.
        n_engines = 1
        while n_engines < n_ants * 4:
            n_engines *= 2
        n_channels_per_substream = n_channels // n_engines
        n_baselines = (n_ants + 1) * (n_ants) * 2

        queue = spead2.InprocQueue()
        send_stream = XSend(
            output_name="test",
            n_ants=n_ants,
            n_channels=n_channels,
            n_channels_per_substream=n_channels_per_substream,
            dump_interval_s=0.0,  # Just send as fast as possible to speed up the test
            send_rate_factor=0,
            channel_offset=n_channels_per_substream * 4,  # Arbitrary for now
            context=context,
            stream_factory=lambda stream_config, buffers: spead2.send.asyncio.InprocStream(
                spead2.ThreadPool(), [queue], stream_config
            ),
            send_enabled=True,
        )
        send_stream.send_enabled_timestamp = int(TIMESTAMP_SCALE * 1.5)
        await self._send_data(send_stream)
        # Stop the queue, to ensure that if recv_data tries to read more heaps
        # than were sent, it will error out rather than hanging.
        queue.stop()

        recv_stream = spead2.recv.asyncio.Stream(spead2.ThreadPool(), spead2.recv.StreamConfig())
        recv_stream.add_inproc_reader(queue)
        await self._recv_data(recv_stream, n_engines, n_channels_per_substream, n_baselines, 2)
