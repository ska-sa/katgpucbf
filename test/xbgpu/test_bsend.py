################################################################################
# Copyright (c) 2023, National Research Foundation (SARAO)
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

"""Unit tests for the :mod:`katgpucbf.xbgpu.bsend` module."""

from typing import Final

import numpy as np
import pytest
import spead2
import spead2.recv.asyncio
from aiokatcp import Sensor, SensorSet
from katsdpsigproc.abc import AbstractContext
from katsdptelstate.endpoint import Endpoint

from katgpucbf import COMPLEX
from katgpucbf.spead import BF_RAW_ID, FREQUENCY_ID, TIMESTAMP_ID
from katgpucbf.utils import TimeConverter
from katgpucbf.xbgpu.bsend import BSend
from katgpucbf.xbgpu.output import BOutput

from . import test_parameters

HEAPS_PER_FENG_PER_CHUNK: Final[int] = 5
N_TX_ITEMS: Final[int] = 2
TOTAL_HEAPS: Final[int] = N_TX_ITEMS * HEAPS_PER_FENG_PER_CHUNK


@pytest.fixture
def time_converter() -> TimeConverter:
    # TODO: Probably best to use adc_sample_rate from meerkat.py::BANDS
    return TimeConverter(123456789.0, 1712e6)


@pytest.fixture
def output() -> BOutput:
    return BOutput(name="test", dst=Endpoint("localhost", 7150))


@pytest.fixture
def sensors(output: BOutput) -> SensorSet:
    """Create sensors that the send code updates."""
    sensors = SensorSet()
    sensors.add(Sensor(int, f"{output.name}.beng-clip-cnt", "Number of output samples that are saturated."))
    return sensors


class TestBSend:
    """Test :class:`katgpucbf.xbgpu.bsend.BSend`."""

    @staticmethod
    async def _send_data(
        send_stream: BSend,
        heap_timestamp_step: int,
        time_converter: TimeConverter,
        sensors: SensorSet,
    ) -> None:
        """Send a fixed number of heaps.

        More specifically, send `N_TX_ITEMS` Chunks, each of which contain
        `HEAPS_PER_FENG_PER_CHUNK` heaps.
        """
        # Send the descriptors as the recv_stream object needs it to
        # interpret the received heaps correctly.
        await send_stream.stream.async_send_heap(send_stream.descriptor_heap)

        for i in range(N_TX_ITEMS):
            # Get a free chunk - there is not always a free one available. This
            # function blocks until one is available.
            chunk = await send_stream.get_free_chunk()

            # Populate the buffer with dummy data.
            chunk.data.fill(0)

            # Give the chunk back to the send_stream to transmit out
            # onto the network.
            chunk.timestamp = i * HEAPS_PER_FENG_PER_CHUNK * heap_timestamp_step
            chunk.send(send_stream, time_converter, sensors)
        # send_heap just queues data for sending but is non-blocking.
        # Flush to ensure that the data all gets sent before we return.
        await send_stream.stream.async_flush()

    @staticmethod
    async def _recv_data(
        recv_stream: spead2.recv.asyncio.Stream,
        channel_offset: int,
        n_channels_per_substream: int,
        n_spectra_per_heap: int,
        heap_timestamp_step: int,
    ) -> None:
        """Receive data transmitted from :func:`_send_data`.

        Error-check data here as well.
        """
        ig = spead2.ItemGroup()

        # Wait for the first packet to arrive - it is expected to be the
        # SPEAD descriptor.
        heap = await recv_stream.get()
        # TODO: No checks on the count sequence yet
        items = ig.update(heap)
        assert items == {}, "This heap contains item values not just the expected descriptors."

        # Check the data heaps
        zero_data = np.zeros(shape=(n_channels_per_substream, n_spectra_per_heap, COMPLEX), dtype=np.int8)
        for i in range(TOTAL_HEAPS):
            heap = await recv_stream.get()
            items = ig.update(heap)
            assert set(items.keys()) == {"timestamp", "frequency", "bf_raw"}
            assert items["timestamp"].id == TIMESTAMP_ID
            assert items["timestamp"].value == i * heap_timestamp_step
            assert items["frequency"].id == FREQUENCY_ID
            assert items["frequency"].value == channel_offset
            assert items["bf_raw"].id == BF_RAW_ID
            assert items["bf_raw"].value.shape == (n_channels_per_substream, n_spectra_per_heap, COMPLEX)
            assert items["bf_raw"].value.dtype == np.int8
            np.testing.assert_equal(items["bf_raw"].value, zero_data)

    @pytest.mark.combinations(
        "num_channels, num_spectra_per_heap",
        test_parameters.num_channels,
        test_parameters.num_spectra_per_heap,
    )
    async def test_send_simple(
        self,
        context: AbstractContext,
        num_channels: int,
        num_spectra_per_heap: int,
        output: BOutput,
        time_converter: TimeConverter,
        sensors: SensorSet,
    ) -> None:
        """
        Test :class:`katgpucbf.xbgpu.bsend.BSend`.

        This test transmits a number of heaps from a BSend object over a spead2
        in-process transport. The received heaps are then checked.

        This test does not generate random data as it will take much more compute
        to check that the random data is received correctly.

        Parameters
        ----------
        context
            Device context for allocating buffers.
        num_channels
            Total number of channels processed by a (theoretical) F-engine.
        num_spectra_per_heap
            Total number of packed spectra in every recevied channel.
        """
        # TODO: We don't do channels * 2 anymore, but n-samples-between-spectra
        heap_timestamp_step = num_channels * 2 * num_spectra_per_heap
        # Arbitrarily chosen, channels-per-substream is dictated by the
        # F-engine anyway.
        n_channels_per_substream = 512
        channel_offset = n_channels_per_substream * 3
        queue = spead2.InprocQueue()
        send_stream = BSend(
            outputs=[output],
            heaps_per_fengine_per_chunk=HEAPS_PER_FENG_PER_CHUNK,
            n_tx_items=N_TX_ITEMS,
            n_channels_per_substream=n_channels_per_substream,
            spectra_per_heap=num_spectra_per_heap,
            timestamp_step=heap_timestamp_step,
            send_rate_factor=0.0,
            channel_offset=channel_offset,
            context=context,
            stream_factory=lambda stream_config, buffers: spead2.send.asyncio.InprocStream(
                spead2.ThreadPool(1), [queue], stream_config
            ),
            tx_enabled=True,
        )
        await self._send_data(send_stream, heap_timestamp_step, time_converter, sensors)
        queue.stop()

        recv_stream = spead2.recv.asyncio.Stream(spead2.ThreadPool(1), spead2.recv.StreamConfig())
        recv_stream.add_inproc_reader(queue)
        await self._recv_data(
            recv_stream, channel_offset, n_channels_per_substream, num_spectra_per_heap, heap_timestamp_step
        )
