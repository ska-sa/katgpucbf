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

from typing import Final, Sequence

import numpy as np
import pytest
import spead2
import spead2.recv.asyncio
import spead2.send.asyncio
from aiokatcp import Sensor, SensorSet
from katsdpsigproc.abc import AbstractContext
from katsdptelstate.endpoint import Endpoint

from katgpucbf import COMPLEX
from katgpucbf.spead import BF_RAW_ID, FREQUENCY_ID, TIMESTAMP_ID
from katgpucbf.utils import TimeConverter
from katgpucbf.xbgpu.bsend import SEND_DTYPE, BSend
from katgpucbf.xbgpu.output import BOutput

from . import test_parameters

HEAPS_PER_CHUNK: Final[int] = 5
N_TX_ITEMS: Final[int] = 2
TOTAL_DATA_HEAPS_PER_SUBSTREAM: Final[int] = N_TX_ITEMS * HEAPS_PER_CHUNK


@pytest.fixture
def time_converter() -> TimeConverter:
    return TimeConverter(123456789.0, 1234e6)


@pytest.fixture
def outputs() -> Sequence[BOutput]:
    """Simulate `--beam` configuration."""
    return [
        BOutput(name="foo", dst=Endpoint("239.10.11.0", 7149)),
        BOutput(name="bar", dst=Endpoint("239.10.12.0", 7149)),
    ]


@pytest.fixture
def sensors(outputs: Sequence[BOutput]) -> SensorSet:
    """Create sensors that the send code updates."""
    sensors = SensorSet()
    for output in outputs:
        sensors.add(Sensor(int, f"{output.name}.beng-clip-cnt", "Number of output samples that are saturated."))
    return sensors


class TestBSend:
    """Test :class:`katgpucbf.xbgpu.bsend.BSend`."""

    @staticmethod
    async def _send_data(
        outputs: Sequence[BOutput],
        send_stream: BSend,
        heap_timestamp_step: int,
        n_channels_per_substream: int,
        n_spectra_per_heap: int,
        time_converter: TimeConverter,
        sensors: SensorSet,
    ) -> np.ndarray:
        """Send a fixed number of heaps.

        More specifically, in addition to a descriptor heap per substream, send
        `N_TX_ITEMS` Chunks, each of which contain `HEAPS_PER_CHUNK` heaps.

        Returns
        -------
        data
            Array of shape (HEAPS_PER_CHUNK, len(outputs), n_channels_per_substream, )
        """
        # Send the descriptors as the recv_stream object needs it to
        # interpret the received heaps correctly.
        for i, _ in enumerate(outputs):
            await send_stream.stream.async_send_heap(
                send_stream.descriptor_heap,
                substream_index=i,
            )

        data = np.zeros(
            shape=(N_TX_ITEMS, HEAPS_PER_CHUNK, len(outputs), n_channels_per_substream, n_spectra_per_heap, COMPLEX),
            dtype=SEND_DTYPE,
        )

        for i in range(N_TX_ITEMS):
            # Get a free chunk - there is not always a free one available. This
            # function blocks until one is available.
            chunk = await send_stream.get_free_chunk()

            # Generate random data for the i-th chunk
            rng = np.random.default_rng(seed=1)
            data[i, ...] = rng.integers(low=-128, high=127, size=(data.shape[1:]), dtype=np.int8)

            # Populate the buffer with dummy data.
            chunk.data[:] = data[i, ...]

            # Give the chunk back to the send_stream to transmit out
            # onto the network.
            chunk.timestamp = i * HEAPS_PER_CHUNK * heap_timestamp_step
            send_stream.send_chunk(chunk, time_converter, sensors)
        # send_heap just queues data for sending but is non-blocking.
        # Flush to ensure that the data all gets sent before we return.
        await send_stream.stream.async_flush()

        return data

    @staticmethod
    async def _recv_data(
        data: np.ndarray,
        outputs: Sequence[BOutput],
        queues: list[spead2.InprocQueue],
        n_engines: int,
        engine_id: int,
        channel_offset: int,
        n_channels_per_substream: int,
        n_spectra_per_heap: int,
        heap_timestamp_step: int,
    ) -> None:
        """Receive data transmitted from :meth:`_send_data`.

        Error-check data here as well.

        Parameters
        ----------
        data
            Random data generated during data transmission, of shape
            - (N_TX_ITEMS, HEAPS_PER_CHUNK, len(outputs), n_channels_per_substream, n_spectra_per_heap, COMPLEX)
        queues
            List of :class:`spead2.InprocQueue` used to transmit heaps
            in :meth:`_send_data`.
        """
        # Reshape as we verify *heaps* per substream, not chunks
        data = data.reshape(
            (TOTAL_DATA_HEAPS_PER_SUBSTREAM, len(outputs), n_channels_per_substream, n_spectra_per_heap, COMPLEX)
        )

        out_config = spead2.recv.StreamConfig()
        out_tp = spead2.ThreadPool()
        for i, queue in enumerate(queues):
            stream = spead2.recv.asyncio.Stream(out_tp, out_config)
            stream.add_inproc_reader(queue)

            # Wait for the first packet to arrive - it is expected to be the
            # SPEAD descriptor.
            ig = spead2.ItemGroup()
            heap = await stream.get()
            assert heap.cnt % n_engines == engine_id % n_engines, "The heap IDs are not correctly strided"
            items = ig.update(heap)
            assert items == {}, "This heap contains item values not just the expected descriptors."

            # Check the data heaps
            for j in range(TOTAL_DATA_HEAPS_PER_SUBSTREAM):
                heap = await stream.get()
                items = ig.update(heap)
                assert set(items.keys()) == {"timestamp", "frequency", "bf_raw"}
                assert items["timestamp"].id == TIMESTAMP_ID
                assert items["timestamp"].value == j * heap_timestamp_step
                assert items["frequency"].id == FREQUENCY_ID
                assert items["frequency"].value == channel_offset
                assert items["bf_raw"].id == BF_RAW_ID
                assert items["bf_raw"].value.shape == (n_channels_per_substream, n_spectra_per_heap, COMPLEX)
                assert items["bf_raw"].value.dtype == np.int8
                np.testing.assert_equal(items["bf_raw"].value, data[j, i, ...])

    @pytest.mark.combinations(
        "num_ants, num_channels, num_spectra_per_heap",
        test_parameters.array_size,
        test_parameters.num_channels,
        test_parameters.num_spectra_per_heap,
    )
    async def test_send_simple(
        self,
        context: AbstractContext,
        num_ants: int,
        num_channels: int,
        num_spectra_per_heap: int,
        outputs: Sequence[BOutput],
        time_converter: TimeConverter,
        sensors: SensorSet,
    ) -> None:
        """
        Test :class:`katgpucbf.xbgpu.bsend.BSend`.

        This test transmits a number of heaps from a BSend object over a spead2
        in-process transport. The received heaps are then checked.

        Parameters
        ----------
        context
            Device context for allocating buffers.
        num_channels
            Total number of channels processed by a (theoretical) F-engine.
        num_spectra_per_heap
            Total number of packed spectra in every recevied channel.
        outputs, time_converter, sensors
            Fixtures.
        """
        # We need the "total number of engines required" to process this array
        # configuration.
        n_engines = 1
        while n_engines < num_ants * 4:
            n_engines *= 2

        # The test still needs to have some idea of "which engine" this is in a
        # sequence of B-engines.
        engine_id = 4  # Arbitrarily chosen

        # TODO: We don't do channels * 2 anymore, but n-samples-between-spectra
        heap_timestamp_step = num_channels * 2 * num_spectra_per_heap
        n_channels_per_substream = num_channels // n_engines
        channel_offset = n_channels_per_substream * engine_id
        queues = [spead2.InprocQueue() for _ in outputs]
        send_stream = BSend(
            outputs=outputs,
            heaps_per_fengine_per_chunk=HEAPS_PER_CHUNK,
            n_tx_items=N_TX_ITEMS,
            n_channels=num_channels,
            n_channels_per_substream=n_channels_per_substream,
            spectra_per_heap=num_spectra_per_heap,
            adc_sample_rate=time_converter.adc_sample_rate,
            timestamp_step=heap_timestamp_step,
            send_rate_factor=0.0,  # Send as fast as possible
            channel_offset=channel_offset,
            context=context,
            stream_factory=lambda stream_config, buffers: spead2.send.asyncio.InprocStream(
                spead2.ThreadPool(1), queues, stream_config
            ),
            tx_enabled=True,
        )
        data = await self._send_data(
            outputs,
            send_stream,
            heap_timestamp_step,
            n_channels_per_substream,
            num_spectra_per_heap,
            time_converter,
            sensors,
        )
        for queue in queues:
            queue.stop()

        await self._recv_data(
            data,
            outputs,
            queues,
            n_engines,
            engine_id,
            channel_offset,
            n_channels_per_substream,
            num_spectra_per_heap,
            heap_timestamp_step,
        )