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

"""Unit tests for Engine functions."""

import asyncio
import logging
from typing import List

import numpy as np
import pytest
import spead2.send

from katgpucbf import COMPLEX, N_POLS
from katgpucbf.fgpu import SAMPLE_BITS, recv, send
from katgpucbf.fgpu.engine import Engine

pytestmark = [pytest.mark.cuda_only, pytest.mark.asyncio]
DIGITISER_ID_ID = 0x3101
DIGITISER_STATUS_ID = 0x3102
RAW_DATA_ID = 0x3300
FLAVOUR = spead2.Flavour(4, 64, 48, 0)  # Flavour for sending digitiser data
# Command-line arguments
SYNC_EPOCH = 1632561921
CHANNELS = 4096
SPECTRA_PER_HEAP = 256
CHUNK_SAMPLES = 1048576  # Lower than the default to make tests quicker
FENG_ID = 42


class TestEngine:
    r"""Grouping of unit tests for :class:`.Engine`\'s various functionality."""

    engine_arglist = [
        "--katcp-port=0",
        "--src-interface=lo",
        "--dst-interface=lo",
        f"--channels={CHANNELS}",
        f"--sync-epoch={SYNC_EPOCH}",
        f"--chunk-samples={CHUNK_SAMPLES}",
        f"--spectra-per-heap={SPECTRA_PER_HEAP}",
        f"--feng-id={FENG_ID}",
        "--send-rate-factor=0",  # Infinitely fast
        "239.10.10.0+7:7149",  # src1
        "239.10.10.8+7:7149",  # src2
        "239.10.11.0+15:7149",  # dst
    ]

    def test_engine_required_arguments(self, engine_server: Engine) -> None:
        """Test proper setting of required arguments.

        .. note::

          This doesn't test if the functionality described by these is in any
          way correct, just whether or not the member variables are being
          correctly populated.
        """
        assert engine_server._port == 0
        assert engine_server._src_interface == "127.0.0.1"
        # TODO: `dst_interface` goes to the _sender member, which doesn't have anything we can query.
        assert engine_server._processor.channels == CHANNELS
        assert engine_server.sync_epoch == SYNC_EPOCH
        assert engine_server._srcs == [
            [
                ("239.10.10.0", 7149),
                ("239.10.10.1", 7149),
                ("239.10.10.2", 7149),
                ("239.10.10.3", 7149),
                ("239.10.10.4", 7149),
                ("239.10.10.5", 7149),
                ("239.10.10.6", 7149),
                ("239.10.10.7", 7149),
            ],
            [
                ("239.10.10.8", 7149),
                ("239.10.10.9", 7149),
                ("239.10.10.10", 7149),
                ("239.10.10.11", 7149),
                ("239.10.10.12", 7149),
                ("239.10.10.13", 7149),
                ("239.10.10.14", 7149),
                ("239.10.10.15", 7149),
            ],
        ]
        # TODO: same problem for `dst` itself.

    def _make_digitiser(self, queues: List[spead2.InprocQueue]) -> "spead2.send.asyncio.AsyncStream":
        """Create send stream for a fake digitiser.

        The resulting stream has one sub-stream per polarisation.
        """
        config = spead2.send.StreamConfig(max_packet_size=9000)  # Just needs to be bigger than the heaps
        return spead2.send.asyncio.InprocStream(spead2.ThreadPool(), queues, config)

    async def _send_digitiser_heap(
        self, stream: "spead2.send.asyncio.AsyncStream", timestamp: int, pol: int, samples: np.ndarray
    ) -> None:
        heap = spead2.send.Heap(FLAVOUR)
        heap.add_item(spead2.Item(recv.TIMESTAMP_ID, "", "", shape=(), format=[("u", 48)], value=timestamp))
        heap.add_item(spead2.Item(DIGITISER_ID_ID, "", "", shape=(), format=[("u", 48)], value=pol))
        heap.add_item(spead2.Item(DIGITISER_STATUS_ID, "", "", shape=(), format=[("u", 48)], value=0))
        heap.add_item(spead2.Item(RAW_DATA_ID, "", "", shape=samples.shape, dtype=samples.dtype, value=samples))
        await stream.async_send_heap(heap, substream_index=pol)

    def _make_samples(self, n_samples: int) -> np.ndarray:
        """Synthesize some digitiser data with a tone.

        Each polarisation has a different tone frequency so that swaps can be
        detected. Returns an array of bytes containing packed digitiser
        samples.
        """
        rng = np.random.default_rng(1)
        data = np.zeros((N_POLS, n_samples), np.float32)
        data[0] = np.cos(np.arange(n_samples) / 16) * 400
        data[1] = np.cos(np.arange(n_samples) / 64) * 700
        # Dither the signal to reduce quantisation artifacts, then quantise
        data += rng.random(size=data.shape, dtype=np.float32)
        data = np.trunc(data).astype(">i2")  # Big endian
        # Unpack the bits, so that we can toss out the top 6
        bits = np.unpackbits(data.view(np.uint8), axis=1).reshape(data.shape + (16,))
        # Put all the bits back into bytes
        return np.packbits(bits[..., -SAMPLE_BITS:].reshape(N_POLS, -1), axis=1)

    async def _send_data(
        self,
        mock_recv_streams: List[spead2.InprocQueue],
        mock_send_stream: List[spead2.InprocQueue],
        engine: Engine,
        dig_data: np.ndarray,
        first_timestamp: int = 0,
    ) -> np.ndarray:
        """Send a contiguous stream of data to the engine and retrieve results.

        This is a little tricky because :func:`.chunk_sets` drops data
        if the pols get more than a chunk out of sync, and if we just push
        all the heaps in at once we have no control over the order in which
        spead2 processes them. To avoid getting too far ahead, we watch the
        sensor that indicates how many heaps have been received, and push
        updates to a queue that we can block on. We must transmit data from
        the next chunk to force spead2 to flush out a prior chunk.
        """
        # Reshape into heap-size pieces (now has indices pol, heap, offset)
        src_layout = engine._src_layout
        assert dig_data.shape[0] == N_POLS
        assert dig_data.shape[1] % src_layout.heap_bytes == 0, "samples must be a whole number of heaps"
        dig_data = dig_data.reshape(N_POLS, -1, src_layout.heap_bytes)
        dig_stream = self._make_digitiser(mock_recv_streams)
        heaps_received = 0
        heaps_received_queue = asyncio.Queue()  # type: asyncio.Queue[int]
        heaps_sensor = engine.sensors["input-heaps-total"]
        heaps_sensor.attach(lambda sensor, reading: heaps_received_queue.put_nowait(reading.value))
        for i in range(dig_data.shape[1]):
            for pol in range(N_POLS):
                await self._send_digitiser_heap(
                    dig_stream, i * src_layout.heap_samples + first_timestamp, pol, dig_data[pol, i]
                )
            while i >= heaps_received // N_POLS + src_layout.chunk_heaps:
                logging.debug("heaps_received = %d, waiting for more", heaps_received)
                heaps_received = await heaps_received_queue.get()
        for queue in mock_recv_streams:
            queue.stop()

        n_out_streams = len(mock_send_stream)
        assert n_out_streams == 16, "Number of output streams does not match command line"
        out_config = spead2.recv.StreamConfig()
        out_tp = spead2.ThreadPool()
        heaps = []
        for i, queue in enumerate(mock_send_stream):
            stream = spead2.recv.asyncio.Stream(out_tp, out_config)
            stream.add_inproc_reader(queue)
            ig = spead2.ItemGroup()
            # We don't have descriptors yet, so we have to build the Items manually
            imm_format = [("u", send.FLAVOUR.heap_address_bits)]
            raw_shape = (CHANNELS // n_out_streams, SPECTRA_PER_HEAP, N_POLS, COMPLEX)
            ig.add_item(send.TIMESTAMP_ID, "timestamp", "", shape=(), format=imm_format)
            ig.add_item(send.FENG_ID_ID, "feng_id", "", shape=(), format=imm_format)
            ig.add_item(send.FREQUENCY_ID, "frequency", "", shape=(), format=imm_format)
            ig.add_item(send.FENG_RAW_ID, "feng_raw", "", shape=raw_shape, dtype=np.int8)
            expected_timestamp = first_timestamp
            timestamp_step = SPECTRA_PER_HEAP * CHANNELS * 2  # TODO not valid for narrowband
            row = []
            async for heap in stream:
                assert set(ig.update(heap)) == {"timestamp", "feng_id", "frequency", "feng_raw"}
                assert ig["feng_id"].value == FENG_ID
                assert ig["timestamp"].value == expected_timestamp
                assert ig["frequency"].value == i * raw_shape[0]
                expected_timestamp += timestamp_step
                row.append(ig["feng_raw"].value.copy())
            # Glue all the heaps together along the time axis
            heaps.append(np.concatenate(row, axis=1))
        # Glue the parts of the band together along the channel axis. This
        # also ensures that there were the same number of heaps per channel.
        return np.concatenate(heaps, axis=0)

    async def test_end_to_end(
        self,
        recv_max_chunks_one,
        mock_recv_streams: List[spead2.InprocQueue],
        mock_send_stream: List[spead2.InprocQueue],
        engine_server: Engine,
    ) -> None:
        """Push data into the input streams and check results from the output streams."""
        src_layout = engine_server._src_layout
        n_samples = 20 * src_layout.chunk_samples
        dig_data = self._make_samples(n_samples)
        out_data = await self._send_data(mock_recv_streams, mock_send_stream, engine_server, dig_data)
        assert out_data.size > 0
