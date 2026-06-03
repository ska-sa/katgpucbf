################################################################################
# Copyright (c) 2025-2026, National Research Foundation (SARAO)
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

"""Unit tests for :mod:`katgpucbf.vgpu.engine`."""

import asyncio
import io
import math
import struct
from collections.abc import Callable
from typing import Final

import aiokatcp
import astropy.units as u
import numpy as np
import pytest
import spead2.send.asyncio
from astropy.time import Time
from baseband.vdif import VDIFFrame

from katgpucbf import COMPLEX, N_POLS
from katgpucbf.utils import TimeConverter
from katgpucbf.vgpu.engine import VEngine, _CaptureSession
from katgpucbf.vgpu.recv import Layout

from .test_recv import gen_heaps

# The sample rate is much smaller than expected in practice, so that we reach
# 1-second chunking without excessive data.
ADC_SAMPLE_RATE: Final = 1712e3
SYNC_TIME: Final = 1234567890
TIME_CONVERTER: Final = TimeConverter(SYNC_TIME, ADC_SAMPLE_RATE)
RECV_CHANNELS: Final = 1024
RECV_SUBSTREAMS = 4
RECV_CHANNELS_PER_SUBSTREAM: Final = RECV_CHANNELS // RECV_SUBSTREAMS
NB_DECIMATION: Final = 8
SEND_BANDWIDTH: Final = 64e3
FIR_TAPS: Final = 7201
STATION: Final = "me"
SAMPLES_PER_FRAME: Final = 4000  # Reduced to keep the tests small
FIRST_TIMESTAMP: Final = 0x30000000  # must be a multiple of chunk_timestamp_step
N_THREADS: Final = 4


@pytest.fixture
def n_recv_streams() -> int:  # noqa: D103
    return N_POLS  # vgpu uses a separate stream for each polarisation


def _make_beng(queues: list[spead2.InprocQueue]) -> "spead2.send.asyncio.AsyncStream":
    config = spead2.send.StreamConfig(max_packet_size=8872)
    return spead2.send.asyncio.InprocStream(spead2.ThreadPool(), queues, config)


async def _send_data(
    layout: Layout,
    chunks: int,
    mock_recv_streams: list[spead2.InprocQueue],
    factory: Callable[[np.ndarray], None] | None = None,
) -> None:
    """Send data to the engine.

    If `factory` is given, it is passed the data and should fill in the values. If not
    given, zeros are sent.
    """
    assert FIRST_TIMESTAMP % layout.chunk_timestamp_step == 0
    beng = _make_beng(mock_recv_streams)
    data = np.zeros(
        (N_POLS, chunks * layout.n_batches_per_chunk, layout.n_channels, layout.n_spectra_per_heap, COMPLEX), np.int8
    )
    if factory is not None:
        factory(data)
    for heap in gen_heaps(layout, data, FIRST_TIMESTAMP):
        await beng.async_send_heap(heap.heap, substream_index=heap.substream_index)


class TestVEngine:
    """Test :class:`.VEngine`."""

    @pytest.fixture
    def capture_complete_event(self, engine: VEngine, monkeypatch: pytest.MonkeyPatch) -> asyncio.Event:
        """Asyncio event that is set when :meth:`._CaptureSession._capture_complete` is called."""
        capture_complete_event = asyncio.Event()

        def capture_complete(self):
            capture_complete_event.set()

        monkeypatch.setattr(_CaptureSession, "_capture_complete", capture_complete)
        return capture_complete_event

    @pytest.fixture
    def engine_arglist(self) -> list[str]:
        """Command-line arguments for the engine."""
        return [
            "--katcp-host=127.0.0.1",
            "--katcp-port=0",  # OS will allocate a port
            "--recv-pols=-y,+x",
            "--recv-interface=lo",
            f"--recv-channels={RECV_CHANNELS}",
            f"--recv-channels-per-substream={RECV_CHANNELS_PER_SUBSTREAM}",
            f"--recv-samples-between-spectra={NB_DECIMATION * RECV_CHANNELS * 2}",
            "--recv-jones-per-batch=16384",  # Reduce batch sizes to speed up test
            "--recv-batches-per-chunk=1",  # Reduce chunk sizes to speed up test
            "--send-interface=lo",
            f"--send-bandwidth={SEND_BANDWIDTH}",
            "--send-pols=x,y",
            f"--send-station={STATION}",
            f"--send-samples-per-frame={SAMPLES_PER_FRAME}",
            f"--fir-taps={FIR_TAPS}",
            f"--adc-sample-rate={ADC_SAMPLE_RATE}",
            f"--sync-time={SYNC_TIME}",
            "--send-rate-factor=0",  # Make it send infinitely fast, to avoid slowing the test
            f"239.10.0.0+{RECV_SUBSTREAMS - 1}:7148",
            f"239.10.1.0+{RECV_SUBSTREAMS - 1}:7148",
            "239.10.2.0",
        ]

    @pytest.mark.parametrize(
        "min_timestamp",
        [
            0,
            FIRST_TIMESTAMP + 0x400000,
        ],
    )
    async def test_smoke(
        self,
        engine: VEngine,
        engine_client: aiokatcp.Client,
        mock_recv_streams: list[spead2.InprocQueue],
        sendmsg_packets: list[bytes],
        capture_complete_event: asyncio.Event,
        min_timestamp: int,
    ) -> None:
        """Test that an engine can be started, receives and transmits some data.

        This is a weak test that considers only the VDIF headers of the output
        and not the payload.
        """
        assert min_timestamp % engine.config.recv_config.layout.chunk_timestamp_step == 0, (
            "min_timestamp is not on a chunk boundary"
        )
        chunks = 60
        await engine_client.request("capture-start", min_timestamp)
        await _send_data(engine.config.recv_config.layout, chunks, mock_recv_streams)
        for queue in mock_recv_streams:
            queue.stop()
        await capture_complete_event.wait()
        await engine_client.request("capture-stop")
        frame_rate = round(SEND_BANDWIDTH / SAMPLES_PER_FRAME) * u.Hz
        # The pipeline rounds things to a 1s boundary, so we should see the
        # data start at the next second boundary after start_time_adc. Note
        # that this only holds if FIRST_TIMESTAMP is not too close to a 1s
        # boundary, because the filters have a group delay that is compensated
        # for.
        start_time_adc = max(FIRST_TIMESTAMP, min_timestamp)
        start_time_unix = math.ceil(TIME_CONVERTER.adc_to_unix(start_time_adc))
        start_time = Time(start_time_unix, format="unix")
        # The first second of data is incomplete because of the footprint of the
        # filters. We then align the start to a second boundary, so we expect
        # data to start 1s after SYNC_TIME.
        for i, packet in enumerate(sendmsg_packets):
            assert len(packet) > 40  # Must have at least 8-byte VTP header and 32-byte VDIF header
            fh = io.BytesIO(packet)
            seq = struct.unpack("<Q", fh.read(8))[0]
            frame = VDIFFrame.fromfile(fh)
            header = frame.header
            frame_time = frame.get_time(frame_rate=frame_rate)
            expected_time = start_time + i // N_THREADS / frame_rate
            assert seq == i
            assert not header.complex_data
            assert not header["invalid_data"]
            assert header.bps == 2
            assert header.nchan == 1
            assert header.samples_per_frame == SAMPLES_PER_FRAME
            assert header.station == STATION
            assert header["thread_id"] == i % N_THREADS
            assert (frame_time - expected_time).sec == pytest.approx(0, abs=1e-9)
        data_timestamps = engine.config.recv_config.layout.chunk_timestamp_step * chunks
        # Again, this calculation only works if the exact value is
        # sufficiently far from a 1s boundary that the filter footprint
        # doesn't mess things up.
        stop_time_unix = math.floor(TIME_CONVERTER.adc_to_unix(FIRST_TIMESTAMP + data_timestamps))
        assert stop_time_unix > start_time_unix, "Test did not send enough data to produce output"
        assert len(sendmsg_packets) == (stop_time_unix - start_time_unix) * frame_rate.value * N_THREADS

    async def test_capture_start_while_capturing(self, engine_client: aiokatcp.Client) -> None:
        """Test that ``?capture-start`` while already capturing fails."""
        await engine_client.request("capture-start", 0)
        with pytest.raises(aiokatcp.FailReply, match="a capture is already in progress"):
            await engine_client.request("capture-start", 0)

    async def test_capture_stop_while_not_capturing(self, engine_client: aiokatcp.Client) -> None:
        """Test that ``?capture-stop`` while not capturing fails."""
        with pytest.raises(aiokatcp.FailReply, match="no capture in progress"):
            await engine_client.request("capture-stop")
        await engine_client.request("capture-start", 0)
        await engine_client.request("capture-stop")
        with pytest.raises(aiokatcp.FailReply, match="no capture in progress"):
            await engine_client.request("capture-stop")
