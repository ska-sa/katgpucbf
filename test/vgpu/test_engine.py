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
from typing import Final

import aiokatcp
import numpy as np
import pytest
import spead2.send.asyncio

from katgpucbf import COMPLEX, N_POLS
from katgpucbf.utils import TimeConverter
from katgpucbf.vgpu.engine import VEngine
from katgpucbf.vgpu.recv import Layout

from .test_recv import gen_heaps

# The sample rate is much smaller than expected in practice, so that we reach
# 1-second chunking without excessive data.
ADC_SAMPLE_RATE: Final = 1712e3
SYNC_TIME: Final = 1234567890
TIME_CONVERTER: Final = TimeConverter(SYNC_TIME, ADC_SAMPLE_RATE)
RECV_CHANNELS: Final = 32768
RECV_SUBSTREAMS = 4
RECV_CHANNELS_PER_SUBSTREAM: Final = RECV_CHANNELS // RECV_SUBSTREAMS
NB_DECIMATION: Final = 8
SEND_BANDWIDTH: Final = 64e3
FIR_TAPS: Final = 7201
STATION: Final = "me"


@pytest.fixture
def n_recv_streams() -> int:  # noqa: D103
    return N_POLS  # vgpu uses a separate stream for each polarisation


def _make_beng(queues: list[spead2.InprocQueue]) -> "spead2.send.asyncio.AsyncStream":
    config = spead2.send.StreamConfig(max_packet_size=8872)
    return spead2.send.asyncio.InprocStream(spead2.ThreadPool(), queues, config)


async def _send_data(layout: Layout, mock_recv_streams: list[spead2.InprocQueue]) -> None:
    """Send data to the engine."""
    beng = _make_beng(mock_recv_streams)
    data = np.zeros(
        (N_POLS, 4 * layout.n_batches_per_chunk, layout.n_channels, layout.n_spectra_per_heap, COMPLEX), np.int8
    )
    for heap in gen_heaps(layout, data, 0):
        await beng.async_send_heap(heap.heap, substream_index=heap.substream_index)


class TestVEngine:
    """Test :class:`.VEngine`."""

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
            "--send-interface=lo",
            f"--send-bandwidth={SEND_BANDWIDTH}",
            "--send-pols=x,y",
            f"--send-station={STATION}",
            "--send-samples-per-frame=4000",  # Reduced to keep the test small
            f"--fir-taps={FIR_TAPS}",
            f"--adc-sample-rate={ADC_SAMPLE_RATE}",
            f"--sync-time={SYNC_TIME}",
            f"239.10.0.0+{RECV_SUBSTREAMS - 1}:7148",
            f"239.10.1.0+{RECV_SUBSTREAMS - 1}:7148",
            "239.10.2.0",
        ]

    async def test_smoke(
        self,
        engine: VEngine,
        engine_client: aiokatcp.Client,
        mock_recv_streams: list[spead2.InprocQueue],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that an engine can be started and receives some data.

        This is a weak test that will need to be filled out later to ensure
        that the framesets contain the right headers and data.
        """
        n_framesets = 0

        def process_frameset(frameset):
            nonlocal n_framesets
            n_framesets += 1

        capture_complete_event = asyncio.Event()
        await engine_client.request("capture-start")
        monkeypatch.setattr(engine._capture, "_capture_complete", capture_complete_event.set)
        monkeypatch.setattr(engine._capture, "_process_frameset", process_frameset)
        await _send_data(engine.config.recv_config.layout, mock_recv_streams)
        for queue in mock_recv_streams:
            queue.stop()
        await capture_complete_event.wait()
        assert n_framesets > 0
        await engine_client.request("capture-stop")
