################################################################################
# Copyright (c) 2026, National Research Foundation (SARAO)
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of
# the License at
#
#   https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

"""Unit tests for :class:`qualification.recv.TiedArrayResampledVoltageReceiver`."""

import io
import socket
import struct
from math import ceil
from unittest import mock

import aiokatcp
import numpy as np
import pytest
from aiokatcp.sensor import Sensor, SensorSet
from baseband.vdif import VDIFFrame as BasebandVDIFFrame

from qualification.cbf import CBFRemoteControl
from qualification.recv import TiedArrayResampledVoltageReceiver, VDIFFrame, VDIFTimestamp

SAMPLES_PER_FRAME = 8000
BANDWIDTH = 64e6


def make_vtp_packet(
    seq_id: int,
    frame_nr: int,
    seconds: int,
    thread_id: int,
    samples_per_frame: int = SAMPLES_PER_FRAME,
    ref_epoch: int = 0,
) -> bytes:
    """Build a VTP packet (8-byte seq header + VDIF frame)."""
    data = np.zeros((SAMPLES_PER_FRAME, 1), dtype=np.float32)
    frame = BasebandVDIFFrame.fromdata(
        data,
        frame_nr=frame_nr,
        seconds=seconds,
        samples_per_frame=samples_per_frame,
        edv=0,
        nchan=1,
        bps=2,
        ref_epoch=ref_epoch,
        complex_data=False,
        thread_id=thread_id,
        legacy_mode=False,
    )
    buf = io.BytesIO()
    frame.tofile(buf)
    return struct.pack("<Q", seq_id) + buf.getvalue()


@pytest.fixture
def mock_cbf() -> CBFRemoteControl:
    """Mock CBFRemoteControl for testing, with values set matching the default test bandwidth."""
    cbf = mock.Mock(spec=CBFRemoteControl)
    cbf.init_sensors = SensorSet()
    cbf.init_sensors.add(
        Sensor(int, "stream0.n-chans", "Number of channels", "chans", default=2, initial_status=Sensor.Status.NOMINAL)
    )
    cbf.init_sensors.add(
        Sensor(
            bytes,
            "stream0.pol-ordering",
            "Polarisation ordering",
            "json string",
            default=b'["V", "H"]',
            initial_status=Sensor.Status.NOMINAL,
        )
    )
    cbf.init_sensors.add(
        Sensor(
            int,
            "stream0.veng-out-bits-per-sample",
            "Bits per sample",
            "bits",
            default=2,
            initial_status=Sensor.Status.NOMINAL,
        )
    )
    cbf.init_sensors.add(
        Sensor(
            float,
            "stream0.scale-factor-timestamp",
            "Scale factor timestamp",
            "scale-factor-timestamp",
            default=1.0,
            initial_status=Sensor.Status.NOMINAL,
        )
    )
    cbf.init_sensors.add(
        Sensor(
            float,
            "stream0.power-int-time",
            "Power integration time",
            "power-int-time",
            default=1.0,
            initial_status=Sensor.Status.NOMINAL,
        )
    )
    cbf.init_sensors.add(
        Sensor(
            float,
            "stream0.n-samples-per-frame",
            "Number of samples per frame",
            "n-samples-per-frame",
            default=SAMPLES_PER_FRAME,
            initial_status=Sensor.Status.NOMINAL,
        )
    )
    cbf.init_sensors.add(
        Sensor(float, "stream0.bandwidth", "Bandwidth", "bandwidth", default=64e6, initial_status=Sensor.Status.NOMINAL)
    )
    cbf.init_sensors.add(
        Sensor(
            float, "stream0.sync-time", "Sync time", "sync-time", default=100.0, initial_status=Sensor.Status.NOMINAL
        )
    )
    cbf.init_sensors.add(
        Sensor(
            bytes,
            "stream0.destination",
            "Destination",
            "ip address",
            default=b"127.0.0.1",
            initial_status=Sensor.Status.NOMINAL,
        )
    )
    cbf.product_controller_client = mock.Mock(spec=aiokatcp.Client)
    cbf.product_controller_client.sensor_value = mock.AsyncMock(return_value=0.0)
    cbf.steady_state_timestamp = mock.AsyncMock(return_value=0)
    return cbf


@pytest.fixture
def mock_socket() -> socket.socket:
    """A fixture that creates a mock socket.socket to inject packets into the receiver."""
    sock = mock.Mock(spec=socket.socket)
    return sock


class TestTiedArrayResampledVoltageReceiver:
    """Tests for :class:`qualification.recv.TiedArrayResampledVoltageReceiver`."""

    async def test_receive_samples_per_frame_from_first_frame(
        self, mock_cbf: CBFRemoteControl, mock_socket: socket.socket
    ) -> None:
        """First frame_set sets buffer."""
        receiver = TiedArrayResampledVoltageReceiver(mock_cbf, "stream0", "127.0.0.1", sock=mock_socket)
        vdif_frame = make_vtp_packet(0, frame_nr=1, seconds=100, thread_id=0)
        mock_socket.recv.return_value = vdif_frame  # type: ignore[attr-defined]
        await receiver._next_packet()
        assert receiver.frame_rate == BANDWIDTH / SAMPLES_PER_FRAME
        assert receiver.buffer == [
            VDIFFrame(
                seq_id=0,
                thread_id=0,
                timestamp=VDIFTimestamp(seconds=100, frame_nr=1, ref_epoch=0),
                raw_frame=vdif_frame[8:],
            )
        ]  # raw frame excludes the vdif header

    async def test_receive_framesets_filters_incomplete_threads(
        self, mock_cbf: CBFRemoteControl, mock_socket: socket.socket
    ) -> None:
        """Incomplete thread sets are recorded as invalid framesets."""
        receiver = TiedArrayResampledVoltageReceiver(mock_cbf, "stream0", "127.0.0.1", sock=mock_socket)
        # 2 threads of data only
        mock_socket.recv.side_effect = [  # type: ignore[attr-defined]
            make_vtp_packet(0, frame_nr=0, seconds=100, thread_id=0),
            make_vtp_packet(1, frame_nr=0, seconds=100, thread_id=1),
            make_vtp_packet(2, frame_nr=1, seconds=100, thread_id=0),
            make_vtp_packet(3, frame_nr=1, seconds=100, thread_id=1),
            make_vtp_packet(4, frame_nr=1, seconds=100, thread_id=2),
            make_vtp_packet(5, frame_nr=1, seconds=100, thread_id=0),
            make_vtp_packet(100, frame_nr=0, seconds=101, thread_id=0),
        ]
        # 4 threads but only 3 unique thread ids

        with pytest.raises(RuntimeError):
            await anext(receiver.complete_framesets())

    async def test_receive_framesets_filters_untill_delay(
        self, mock_cbf: CBFRemoteControl, mock_socket: socket.socket
    ) -> None:
        """
        Framesets are filtered until the delay is reached.

        When no min_timestamp is provided, the delay used is provided by the CBF's steady_state_timestamp method.
        """
        # delay time is relative to sync time, so set a known sync time to be relative to the first frame.
        mock_cbf.init_sensors.add(
            Sensor(
                float,
                "stream0.sync-time",
                "Sync time",
                "sync-time",
                default=VDIFTimestamp(0, 0, 0).timestamp(ceil(BANDWIDTH / SAMPLES_PER_FRAME)).unix,
                initial_status=Sensor.Status.NOMINAL,
            )
        )
        # steady state timestamp is used when no min_timestamp is provided
        mock_cbf.steady_state_timestamp.return_value = 10.0 / (BANDWIDTH / SAMPLES_PER_FRAME)  # type: ignore[attr-defined]
        receiver = TiedArrayResampledVoltageReceiver(mock_cbf, "stream0", "127.0.0.1", sock=mock_socket)
        mock_socket.recv.side_effect = [  # type: ignore[attr-defined]
            # these frames are still before the steady state timestamp value.
            make_vtp_packet(0, frame_nr=8, seconds=0, thread_id=0),
            make_vtp_packet(1, frame_nr=8, seconds=0, thread_id=1),
            make_vtp_packet(2, frame_nr=8, seconds=0, thread_id=2),
            make_vtp_packet(3, frame_nr=8, seconds=0, thread_id=3),
            # these frames are after the steady state timestamp value.
            make_vtp_packet(4, frame_nr=100, seconds=0, thread_id=0),
            make_vtp_packet(5, frame_nr=100, seconds=0, thread_id=1),
            make_vtp_packet(6, frame_nr=100, seconds=0, thread_id=2),
            make_vtp_packet(7, frame_nr=100, seconds=0, thread_id=3),
        ]
        frameset = await anext(receiver.complete_framesets())
        # The first frameset is the one after the steady state timestamp value.
        assert frameset.timestamp == VDIFTimestamp(seconds=0, frame_nr=100, ref_epoch=0)
        assert len(frameset.frames) == 4

    async def test_receive_framesets_filters_duplicate_seq_ids(
        self, mock_cbf: CBFRemoteControl, mock_socket: socket.socket
    ) -> None:
        """Framesets with duplicate sequence IDs are filtered out."""
        receiver = TiedArrayResampledVoltageReceiver(mock_cbf, "stream0", "127.0.0.1", sock=mock_socket)
        mock_socket.recv.side_effect = [  # type: ignore[attr-defined]
            make_vtp_packet(2, frame_nr=1, seconds=100, thread_id=0),
            make_vtp_packet(3, frame_nr=1, seconds=100, thread_id=1),
            make_vtp_packet(4, frame_nr=1, seconds=100, thread_id=2),
            make_vtp_packet(4, frame_nr=1, seconds=100, thread_id=2),
            make_vtp_packet(5, frame_nr=1, seconds=100, thread_id=3),
            make_vtp_packet(6, frame_nr=2, seconds=100, thread_id=0),
            make_vtp_packet(6, frame_nr=2, seconds=100, thread_id=2),
            make_vtp_packet(6, frame_nr=2, seconds=100, thread_id=3),
            make_vtp_packet(100, frame_nr=0, seconds=101, thread_id=0),
        ]

        frameset = await anext(receiver.complete_framesets())
        assert len(frameset.frames) == 4

    async def test_receive_framesets_unordered(self, mock_cbf: CBFRemoteControl, mock_socket: socket.socket) -> None:
        """Framesets are yielded correctly when packets arrive out of order."""
        receiver = TiedArrayResampledVoltageReceiver(mock_cbf, "stream0", "127.0.0.1", sock=mock_socket)
        mock_socket.recv.side_effect = [  # type: ignore[attr-defined]
            make_vtp_packet(8, frame_nr=10, seconds=98, thread_id=0),
            make_vtp_packet(10, frame_nr=1, seconds=98, thread_id=1),
            make_vtp_packet(6, frame_nr=0, seconds=98, thread_id=2),
            make_vtp_packet(9, frame_nr=1, seconds=98, thread_id=0),
            make_vtp_packet(4, frame_nr=0, seconds=98, thread_id=0),
            make_vtp_packet(12, frame_nr=1, seconds=98, thread_id=3),
            make_vtp_packet(11, frame_nr=1, seconds=98, thread_id=2),
            make_vtp_packet(5, frame_nr=0, seconds=98, thread_id=1),
            make_vtp_packet(7, frame_nr=0, seconds=98, thread_id=3),
            make_vtp_packet(1000, frame_nr=0, seconds=101, thread_id=0),
        ]
        framesets = []
        with pytest.raises(RuntimeError):
            async for frameset in receiver.complete_framesets():
                framesets.append(frameset)

        assert len(framesets) == 2
        assert framesets[0].timestamp == VDIFTimestamp(seconds=98, frame_nr=0, ref_epoch=0)
        assert len(framesets[0].frames) == 4
        assert framesets[1].timestamp == VDIFTimestamp(seconds=98, frame_nr=1, ref_epoch=0)
        assert len(framesets[1].frames) == 4

    async def test_receive_duplicate_packets_ignored(
        self, mock_cbf: CBFRemoteControl, mock_socket: socket.socket
    ) -> None:
        """Duplicate packets are ignored."""
        receiver = TiedArrayResampledVoltageReceiver(mock_cbf, "stream0", "127.0.0.1", sock=mock_socket)
        mock_socket.recv.side_effect = [  # type: ignore[attr-defined]
            make_vtp_packet(8, frame_nr=10, seconds=98, thread_id=0),
            make_vtp_packet(8, frame_nr=10, seconds=98, thread_id=0),
            make_vtp_packet(9, frame_nr=10, seconds=98, thread_id=1),
            make_vtp_packet(10, frame_nr=10, seconds=98, thread_id=2),
            make_vtp_packet(11, frame_nr=10, seconds=98, thread_id=3),
            make_vtp_packet(1000, frame_nr=0, seconds=101, thread_id=0),
        ]
        framesets = []
        with pytest.raises(RuntimeError):
            async for frameset in receiver.complete_framesets():
                framesets.append(frameset)

        assert len(framesets) == 1
        assert framesets[0].timestamp == VDIFTimestamp(seconds=98, frame_nr=10, ref_epoch=0)
        assert len(framesets[0].frames) == 4

    async def test_timestamp_leap_seconds(self, mock_cbf: CBFRemoteControl, mock_socket: socket.socket) -> None:
        """Timestamp is calculated correctly from the epoch."""
        receiver = TiedArrayResampledVoltageReceiver(mock_cbf, "stream0", "127.0.0.1", sock=mock_socket)
        mock_socket.recv.side_effect = [  # type: ignore[attr-defined]
            make_vtp_packet(0, frame_nr=1, seconds=15897600, thread_id=0, ref_epoch=33),
            make_vtp_packet(1, frame_nr=1, seconds=15897700, thread_id=1, ref_epoch=33),
            make_vtp_packet(2, frame_nr=1, seconds=100, thread_id=2, ref_epoch=34),
        ]
        await receiver._next_packet()
        await receiver._next_packet()
        await receiver._next_packet()
        ref_time = receiver.buffer[2].timestamp.timestamp(receiver.frame_rate).unix
        assert ref_time - receiver.buffer[1].timestamp.timestamp(receiver.frame_rate).unix == pytest.approx(1), (
            "should have a leap second difference"
        )
        assert ref_time - receiver.buffer[0].timestamp.timestamp(receiver.frame_rate).unix == pytest.approx(101), (
            "should have a difference of 101 seconds"
        )

    async def test_close_clears_state(self, mock_cbf: CBFRemoteControl, mock_socket: socket.socket) -> None:
        """clear() resets all buffered state."""
        receiver = TiedArrayResampledVoltageReceiver(mock_cbf, "stream0", "127.0.0.1", sock=mock_socket)
        mock_socket.recv.side_effect = [  # type: ignore[attr-defined]
            make_vtp_packet(0, frame_nr=0, seconds=100, thread_id=0),
            make_vtp_packet(1, frame_nr=0, seconds=100, thread_id=0),
        ]
        await receiver._next_packet()
        receiver.close()
        assert receiver.buffer == []
        assert receiver.sock.close.call_count == 1  # type: ignore[attr-defined]
