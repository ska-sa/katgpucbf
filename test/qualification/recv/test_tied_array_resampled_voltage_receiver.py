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
from collections.abc import AsyncGenerator
from unittest import mock

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
    return cbf


@pytest.fixture
async def mock_socket() -> AsyncGenerator[socket.socket, None]:
    """A fixture that creates a mock socket.socket to inject packets into the receiver."""
    sock = mock.AsyncMock(spec=socket.socket)
    with mock.patch("socket.socket", return_value=sock):
        yield sock


async def test_receive_samples_per_frame_from_first_frame(
    mock_cbf: CBFRemoteControl, mock_socket: socket.socket
) -> None:
    """First frame sets samples_per_frame on the buffer."""
    receiver = TiedArrayResampledVoltageReceiver(mock_cbf, ["stream0"], "127.0.0.1")
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
    mock_cbf: CBFRemoteControl, mock_socket: socket.socket
) -> None:
    """Incomplete thread sets are recorded as invalid framesets."""
    receiver = TiedArrayResampledVoltageReceiver(mock_cbf, ["stream0"], "127.0.0.1")
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


async def test_receive_framesets_filters_duplicate_seq_ids(
    mock_cbf: CBFRemoteControl, mock_socket: socket.socket
) -> None:
    """Framesets with duplicate sequence IDs are filtered out."""
    receiver = TiedArrayResampledVoltageReceiver(mock_cbf, ["stream0"], "127.0.0.1")
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


async def test_receive_framesets_unordered(mock_cbf: CBFRemoteControl, mock_socket: socket.socket) -> None:
    """Framesets are yielded correctly when packets arrive out of order."""
    receiver = TiedArrayResampledVoltageReceiver(mock_cbf, ["stream0"], "127.0.0.1")
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

    # assert len(decoder.invalid_framesets) == 1
    assert len(framesets) == 2
    assert framesets[0].timestamp == VDIFTimestamp(seconds=98, frame_nr=0, ref_epoch=0)
    assert len(framesets[0].frames) == 4
    assert framesets[1].timestamp == VDIFTimestamp(seconds=98, frame_nr=1, ref_epoch=0)
    assert len(framesets[1].frames) == 4


async def test_timestamp_from_epoch(mock_cbf: CBFRemoteControl, mock_socket: socket.socket) -> None:
    """Timestamp is calculated correctly from the epoch."""
    receiver = TiedArrayResampledVoltageReceiver(mock_cbf, ["stream0"], "127.0.0.1")
    mock_socket.recv.side_effect = [  # type: ignore[attr-defined]
        make_vtp_packet(0, frame_nr=1, seconds=100, thread_id=0, ref_epoch=2)
    ]
    await receiver._next_packet()
    assert receiver.buffer[0].timestamp.unix_timestamp(receiver.frame_rate) == pytest.approx(978307300.000125)


async def test_close_clears_state(mock_cbf: CBFRemoteControl, mock_socket: socket.socket) -> None:
    """clear() resets all buffered state."""
    receiver = TiedArrayResampledVoltageReceiver(mock_cbf, ["stream0"], "127.0.0.1")
    mock_socket.recv.side_effect = [  # type: ignore[attr-defined]
        make_vtp_packet(0, frame_nr=0, seconds=100, thread_id=0),
        make_vtp_packet(1, frame_nr=0, seconds=100, thread_id=0),
    ]
    await receiver._next_packet()
    receiver.close()
    assert receiver.buffer == []
    assert receiver.frame_rate == 0
    assert receiver.sock.close.call_count == 1  # type: ignore[attr-defined]
