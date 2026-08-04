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

"""Unit tests for :mod:`qualification.vlbi_decoder.vtp_decoder`."""

import io
import struct

import numpy as np
import pytest
from baseband.vdif import VDIFFrame

from qualification.vlbi_decoder.vtp_decoder import VDIFFramesetData, VDIFFramesetKey, VTPBuffer, VTPDecoder

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
    data = np.zeros((SAMPLES_PER_FRAME, 1), dtype=np.complex64)
    frame = VDIFFrame.fromdata(
        data,
        frame_nr=frame_nr,
        seconds=seconds,
        samples_per_frame=samples_per_frame,
        nchan=2,
        bps=2,
        ref_epoch=ref_epoch,
        complex_data=False,
        thread_id=thread_id,
    )
    buf = io.BytesIO()
    frame.tofile(buf)
    return struct.pack("<Q", seq_id) + buf.getvalue()


def test_samples_per_frame_from_first_frame() -> None:
    """First frame sets samples_per_frame on the buffer."""
    buffer = VTPBuffer()
    buffer.add_packet(make_vtp_packet(0, frame_nr=1, seconds=100, thread_id=0))
    assert buffer.samples_per_frame == SAMPLES_PER_FRAME
    assert buffer.seq_ids == [0]


def test_decode_vdif_framesets_filters_incomplete_threads() -> None:
    """Incomplete thread sets are recorded as invalid framesets."""
    buffer = VTPBuffer()
    # 2 threads of data only
    buffer.add_packet(make_vtp_packet(0, frame_nr=0, seconds=100, thread_id=0))
    buffer.add_packet(make_vtp_packet(1, frame_nr=0, seconds=100, thread_id=1))
    # 4 threads but only 3 unique thread ids
    buffer.add_packet(make_vtp_packet(2, frame_nr=1, seconds=100, thread_id=0))
    buffer.add_packet(make_vtp_packet(3, frame_nr=1, seconds=100, thread_id=1))
    buffer.add_packet(make_vtp_packet(4, frame_nr=1, seconds=100, thread_id=2))
    buffer.add_packet(make_vtp_packet(5, frame_nr=1, seconds=100, thread_id=0))
    decoder = VTPDecoder(buffer, 4, BANDWIDTH)

    framesets = list(decoder.vtp_framesets())
    assert len(framesets) == 0
    assert len(decoder.invalid_framesets) == 2


def test_decode_vdif_framesets_filters_duplicate_seq_ids() -> None:
    """Framesets with duplicate sequence IDs are filtered out."""
    buffer = VTPBuffer()
    buffer.add_packet(make_vtp_packet(4, frame_nr=1, seconds=100, thread_id=0))
    buffer.add_packet(make_vtp_packet(5, frame_nr=1, seconds=100, thread_id=1))
    buffer.add_packet(make_vtp_packet(5, frame_nr=2, seconds=100, thread_id=0))
    buffer.add_packet(make_vtp_packet(5, frame_nr=2, seconds=100, thread_id=1))
    buffer.add_packet(make_vtp_packet(5, frame_nr=3, seconds=100, thread_id=1))
    decoder = VTPDecoder(buffer, 2, BANDWIDTH)

    framesets = list(decoder.vtp_framesets())
    assert len(framesets) == 1
    assert len(decoder.invalid_framesets) == 2


def test_decode_vdif_framesets_unordered() -> None:
    """Framesets are yielded correctly when packets arrive out of order."""
    buffer = VTPBuffer()
    buffer.add_packet(make_vtp_packet(5, frame_nr=10, seconds=99, thread_id=0))
    buffer.add_packet(make_vtp_packet(6, frame_nr=0, seconds=100, thread_id=0))
    buffer.add_packet(make_vtp_packet(11, frame_nr=1, seconds=100, thread_id=1))
    buffer.add_packet(make_vtp_packet(7, frame_nr=1, seconds=100, thread_id=0))
    buffer.add_packet(make_vtp_packet(13, frame_nr=1, seconds=100, thread_id=3))
    buffer.add_packet(make_vtp_packet(8, frame_nr=0, seconds=100, thread_id=1))
    buffer.add_packet(make_vtp_packet(9, frame_nr=0, seconds=100, thread_id=2))
    buffer.add_packet(make_vtp_packet(10, frame_nr=0, seconds=100, thread_id=3))
    buffer.add_packet(make_vtp_packet(12, frame_nr=1, seconds=100, thread_id=2))
    decoder = VTPDecoder(buffer, 4, BANDWIDTH)

    framesets = list(decoder.vtp_framesets())
    assert len(decoder.invalid_framesets) == 1
    assert len(framesets) == 2
    assert framesets[0] == (
        VDIFFramesetData(seq_ids=[6, 8, 9, 10], thread_ids={0, 1, 2, 3}),
        VDIFFramesetKey(100, 0, 0),
    )
    assert framesets[1] == (
        VDIFFramesetData(seq_ids=[7, 11, 12, 13], thread_ids={0, 1, 2, 3}),
        VDIFFramesetKey(100, 1, 0),
    )


def test_decode_vdif_framesets_second_border() -> None:
    """Framesets are grouped correctly across a second boundary."""
    buffer = VTPBuffer()
    buffer.add_packet(make_vtp_packet(0, frame_nr=0, seconds=100, thread_id=0))
    buffer.add_packet(make_vtp_packet(1, frame_nr=0, seconds=100, thread_id=1))
    buffer.add_packet(make_vtp_packet(2, frame_nr=1, seconds=100, thread_id=0))
    buffer.add_packet(make_vtp_packet(3, frame_nr=1, seconds=100, thread_id=1))
    buffer.add_packet(make_vtp_packet(4, frame_nr=0, seconds=101, thread_id=0))
    buffer.add_packet(make_vtp_packet(5, frame_nr=0, seconds=101, thread_id=1))
    buffer.add_packet(make_vtp_packet(6, frame_nr=1, seconds=101, thread_id=0))
    buffer.add_packet(make_vtp_packet(7, frame_nr=1, seconds=101, thread_id=1))
    decoder = VTPDecoder(buffer, 2, BANDWIDTH)

    framesets = list(decoder.vtp_framesets())
    assert len(decoder.invalid_framesets) == 0
    assert len(framesets) == 4
    assert framesets[0] == (VDIFFramesetData(seq_ids=[0, 1], thread_ids={0, 1}), VDIFFramesetKey(100, 0, 0))
    assert framesets[1] == (VDIFFramesetData(seq_ids=[2, 3], thread_ids={0, 1}), VDIFFramesetKey(100, 1, 0))
    assert framesets[2] == (VDIFFramesetData(seq_ids=[4, 5], thread_ids={0, 1}), VDIFFramesetKey(101, 0, 0))
    assert framesets[3] == (VDIFFramesetData(seq_ids=[6, 7], thread_ids={0, 1}), VDIFFramesetKey(101, 1, 0))


def test_frame_rate_from_bandwidth() -> None:
    """Decoder derives frame_rate from bandwidth and samples_per_frame."""
    buffer = VTPBuffer()
    buffer.add_packet(make_vtp_packet(0, frame_nr=0, seconds=100, thread_id=0))
    decoder = VTPDecoder(buffer, 1, BANDWIDTH)
    assert decoder.frame_rate == round(BANDWIDTH / SAMPLES_PER_FRAME)


def test_timestamp_from_epoch() -> None:
    """Timestamp is calculated correctly from the epoch."""
    buffer = VTPBuffer()
    buffer.add_packet(make_vtp_packet(0, frame_nr=1, seconds=100, thread_id=0, ref_epoch=2))
    decoder = VTPDecoder(buffer, 1, BANDWIDTH)
    _, key = next(decoder.vtp_framesets())
    assert key.timestamp(decoder.frame_rate) == pytest.approx(978307300.000125)


def test_clear_clears_state() -> None:
    """clear() resets all buffered state."""
    buffer = VTPBuffer()
    buffer.add_packet(make_vtp_packet(0, frame_nr=0, seconds=100, thread_id=0))
    buffer.add_packet(make_vtp_packet(1, frame_nr=0, seconds=100, thread_id=0))
    buffer.clear()
    assert buffer.seq_ids == []
    assert buffer.seconds == []
    assert buffer.thread_ids == []
    assert buffer.samples_per_frame == 0
    assert buffer.frame_ids == []
