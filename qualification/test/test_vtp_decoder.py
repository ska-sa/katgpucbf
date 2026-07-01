#! /usr/bin/env python3

################################################################################
# Copyright (c) 2026, National Research Foundation (SARAO)
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

"""Check for :class:`qualification.recv.VTPDecoder`.

Run directly::
    qualification/.venv/bin/python qualification/test/test_vtp_decoder.py
"""

import asyncio
import io
import struct
import sys
from pathlib import Path

import numpy as np
from baseband.vdif import VDIFFrame

from qualification.recv import VTPBuffer, VTPDecoder

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

SAMPLES_PER_FRAME = 8000
BANDWIDTH = 64000000
N_THREADS = 4


def make_vtp_packet(
    seq_id: int, frame_nr: int, seconds: int, thread_id: int, samples_per_frame: int = SAMPLES_PER_FRAME
) -> bytes:
    """Build a VTP packet (8-byte seq header + VDIF frame)."""
    data = np.zeros((SAMPLES_PER_FRAME, 1), dtype=np.complex64)
    frame = VDIFFrame.fromdata(
        data,
        frame_nr=frame_nr,
        seconds=seconds,
        samples_per_frame=samples_per_frame,
        nchan=1,
        bps=2,
        complex_data=True,
        thread_id=thread_id,
    )
    buf = io.BytesIO()
    frame.tofile(buf)
    return struct.pack("<Q", seq_id) + buf.getvalue()


def check_samples_per_frame_from_first_frame() -> None:
    """`check_samples_per_frame_from_first_frame` checks that the first frame sets the samples_per_frame."""
    buffer = VTPBuffer()
    buffer.add_packet(make_vtp_packet(0, frame_nr=1, seconds=100, thread_id=0))
    assert buffer.samples_per_frame == SAMPLES_PER_FRAME
    assert buffer.seq_ids == [0]


async def check_decode_vdif_framesets_filters_incomplete_threads() -> None:
    """`check_decode_vdif_framesets_filters_incomplete_threads` checks that decode_vdif_framesets correctly filters out framesets with incomplete threads."""  # noqa: E501
    buffer = VTPBuffer()
    # 2 theads data only
    buffer.add_packet(make_vtp_packet(0, frame_nr=0, seconds=100, thread_id=0))
    buffer.add_packet(make_vtp_packet(1, frame_nr=0, seconds=100, thread_id=1))
    # 4 threads but only 3 unique thread ids
    buffer.add_packet(make_vtp_packet(2, frame_nr=1, seconds=100, thread_id=0))
    buffer.add_packet(make_vtp_packet(3, frame_nr=1, seconds=100, thread_id=1))
    buffer.add_packet(make_vtp_packet(4, frame_nr=1, seconds=100, thread_id=2))
    buffer.add_packet(make_vtp_packet(5, frame_nr=1, seconds=100, thread_id=0))
    decoder = VTPDecoder(buffer, 4)

    framesets = [item async for item in decoder.vtp_framesets()]
    assert len(framesets) == 0
    assert len(decoder.invalid_framesets) == 2


async def check_decode_vdif_framesets_unordered() -> None:
    """`check_decode_vdif_framesets_unordered` checks that decode_vdif_framesets yields on each frameset boundary."""
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
    decoder = VTPDecoder(buffer, 4)
    # todo: just do random access in a test instead.

    framesets = [item async for item in decoder.vtp_framesets()]
    assert len(decoder.invalid_framesets) == 1, f"Invalid framesets: {decoder.invalid_framesets}"
    assert len(framesets) == 2
    assert framesets[0] == ([6, 8, 9, 10], 100)
    assert framesets[1] == ([7, 11, 12, 13], 100)


async def check_decode_vdif_framesets_second_border() -> None:
    """`check_decode_vdif_framesets_second_border` checks that decode_vdif_framesets yields on each frameset boundary over multiple seconds."""  # noqa: E501
    buffer = VTPBuffer()
    buffer.add_packet(make_vtp_packet(0, frame_nr=0, seconds=100, thread_id=0))
    buffer.add_packet(make_vtp_packet(1, frame_nr=0, seconds=100, thread_id=1))
    buffer.add_packet(make_vtp_packet(2, frame_nr=1, seconds=100, thread_id=0))
    buffer.add_packet(make_vtp_packet(3, frame_nr=1, seconds=100, thread_id=1))
    buffer.add_packet(make_vtp_packet(4, frame_nr=0, seconds=101, thread_id=0))
    buffer.add_packet(make_vtp_packet(5, frame_nr=0, seconds=101, thread_id=1))
    buffer.add_packet(make_vtp_packet(6, frame_nr=1, seconds=101, thread_id=0))
    buffer.add_packet(make_vtp_packet(7, frame_nr=1, seconds=101, thread_id=1))
    decoder = VTPDecoder(buffer, 2)
    # todo: just do random access in a test instead.

    framesets = [item async for item in decoder.vtp_framesets()]
    assert len(decoder.invalid_framesets) == 0
    assert len(framesets) == 4
    assert framesets[0] == ([0, 1], 100)
    assert framesets[1] == ([2, 3], 100)
    assert framesets[2] == ([4, 5], 101)
    assert framesets[3] == ([6, 7], 101)


def check_close_clears_state() -> None:
    """`check_close_clears_state` checks that clear() resets all buffered state."""
    buffer = VTPBuffer()
    buffer.add_packet(make_vtp_packet(0, frame_nr=0, seconds=100, thread_id=0))
    buffer.add_packet(make_vtp_packet(1, frame_nr=0, seconds=100, thread_id=0))
    buffer.clear()
    assert buffer.seq_ids == []
    assert buffer.seconds == []
    assert buffer.thread_ids == []
    assert buffer.samples_per_frame is None
    assert buffer.frame_ids == []


def main() -> None:
    """Run all checks."""
    checks = [check_close_clears_state, check_samples_per_frame_from_first_frame]
    for check in checks:
        print(f"running {check.__name__}...", flush=True)
        check()
        print("  ok", flush=True)

    print("running check_decode_vdif_framesets...", flush=True)
    asyncio.run(check_decode_vdif_framesets_filters_incomplete_threads())
    print("  ok", flush=True)
    print("running check_decode_vdif_framesets_unordered...", flush=True)
    asyncio.run(check_decode_vdif_framesets_unordered())
    print("  ok", flush=True)
    print("running check_decode_vdif_framesets_second_border...", flush=True)
    asyncio.run(check_decode_vdif_framesets_second_border())
    print("  ok", flush=True)

    print("all checks passed")


if __name__ == "__main__":
    main()
