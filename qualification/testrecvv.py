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

"""Test a sample implementation of a receiver for the tied-array-resampled-voltage stream."""

import argparse
import asyncio
import io
import logging
import socket
import struct
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

import astropy
from baseband.vdif import VDIFFrame

logger = logging.getLogger(__name__)


class VTPBuffer:
    """Buffer for storing VTP packets, and decoding them into VDIF framesets."""

    def __init__(self, bandwidth: float) -> None:
        self.seq_ids: list[int] = []
        self.frameset_timestamps: list[int] = []
        self.bandwidth = bandwidth
        self.framerate = None
        self.samples_per_frame = None
        self.zero_seq_id = list[int]()

    async def decode_vtp(self, packet: bytes) -> None:
        """Decode the VTP packets in the buffer. throw away the frame for now."""
        new_seq_id = struct.unpack("<Q", packet[:8])[0]
        frame = VDIFFrame.fromfile(io.BytesIO(packet[8:]))
        if frame.header["frame_nr"] == 0:
            if self.samples_per_frame is None:
                self.samples_per_frame = frame.header.samples_per_frame
                if TYPE_CHECKING:
                    assert self.samples_per_frame is not None
                self.framerate = round(self.bandwidth / self.samples_per_frame)
            self.zero_seq_id.append(new_seq_id)
            if self.framerate is not None:
                self.frameset_timestamps.append(
                    frame.header.get_time(frame_rate=self.framerate * astropy.units.Hz).to_value("unix")
                )
                print(f"New frameset: {new_seq_id} at {self.frameset_timestamps[-1]}")
        self.seq_ids.append(new_seq_id)

    def close(self) -> None:
        """Close the buffer."""
        self.seq_ids.clear()


class TiedArrayResampledVoltageReceiver:
    """Receive tied-array-resampled-voltage streams from the V-engines."""

    max_packet_size = 5040

    def __init__(
        self,
        interface_address: str,
        multicast_group: str,
        multicast_port: str,
        bandwidth: float,
    ) -> None:
        self.stream_names = ["tied-array-resampled-voltage"]
        # all multicast groups must use the same port
        #        port = self.multicast_groups[0].port
        #        for multicast_group in multicast_groups:
        #            if multicast_group.port != port:
        #                raise ValueError("All multicast groups must use the same port")

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.setsockopt(socket.SOL_SOCKET, 49, 0)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 * 1024 * 1024)
        self.socket.bind(("", multicast_port))
        mreq = socket.inet_aton(multicast_group) + socket.inet_aton(interface_address)
        self.socket.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        self.socket.setblocking(False)
        self.vtp_buffer = VTPBuffer(bandwidth)

    async def _read(self) -> bytes:
        loop = asyncio.get_running_loop()
        return await loop.sock_recv(self.socket, self.max_packet_size)

    async def listen(self) -> AsyncGenerator[tuple[int, bytes], None]:
        """Listen for packets from the v engine and store them in the VTPBuffer."""
        while True:
            await self.vtp_buffer.decode_vtp(await self._read())

    def check_sequences(self) -> None:
        """Check the sequences in the VTPBuffer."""
        seq_ids = sorted(self.vtp_buffer.seq_ids)
        prev_seq_id = seq_ids[0]
        print(f"First sequence ID: {prev_seq_id}")
        missed_sequences = 0
        for seq_id in seq_ids[1:]:
            if seq_id != prev_seq_id + 1:
                missed_sequences += seq_id - prev_seq_id - 1
            prev_seq_id = seq_id
        print(f"Last sequence ID: {prev_seq_id}")
        print(f"Missed {missed_sequences} sequences")

    def close(self) -> None:
        """Close the socket."""
        self.socket.close()
        self.vtp_buffer.close()


def parse_args() -> argparse.Namespace:
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface-address", type=str, required=True)
    parser.add_argument("--multicast-group", type=str, required=True)
    parser.add_argument("--multicast-port", type=int, required=True)
    parser.add_argument("--bandwidth", type=float, required=True)
    return parser.parse_args()


if __name__ == "__main__":

    async def main(args: argparse.Namespace):
        """Run the application."""
        receiver = TiedArrayResampledVoltageReceiver(
            args.interface_address, args.multicast_group, args.multicast_port, args.bandwidth
        )
        task = asyncio.create_task(receiver.listen())
        await asyncio.sleep(60)
        task.cancel()
        receiver.check_sequences()
        receiver.close()

    args = parse_args()
    asyncio.run(main(args))
