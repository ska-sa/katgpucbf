#!/usr/bin/env python3

################################################################################
# Copyright (c) 2021-2024, 2026, National Research Foundation (SARAO)
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

import argparse
import asyncio
import bisect
import io
import logging
import math
import socket
import struct
import sys
from collections.abc import AsyncGenerator, Buffer, Sequence
from dataclasses import dataclass

import aiokatcp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time, TimeDelta
from baseband.vdif import VDIFFrame as BasebandVDIFFrame
from katsdpservices import get_interface_address
from katsdptelstate.endpoint import Endpoint, endpoint_list_parser, endpoint_parser

from katgpucbf import DEFAULT_VTP_PORT

CPLX = 2
CORR2_SENSOR_REMAP = {
    "antenna-channelised-voltage.adc-sample-rate": "adc-sample-rate",
}
# According to the ICD.
TIMESTAMP = 0x1600
FREQUENCY = 0x4103
logger = logging.getLogger(__name__)
SAMPLES_PER_FRAME = 8000


async def get_product_controller_endpoint(mc_endpoint: Endpoint, product_name: str) -> Endpoint:
    """Get the katcp address for a named product controller from the master."""
    client = await aiokatcp.Client.connect(*mc_endpoint)
    async with client:
        return endpoint_parser(None)(str(await client.sensor_value(f"{product_name}.katcp-address", aiokatcp.Address)))


async def get_subordinate_endpoint(mc_endpoint: Endpoint, product_name: str) -> Endpoint:
    """Get the katcp address for an array's subordinate controller from the CMC."""
    client = await aiokatcp.Client.connect(*mc_endpoint)
    async with client:
        reply, informs = await client.request("subordinate-list")
        for inform in informs:
            if aiokatcp.decode(str, inform.arguments[0]) == product_name:
                ports = aiokatcp.decode(str, inform.arguments[1]).split(",")
                return Endpoint(mc_endpoint.host, int(ports[0]))


def get_bls_subset(baselines: Sequence[str], bls_ordering: Sequence[tuple[str, str]]) -> list[int]:
    """Parse the command-line --baseline arguments."""
    if baselines:
        bls_subset = []
        for bls in baselines:
            bls_tuple = tuple(bls.split(","))  # Turn "m000h,m000v" into ("m000h", "m000v")
            try:
                idx = bls_ordering.index(bls_tuple)
            except ValueError:
                print(f"Baseline {bls} not found", file=sys.stderr)
                sys.exit(1)
            bls_subset.append(idx)
    else:
        bls_subset = list(range(len(bls_ordering)))
    return bls_subset


@dataclass(frozen=True, order=True, slots=True)
class VDIFTimestamp:
    """Identifier of an VDIF frameset."""

    seconds: int
    frame_nr: int
    ref_epoch: int
    frame_rate: int

    @property
    def timestamp(self) -> Time:
        """Timestamp of the VDIF frameset."""
        # ref_epoch is the number of half-years since 2000-01-01.
        ref_time = Time(
            dict(year=2000 + self.ref_epoch // 2, month=self.ref_epoch % 2 * 6 + 1), format="ymdhms", scale="utc"
        )
        seconds = self.seconds + self.frame_nr / self.frame_rate
        return ref_time + TimeDelta(seconds, format="sec", scale="tai")

    @property
    def linear(self) -> int:
        """Linear frame number counting from the ref epoch."""
        return self.seconds * self.frame_rate + self.frame_nr


@dataclass(frozen=True, slots=True)
class VDIFFrame:
    """Holds low-level information about a single VDIF frame."""

    seq_id: int
    thread_id: int
    timestamp: VDIFTimestamp
    raw_frame: Buffer  #: VDIF frame (including VDIF header but without VTP sequence header)


@dataclass(slots=True)
class VDIFFrameset:
    """Data for a VDIF frameset."""

    frames: Sequence[VDIFFrame]

    @property
    def timestamp(self) -> VDIFTimestamp:
        """Get the common timestamp of the frames."""
        return self.frames[0].timestamp


IP_MULTICAST_ALL = 49
VDIF_HEADER_SIZE = 32


class TiedArrayResampledVoltageReceiver:
    """Receive tied-array-resampled-voltage streams from the V-engines."""

    VTP_HEADER_SIZE = 8

    def __init__(
        self,
        stream_name: str,
        interface_address: str,
        multicast_groups: list[Endpoint],
        sock: socket.socket | None = None,
        reorder_window: int = 32,
    ) -> None:
        self.stream_name = stream_name
        self.multicast_groups = multicast_groups
        self.n_chans: int = 2
        self.n_threads = self.n_chans * 2
        self.veng_out_bits_per_sample: int = 2

        # all multicast groups must use the same port
        port = self.multicast_groups[0].port
        for multicast_group in self.multicast_groups:
            if multicast_group.port != port:
                raise ValueError("All multicast groups must use the same port")

        self.steady_state_timestamp: float = 0.0

        self.scale_factor_timestamp: float = 1
        self.power_int_time: float = 0.5
        self.bandwidth: int = int(64e6)
        self.n_samples_per_frame: int = 8000
        self.frame_rate = round(self.bandwidth / self.n_samples_per_frame)
        self.sync_time: float = 0.0
        self.sample_bytes_per_frame = self.n_samples_per_frame * self.veng_out_bits_per_sample // 8
        packet_size = self.sample_bytes_per_frame + VDIF_HEADER_SIZE + self.VTP_HEADER_SIZE
        self._packet = bytearray(packet_size)
        # Number of times we've received a packet without sleeping
        self._sync_packets = 0

        if sock is None:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.setsockopt(socket.SOL_SOCKET, IP_MULTICAST_ALL, 0)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 64 * 1024 * 1024)
            sock.bind(("", port))
            for multicast_group in self.multicast_groups:
                mreq = socket.inet_aton(multicast_group.host) + socket.inet_aton(interface_address)
                sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
            sock.setblocking(False)
        self.sock = sock
        self.buffer: list[VDIFFrame] = []  # Kept sorted by sequence ID
        self.min_seq_id = 0  # Minimum sequence ID we're still willing to accept for reordering
        self.reorder_window = reorder_window
        # Frames dropped due to incomplete frameset or too old for reorder window and already missed frameset
        self.dropped_frames = 0
        self.missed_framesets = 0
        self.total_framesets = 0
        self.last_frameset_timestamp: int | None = None
        self.filter_shift = 0.0

    async def _next_packet(self) -> None:
        # sock_recv_into will work synchronously if it can, but that can
        # prevent the event loop from ever getting a chance to run.
        # sleep(0) allows this. Performance is better if we don't sleep(0)
        # every time, so we pick a tuning number to balance responsiveness
        # with performance.
        self._sync_packets += 1
        if self._sync_packets == 16:
            await asyncio.sleep(0)
            self._sync_packets = 0

        packet_size = await asyncio.get_event_loop().sock_recv_into(self.sock, self._packet)
        assert packet_size == len(self._packet)
        # Using baseband to parse the header is expensive. We extract
        # words from the header then slice out the fields we want.
        (seq_id, seconds, ref_epoch_frame_nr, length, sample_bits_thread_id) = struct.unpack_from(
            "<QIIIxxH", self._packet
        )
        seconds = seconds & 0x3FFF_FFFF
        ref_epoch = (ref_epoch_frame_nr >> 24) & 0x3F
        frame_nr = ref_epoch_frame_nr & 0xFF_FFFF
        thread_id = sample_bits_thread_id & 0x3F
        length = (length & 0xFF_FFFF) * 8 - VDIF_HEADER_SIZE  # Raw value includes the header
        if frame_nr == 0:
            is_complex = sample_bits_thread_id >> 15
            assert not is_complex

        timestamp = VDIFTimestamp(seconds=seconds, frame_nr=frame_nr, ref_epoch=ref_epoch, frame_rate=self.frame_rate)
        frame = VDIFFrame(seq_id=seq_id, thread_id=thread_id, timestamp=timestamp, raw_frame=self._packet[8:])
        assert length == self.sample_bytes_per_frame
        if seq_id >= self.min_seq_id:
            self.min_seq_id = max(self.min_seq_id, seq_id - self.reorder_window)
            # Insert the new frame into its sorted position
            pos = bisect.bisect_left(self.buffer, seq_id, key=lambda f: f.seq_id)
            if pos == len(self.buffer) or self.buffer[pos].seq_id != seq_id:
                self.buffer.insert(pos, frame)
            else:
                logger.warning("Duplicate sequence ID: %d", seq_id)
        else:
            logger.debug("Frame too old: %d < %d", seq_id, self.min_seq_id)
            self.dropped_frames += 1

    def _calc_min_frame(self, min_time: Time, ref_epoch: int) -> int:
        """Compute the minimum linearised frame number that is at least `min_time`."""
        ref_time = VDIFTimestamp(0, 0, ref_epoch, frame_rate=self.frame_rate).timestamp
        to_wait_sec = (min_time - ref_time).sec
        return math.ceil(to_wait_sec * self.frame_rate)

    async def complete_framesets(
        self,
        min_timestamp: int | None = None,
        *,
        time_limit: float | None = None,
    ) -> AsyncGenerator[VDIFFrameset, None]:
        """Yield complete framesets.

        Note that this iterator is only suitable for a single capture session.
        If the stream is stopped then restarted, create a new iterator.

        Parameters
        ----------
        min_timestamp
            Framesets that incorporate ADC samples dated prior to this time
            (after F-engine delay is applied) will be discarded. If the default
            of ``None`` is used, a value is computed via
            :meth:`CBFRemoteControl.steady_state_timestamp`.
        max_delay
            An upper bound on the delay set on any F-engine. This is used in
            the calculation of `min_timestamp` when no value is provided.
        time_limit
            If a floating-point value is given, the iteration will end after
            this many seconds. Note that no :exc:`asyncio.TimeoutError` will be
            raised.
        """
        if min_timestamp is None:
            min_timestamp = self.steady_state_timestamp
        min_time = Time(self.sync_time, format="unix", scale="utc") + TimeDelta(
            min_timestamp / self.scale_factor_timestamp, format="sec", scale="tai"
        )

        min_time += TimeDelta(0.0, format="sec", scale="tai") + self.filter_shift

        # Computing timestamps is expensive. So once we know the frame rate and
        # ref epoch, we can convert it into a minimum linearised frame number.
        min_frame: int | None = None
        c = 0
        try:
            async with asyncio.timeout(time_limit) as timer:
                while True:
                    c += 1
                    if c % 8000 == 0:
                        logger.debug("Missed framesets: %d", self.missed_framesets)
                        logger.debug("Total framesets: %d", self.total_framesets)
                        logger.debug("Dropped frames: %d", self.dropped_frames)
                    # Check if we have a complete frameset at the start of the buffer.
                    if self.buffer:
                        frame0 = self.buffer[0]
                        if min_frame is None:
                            min_frame = self._calc_min_frame(min_time, frame0.timestamp.ref_epoch)
                        if frame0.seq_id <= self.min_seq_id and len(self.buffer) >= self.n_threads:
                            prefix = self.buffer[: self.n_threads]
                            if all(
                                frame.timestamp == frame0.timestamp and frame.seq_id < frame0.seq_id + self.n_threads
                                for frame in prefix
                            ) and sorted(frame.thread_id for frame in prefix) == list(range(self.n_threads)):
                                # Make sure we don't re-accept these packets
                                self.min_seq_id = max(self.min_seq_id, prefix[0].seq_id + self.n_threads)
                                prefix.sort(key=lambda frame: frame.thread_id)
                                frame0_linear = frame0.timestamp.linear
                                if frame0_linear >= min_frame:
                                    n_missed_framesets = 0
                                    if self.last_frameset_timestamp is not None:
                                        n_missed_framesets = frame0_linear - self.last_frameset_timestamp - 1
                                        if n_missed_framesets < 0:
                                            logger.warning(
                                                "Negative number of framesets between last and new frameset!:"
                                                + " last frameset: %d new frameset: %d, diff: %d",
                                                self.last_frameset_timestamp,
                                                frame0_linear,
                                                n_missed_framesets,
                                            )
                                            n_missed_framesets = 0

                                    self.missed_framesets += n_missed_framesets
                                    self.total_framesets += n_missed_framesets + 1
                                    self.last_frameset_timestamp = frame0_linear

                                    yield VDIFFrameset(prefix)
                                else:
                                    logger.debug(
                                        "Skipping frameset because of min_timestamp: %d < %d", frame0_linear, min_frame
                                    )
                                    self.total_framesets += 1
                                del self.buffer[: self.n_threads]
                                continue

                        if frame0.seq_id <= self.min_seq_id - self.n_threads:
                            # If this frameset isn't complete now, it never will be. Drop the frame.
                            logger.debug("Dropping frame due to incomplete frameset: %d", frame0.timestamp.linear)
                            del self.buffer[0]
                            self.dropped_frames += 1
                            continue

                    # If we didn't hit a continue above, we need more data from the network
                    await self._next_packet()
        except TimeoutError:
            if not timer.expired():
                raise  # The TimeoutError came from something else

    async def next_complete_frameset(self, min_timestamp: int | None = None) -> VDIFFrameset:
        """Listen until a single complete VDIF frameset is available, then return it."""
        async for frameset in self.complete_framesets(min_timestamp=min_timestamp):
            return frameset
        raise RuntimeError("stream was shut down before we received a complete frameset")

    async def wait_complete_frameset(
        self, min_timestamp: int | None = None, *, timeout: float | None = 10.0
    ) -> VDIFTimestamp:
        """Wait until a complete VDIF frameset is available and return the timestamp."""
        async with asyncio.timeout(timeout):
            return (await self.next_complete_frameset(min_timestamp=min_timestamp)).timestamp

    def close(self) -> None:
        """Close the socket."""
        self.sock.close()
        self.buffer.clear()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--interface",
        type=get_interface_address,
        required=True,
        help="Name of network interface.",
    )
    parser.add_argument(
        "--mc-address",
        type=endpoint_parser(5001),
        default="lab-mc.sdp.kat.ac.za:5001",  # Naturally this applies only to our lab...
        metavar="HOST:PORT",
        help="Master controller to query for details about the product. [%(default)s]",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Show dumps as they are received",
    )
    parser.add_argument(
        "--phase",
        action="store_true",
        help="Plot phase instead of magnitude",
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=4,
        help="Number of columns to plot",
    )
    parser.add_argument(
        "--fake-data",
        action="store_true",
        help="Use fake data instead of real data",
        default=False,
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level",
    )
    #    parser.add_argument("product_name", type=str, help="Name of the subarray to get baselines from.")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level)
    if args.interactive:
        matplotlib.use("WebAgg")
        # plt.show will run a Tornado event loop, which in turn runs an asyncio
        # event loop. So we can't directly run an event loop ourselves, but
        # must instead schedule async_main onto the one run by matplotlib.
        asyncio.get_event_loop().create_task(async_main(args))
        plt.show()
    else:
        asyncio.run(async_main(args))


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


class FakeSocket:
    def __init__(self):
        self.seq_id = 0
        self.frame_nr = 0
        self.seconds = 1
        self.thread_id = 0

    def recv_into(self, buf: Buffer) -> int:
        mv = memoryview(buf).cast("B")
        if self.thread_id > 3:
            self.thread_id = 0
            self.frame_nr += 1
        if self.frame_nr >= 8000:
            self.frame_nr = 0
            self.seconds += 1
            logger.debug("Generating second %d", self.seconds)

        packet = make_vtp_packet(self.seq_id, self.frame_nr, self.seconds, self.thread_id)
        self.seq_id += 1
        self.thread_id += 1
        if self.seconds >= 11:
            raise ConnectionResetError("ran out of pre-defined packets") from None
        assert len(mv) >= len(packet)
        mv[: len(packet)] = packet
        return len(packet)


async def async_main(args: argparse.Namespace) -> None:
    if not args.fake_data:
        host, port = await get_product_controller_endpoint(args.mc_address, args.product_name)
        client = await aiokatcp.Client.connect(host, port)

        async with client:

            async def sensor_val(name, sensor_type=None):
                if args.corr2:
                    name = CORR2_SENSOR_REMAP.get(name, name.replace(".", "-"))
                return await client.sensor_value(name, sensor_type)

            # Spead2 doesn't know katsdptelstate so it can't recognise Endpoints.
            # But we can cast Endpoints to tuples, which it does know.
            multicast_endpoints = [
                tuple(endpoint)
                for endpoint in endpoint_list_parser(DEFAULT_VTP_PORT)(
                    await sensor_val("baseline-resampled-voltage.destination", str)
                )
            ]
            receiver = TiedArrayResampledVoltageReceiver(
                stream_name=args.product_name,
                interface_address=args.interface,
                multicast_groups=multicast_endpoints,
                sock=FakeSocket(),
            )

    else:
        receiver = TiedArrayResampledVoltageReceiver(
            stream_name="tied-array-resampled-voltage",
            interface_address=args.interface,
            multicast_groups=[Endpoint("127.0.0.1", 5001)],
            sock=FakeSocket(),
        )

    fig, axs = plt.subplots(
        receiver.n_chans,
        2,  # X and Y polarisations
        sharex=True,
        squeeze=False,
        figsize=(12, 8) if args.interactive else (24, 16),
        constrained_layout=True,
    )
    axs = axs.ravel()  # Now that the plot is several columns wide, this comes as a 2-d array. Which is a pain.
    axs = axs[:4]  # If there are gaps at the end, ignore them
    lines = []
    seconddata = np.zeros((4, 8000))
    for i in range(4):
        lines.append(axs[i].plot([0.0] * 8000, [0.0] * 8000)[0])
        axs[i].set_ylabel(i)
        axs[i].set_xlabel("Sample number")
    if args.interactive:
        fig.canvas.mpl_connect("close_event", lambda event: receiver.close())
        plt.ion()
        plt.show()
    logger.info("Starting to receive framesets")
    counter = 0
    async for chunk in receiver.complete_framesets():
        counter += 1
        for frame in chunk.frames:
            vdif_frame = BasebandVDIFFrame.fromfile(io.BytesIO(frame.raw_frame))
            seconddata[frame.thread_id, counter - 1] = np.average(vdif_frame.data)
        if counter == 8000:
            # update line
            for i in range(4):
                lines[i].set_ydata(seconddata[i, :])
            plt.title(f"Second {chunk.timestamp.seconds}")
            if not args.interactive:
                plt.savefig(f"{chunk.timestamp.seconds}.png")
            logger.info("Wrote second %d", chunk.timestamp.seconds)
            return


if __name__ == "__main__":
    main()
