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
import time
from collections.abc import AsyncGenerator, Buffer, Sequence
from dataclasses import dataclass

import aiokatcp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time, TimeDelta
from baseband.base.encoding import decoder_levels, encode_2bit_base
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
CUTOFF = True
ZOOM = True
PHASE_FREQ = 1260e6
SYNC = True
SAMPLE_DELTA = 1 / (8000 * 8000)  # TODO actually calculate this

window_sample_time = np.arange(8000) * SAMPLE_DELTA
phase_signal = np.sin(2 * math.pi * PHASE_FREQ * window_sample_time)
phase_signal_quantised_sin = decoder_levels[2][encode_2bit_base(phase_signal)]
phase_signal_quantised_cos = decoder_levels[2][encode_2bit_base(np.cos(2 * math.pi * PHASE_FREQ * window_sample_time))]


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


@dataclass
class Band:
    """Holds presets for a known band."""

    long_name: str
    adc_sample_rate: float
    centre_frequency: float


def get_pass_channels(narrowband_decimation: int, n_channels: int) -> slice:
    band = Band("L", adc_sample_rate=1712e6, centre_frequency=1284e6)
    bandwidth = band.adc_sample_rate * 0.5 / narrowband_decimation

    max_pass_bandwidth = 0.7 * bandwidth
    # Use a power-of-two MHz bandwidth, since that's what VLBI
    # observations generally use.
    pass_bandwidth = 1e6
    while pass_bandwidth * 2 <= max_pass_bandwidth:
        pass_bandwidth *= 2
    pass_fraction = pass_bandwidth / bandwidth
    lo = math.ceil(n_channels * 0.5 * (1.0 - pass_fraction))
    # + 1 because this is an exclusive range
    hi = math.floor(n_channels * 0.5 * (1.0 + pass_fraction)) + 1
    return slice(lo, hi)


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
        bandwidth: float,
        samples_per_frame: int,
        multicast_groups: list[Endpoint],
        sock: socket.socket | None = None,
        reorder_window: int = 32,
        scale_factor_timestamp: float = 1.0,
        vlbi_delay: float = 0.0,
        steady_state_timestamp: int = 0,
        sync_time: float = 0.0,
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

        self.scale_factor_timestamp: float = scale_factor_timestamp
        self.bandwidth: float = bandwidth
        self.n_samples_per_frame: int = samples_per_frame
        self.frame_rate = round(self.bandwidth / self.n_samples_per_frame)
        self.sync_time: float = sync_time
        self.sample_bytes_per_frame = self.n_samples_per_frame * self.veng_out_bits_per_sample // 8
        packet_size = self.sample_bytes_per_frame + VDIF_HEADER_SIZE + self.VTP_HEADER_SIZE
        self._packet = bytearray(packet_size)
        # Number of times we've received a packet without sleeping
        self._sync_packets = 0

        if sock is None:
            logger.info("Binding to port %d", port)
            logger.info("Multicast groups: %s", self.multicast_groups)
            logger.info("Interface address: %s", interface_address)
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
        self.listening = True
        self.vlbi_delay = vlbi_delay
        self.steady_state_timestamp = steady_state_timestamp

    async def _next_packet(self) -> None:
        if not self.listening:
            packet_size = await asyncio.get_event_loop().sock_recv_into(self.sock, self._packet)

            return
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
        logger.debug("ref_time: %s", ref_time)
        logger.debug("min_time: %s", min_time)
        to_wait_sec = (min_time - ref_time).sec
        logger.debug("to_wait_sec: %s", to_wait_sec)
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

        min_time += TimeDelta(self.vlbi_delay, format="sec", scale="tai") + self.filter_shift

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
                                    logger.debug("min frame: %d, frame0_linear: %d", min_frame, frame0_linear)
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

    async def discard_pending(self) -> None:
        """Read and drop socket data until this task is cancelled."""
        while True:
            await asyncio.get_event_loop().sock_recv_into(self.sock, self._packet)
            self.buffer.clear()
            self.min_seq_id = 0
            self.last_frameset_timestamp = None

    def pause(self) -> None:
        """Pause the receiver."""
        self.listening = False

    def resume(self) -> None:
        """Resume the receiver."""
        self.last_frameset_timestamp = None
        self.buffer.clear()
        self.listening = True

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
    parser.add_argument("product_name", type=str, help="Name of the subarray to get baselines from.")
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


NP_DTYPE = np.int8
NP_DTYPE_SIZE_BITS = np.dtype(NP_DTYPE).itemsize * 8
BYTES_PER_NP_ARRAY_FOR_FRAME = SAMPLES_PER_FRAME * (NP_DTYPE_SIZE_BITS // 8)  # The size of the numpy array for a frame.


def create_identity_data(samples_per_frame: int) -> np.ndarray:
    """Create an identity data array use as payload for a VDIF frame.

    Because the VDIF frames are low bit quantization, we need to have an identity matrix with data sufficiently far from
    another for the difference to be visible.
    """
    identity_size = math.ceil(math.sqrt(samples_per_frame))
    # even though we only have 2bits per sample, np.array can do minimum 8 bits per sample
    identity_data = np.identity(identity_size, dtype=NP_DTYPE)
    # the identity matrix size is rounded up to the nearest square number, correct it.
    identity_data = identity_data.reshape(-1, 1)[:samples_per_frame]
    # Quantization of -1 and 1 to the 2bit data samples can be reversed to get the original values.
    data = (np.ones_like(identity_data) * -1) + identity_data * 2
    data = data.reshape(-1, 1)[:samples_per_frame]
    assert data.shape == (SAMPLES_PER_FRAME, 1)
    return data


def create_noisy_sine_data(samples_per_frame: int) -> np.ndarray:
    """Create an identity data array use as payload for a VDIF frame.

    Because the VDIF frames are low bit quantization, we need to have an identity matrix with data sufficiently far from
    another for the difference to be visible.
    """
    identity_size = math.ceil(math.sqrt(samples_per_frame))
    # even though we only have 2bits per sample, np.array can do minimum 8 bits per sample
    identity_data = np.identity(identity_size, dtype=NP_DTYPE)
    # the identity matrix size is rounded up to the nearest square number, correct it.
    identity_data = identity_data.reshape(-1, 1)[:samples_per_frame]
    # Quantization of -1 and 1 to the 2bit data samples can be reversed to get the original values.
    # data = ((np.ones_like(identity_data) * -1) + identity_data * 2) + window_sample_time
    data = phase_signal_quantised_sin
    data = data.reshape(-1, 1)[:samples_per_frame]

    assert data.shape == (SAMPLES_PER_FRAME, 1)
    return data


def create_noisy_sine_data_offset_phase(samples_per_frame: int, offset_phase: float) -> np.ndarray:
    """Create an identity data array use as payload for a VDIF frame.

    Because the VDIF frames are low bit quantization, we need to have an identity matrix with data sufficiently far from
    another for the difference to be visible.
    """
    identity_size = math.ceil(math.sqrt(samples_per_frame))
    # even though we only have 2bits per sample, np.array can do minimum 8 bits per sample
    identity_data = np.identity(identity_size, dtype=NP_DTYPE)
    # the identity matrix size is rounded up to the nearest square number, correct it.
    identity_data = identity_data.reshape(-1, 1)[:samples_per_frame]
    # Quantization of -1 and 1 to the 2bit data samples can be reversed to get the original values.
    # data = ((np.ones_like(identity_data) * -1) + identity_data * 2) + window_sample_time
    # TODO: use the offset phase as a factor of the frequency to determine
    # the correct number of samples to shift. theta = 2 * np.pi * frequency * time
    data = np.roll(phase_signal_quantised_sin, 5)
    data = data.reshape(-1, 1)[:samples_per_frame]

    assert data.shape == (SAMPLES_PER_FRAME, 1)
    return data


def create_dirac_data(samples_per_frame: int, index: int) -> np.ndarray:
    """Create a dirac data array use as payload for a VDIF frame."""
    data = np.zeros((samples_per_frame, 1), dtype=NP_DTYPE) - 1
    data[index] = 1
    assert data.shape == (SAMPLES_PER_FRAME, 1)
    return data


default_data = np.zeros((SAMPLES_PER_FRAME, 1), dtype=NP_DTYPE)


def make_vtp_packet(
    seq_id: int,
    frame_nr: int,
    seconds: int,
    thread_id: int,
    samples_per_frame: int = SAMPLES_PER_FRAME,
    ref_epoch: int = 0,
    data=default_data,
) -> bytes:
    """Build a VTP packet (8-byte seq header + VDIF frame)."""
    # assert data.nbytes == BYTES_PER_NP_ARRAY_FOR_FRAME
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
    def __init__(self, data_fnc=create_noisy_sine_data):
        self.seq_id = 0
        self.frame_nr = 0
        self.seconds = 1
        self.thread_id = 0
        self.data_fnc = data_fnc

    def recv_into(self, buf: Buffer) -> int:
        mv = memoryview(buf).cast("B")
        if self.thread_id > 3:
            self.thread_id = 0
            self.frame_nr += 1
        if self.frame_nr >= 8000:
            self.frame_nr = 0
            self.seconds += 1
            logger.debug("Generating second %d", self.seconds)

        packet = make_vtp_packet(
            self.seq_id, self.frame_nr, self.seconds, self.thread_id, data=self.data_fnc(SAMPLES_PER_FRAME)
        )
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
                return await client.sensor_value(name, sensor_type)

            # Spead2 doesn't know katsdptelstate so it can't recognise Endpoints.
            # But we can cast Endpoints to tuples, which it does know.
            multicast_groups = [
                endpoint
                for endpoint in endpoint_list_parser(DEFAULT_VTP_PORT)(
                    await sensor_val("tied-array-resampled-voltage.destination", str)
                )
            ]

            bandwidth = float(await sensor_val("tied-array-resampled-voltage.bandwidth", float))
            samples_per_frame = int(await sensor_val("tied-array-resampled-voltage.n-samples-per-frame", int))

            receiver = TiedArrayResampledVoltageReceiver(
                stream_name="tied-array-resampled-voltage",
                interface_address=args.interface,
                bandwidth=bandwidth,
                samples_per_frame=samples_per_frame,
                multicast_groups=multicast_groups,
                scale_factor_timestamp=float(
                    await sensor_val("tied-array-resampled-voltage.scale-factor-timestamp", float)
                ),
                vlbi_delay=float(await sensor_val("tied-array-resampled-voltage.delay", float)),
                steady_state_timestamp=int(await sensor_val("sim.m800.1712000000.0.steady-state-timestamp", int)),
                sync_time=float(await sensor_val("tied-array-resampled-voltage.sync-time", float)),
            )

    else:
        receiver = TiedArrayResampledVoltageReceiver(
            stream_name="tied-array-resampled-voltage",
            interface_address=args.interface,
            bandwidth=int(64e6),
            samples_per_frame=8000,
            multicast_groups=[Endpoint("127.0.0.1", 5001)],
            # sock=FakeSocket(data_fnc=lambda x: create_dirac_data(x, 1)),
            # sock=FakeSocket(data_fnc=create_noisy_sine_data),
            sock=FakeSocket(data_fnc=lambda x: create_noisy_sine_data_offset_phase(x, 180)),
        )

    # X and Y polarisations, freq & phase for frequency domain,
    # voltage and analysis signals for time domain, ~16 plots in total.
    fig, axs = plt.subplots(
        receiver.n_chans,
        2 * 4,
        sharex="col",
        squeeze=False,
        figsize=(12, 8) if args.interactive else (24, 16),
        constrained_layout=True,
    )
    axs = axs.ravel()  # Now that the plot is several columns wide, this comes as a 2-d array. Which is a pain.
    axs = axs[:16]  # If there are gaps at the end, ignore them
    lines = []
    bandpass_n_channels = 32768
    target_fft_n_channels = 200
    frequency_sample_size_frames = math.ceil(target_fft_n_channels * 2 / receiver.n_samples_per_frame)
    n_fft = receiver.n_samples_per_frame * frequency_sample_size_frames
    assert n_fft == 8000
    timedata = np.zeros((4, n_fft))
    narrowband_decimation = 8
    band = Band("L", adc_sample_rate=1712e6, centre_frequency=1284e6)
    nb_bandwidth = band.adc_sample_rate * 0.5 / narrowband_decimation
    pass_channels = get_pass_channels(narrowband_decimation, bandpass_n_channels)
    centre = band.centre_frequency
    baseband = np.fft.rfftfreq(n_fft, d=1.0 / receiver.bandwidth)
    x_by_time = np.arange(n_fft) * SAMPLE_DELTA

    # Bin 0 is the outer edge of a half, Nyquist is the centre.
    x_lower_all = centre - receiver.bandwidth / 2 + baseband
    x_upper_all = centre + baseband
    if ZOOM:
        f_pass_lo = (pass_channels.start - bandpass_n_channels / 2) * nb_bandwidth / bandpass_n_channels + centre
        f_pass_hi = (pass_channels.stop - bandpass_n_channels / 2) * nb_bandwidth / bandpass_n_channels + centre
        lower_bins = np.flatnonzero((x_lower_all >= f_pass_lo) & (x_lower_all < centre))
        upper_bins = np.flatnonzero((x_upper_all >= centre) & (x_upper_all < f_pass_hi))
        x_by_thread = (
            x_upper_all[upper_bins],
            x_upper_all[upper_bins],
            x_lower_all[lower_bins],
            x_lower_all[lower_bins],
        )
        bins_by_thread = (upper_bins, upper_bins, lower_bins, lower_bins)
    else:
        x_by_thread = (baseband, baseband, baseband, baseband)
        bins_by_thread = (
            np.arange(baseband.size),
            np.arange(baseband.size),
            np.arange(baseband.size),
            np.arange(baseband.size),
        )
    fig_order = {
        0: "X Upper",
        1: "Y Upper",
        2: "X Lower",
        3: "Y Lower",
    }
    for i in range(4):
        lines.append(axs[i].plot(x_by_thread[i], np.zeros(x_by_thread[i].shape))[0])
        axs[i].set_ylabel(f"{i}'s magnitude (dB)")
        axs[i].set_xlabel(f"{fig_order[i]} half frequency (Hz)")
    for i in range(4):
        lines.append(axs[i + 8].plot(x_by_thread[i], np.zeros(x_by_thread[i].shape))[0])
        axs[i + 8].set_ylabel(f"{i}'s phase (rad)")
        axs[i + 8].set_xlabel(f"{fig_order[i]} half frequency (Hz)")
    for i in range(4):
        lines.append(axs[i + 4].plot(x_by_time, np.zeros(x_by_time.shape))[0])
        axs[i + 4].set_ylabel(f"{i}'s time domain voltage")
        axs[i + 4].set_xlabel("time (s)")
    axs[12].set_title("sin multplication")
    lines.append(axs[12].plot(x_by_time, np.zeros(x_by_time.shape))[0])
    axs[13].set_title("cos multplication")
    lines.append(axs[13].plot(x_by_time, np.zeros(x_by_time.shape))[0])
    axs[14].set_title("zz")
    lines.append(axs[14].plot(x_by_time, np.zeros(x_by_time.shape))[0])
    if args.interactive:
        fig.canvas.mpl_connect("close_event", lambda event: receiver.close())
        plt.ion()
        plt.show()

    logger.info("Bins and frequencies:")
    logger.info("N FFT: %s", n_fft)
    logger.info("Frequency sample size frames: %s", frequency_sample_size_frames)
    logger.info("Target FFT n channels: %s", target_fft_n_channels)
    logger.info("Bandpass n channels: %s", bandpass_n_channels)
    logger.info("Narrowband decimation: %s", narrowband_decimation)
    logger.info("Band: %s", band)
    logger.info("NB bandwidth: %s", nb_bandwidth)
    logger.info("Pass channels: %s", pass_channels)
    logger.info("Centre: %s", centre)
    await asyncio.sleep(4)
    logger.info("Starting to receive framesets")
    plt.title("VDIF FFTs")
    counter = 0
    integration_factor = 1 * SAMPLE_DELTA

    min_timestamp = None
    while True:
        chunk = await anext(receiver.complete_framesets(min_timestamp=min_timestamp))
        if receiver.steady_state_timestamp == 0:
            logger.info("Steady state timestamp is zero")
        for frame in chunk.frames:
            vdif_frame = BasebandVDIFFrame.fromfile(io.BytesIO(frame.raw_frame))
            timedata[frame.thread_id][
                counter * receiver.n_samples_per_frame : (counter + 1) * receiver.n_samples_per_frame
            ] = vdif_frame.data.reshape(-1)
        counter += 1
        if counter == frequency_sample_size_frames:
            # timedata = np.zeros((4, n_fft))
            # for i in range(4):
            #     timedata[i][1] = 1
            # update line
            # this takes a while so let's pause the receiver
            logger.info("Generating Image")
            logger.debug("Timedata: %s", timedata[0][0])
            logger.debug("Timedata: %s", timedata[0][1])
            logger.debug("Timedata: %s", timedata[0][7999])
            logger.debug("Min Timestamp: %s", min_timestamp)

            # do a fourier transform of the time domain data here and plot the magnitude and phase
            s_phase = np.zeros(4)
            for i in range(4):
                fft_data = np.fft.rfft(timedata[i])[bins_by_thread[i]]
                if CUTOFF:
                    threshold = np.max(np.abs(fft_data)) / 10
                    fft_data[np.abs(fft_data) < threshold] = 0
                # as dBm
                magnitude = 10 * np.log10(np.abs(fft_data))
                phase = np.angle(fft_data)
                # imag = np.imag(fft_data)
                # get the phase via time domain data instead? https://dsp.stackexchange.com/questions/75907/signal-processing-techniques-for-accurate-phase-measurement
                # p_tmp = phase[np.abs(x_by_thread[i] - PHASE_FREQ) < 10000]

                theta = np.acos(np.sum(timedata[i] * phase_signal) * integration_factor)
                s_phase[i] = theta
                # if p_tmp.size > 0:
                #     s_phase[i] = p_tmp[len(p_tmp) // 2]
                # else:
                #     s_phase[i] = np.nan
                lines[i].set_data(x_by_thread[i], magnitude)
                lines[i + 4].set_data(x_by_thread[i], phase)

                lines[i + 8].set_data(x_by_time, timedata[i])
                if i == 0:
                    _q = timedata[i] * phase_signal_quantised_sin
                    lines[12].set_data(x_by_time, _q)
                    axs[12].relim()
                    axs[12].autoscale_view()
                    _i = timedata[i] * phase_signal_quantised_cos
                    lines[13].set_data(x_by_time, _i)
                    axs[13].relim()
                    axs[13].autoscale_view()
                    # Summing I and Q is a boxcar lowpass of bandwidth fs/N
                    # (8 kHz for 64 Msps and 8000 samples). That rejects the
                    # double-frequency image and leaves one phase.
                    theta = np.atan2(np.sum(_q), np.sum(_i))
                    logger.info("Theta for thread %s: %.2f deg", i, np.degrees(theta))
                    logger.info("----------------------------------")
                axs[i].relim()
                axs[i].autoscale_view()
                axs[i + 8].relim()
                axs[i + 8].autoscale_view()
                axs[i + 4].relim()
                axs[i + 4].autoscale_view()
            # plt.suptitle(f"VDIF FFTs ({PHASE_FREQ} phase is {', '.join(f'{s_p:.2f} rad' for s_p in s_phase)})")
            plt.suptitle(f"VDIF FFTs ({PHASE_FREQ} phase is in phase with the synthetic signal)")

            if not args.interactive:
                plt.savefig("rep.png")
            logger.info("Wrote second %d", chunk.timestamp.seconds)
            counter = 0
            time.sleep(2)
            if args.fake_data:
                continue
            while True:
                client = await aiokatcp.Client.connect(host, port)
                async with client:
                    new_min_timestamp = await client.sensor_value("sim.m800.1712000000.0.steady-state-timestamp", int)
                    if new_min_timestamp != min_timestamp:
                        min_timestamp = new_min_timestamp
                        logger.info("New min timestamp: %s", min_timestamp)
                        break
                logger.info("Waiting for sync signal")
                await asyncio.sleep(0)


if __name__ == "__main__":
    main()
