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
import scipy.signal
from astropy.time import Time, TimeDelta
from baseband.vdif import VDIFFrame as BasebandVDIFFrame
from katsdpservices import get_interface_address
from katsdptelstate.endpoint import Endpoint, endpoint_list_parser, endpoint_parser

from katgpucbf import DEFAULT_VTP_PORT
from katgpucbf.utils import TimeConverter

CPLX = 2
CORR2_SENSOR_REMAP = {
    "antenna-channelised-voltage.adc-sample-rate": "adc-sample-rate",
}
# According to the ICD.
logger = logging.getLogger(__name__)
CUTOFF = False
ZOOM = True
PHASE_FREQ = 1290e6
SYNC = True
SCAN_OFFSET = 8
BANDPASS_HALF_WIDTH = 10e3
FAKE_DATA_REF_EPOCH = 35


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

    def sync_time_for(self, min_timestamp: int | None) -> Time:
        """Absolute time of the first sample that should be kept for `min_timestamp`."""
        if min_timestamp is None:
            min_timestamp = self.steady_state_timestamp
        min_time = Time(self.sync_time, format="unix", scale="utc") + TimeDelta(
            min_timestamp / self.scale_factor_timestamp, format="sec", scale="tai"
        )
        return min_time + TimeDelta(self.vlbi_delay, format="sec", scale="tai") + self.filter_shift

    def _calc_min_frame(self, min_time: Time, ref_epoch: int) -> int:
        """First frame that still contains samples at or after `min_time`.

        The frame that contains the sync starts before it. Floor keeps that frame so the
        caller can drop only the leading samples, instead of ceil discarding the whole frame.
        """
        ref_time = VDIFTimestamp(0, 0, ref_epoch, frame_rate=self.frame_rate).timestamp
        logger.debug("ref_time: %s", ref_time)
        logger.debug("min_time: %s", min_time)
        to_wait_sec = (min_time - ref_time).sec
        logger.debug("to_wait_sec: %s", to_wait_sec)
        return math.floor(to_wait_sec * self.frame_rate)

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
        sync_time = self.sync_time_for(min_timestamp)

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
                            min_frame = self._calc_min_frame(sync_time, frame0.timestamp.ref_epoch)
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
    parser.add_argument(
        "--n-channels",
        type=int,
        default=None,
        help="Number of channels to use for the FFT. If not provided, the default is 20000.",
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


def samples_before_sync(frame_start: Time, sync_time: Time, sample_rate: float, n_samples: int) -> int:
    """How many leading samples in a frame fall before `sync_time`.

    A sample that lands on the sync is kept. The result is clamped to the frame.
    """
    delta_sec = (sync_time - frame_start).sec
    if delta_sec <= 0:
        return 0
    # A fraction of a sample of slack stops float rounding from dropping the sync sample.
    skip = math.ceil(delta_sec * sample_rate - 1e-4)
    if skip <= 0:
        return 0
    return min(skip, n_samples)


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
    assert data.shape == (samples_per_frame, 1)
    return data


def quantise_2bit(values: np.ndarray) -> np.ndarray:
    inverse_threshold = 1.0 / 0.969
    return np.clip(values * inverse_threshold + 2, 0, 3)


def generate_time_data(
    samples_per_frame: int,
    n_frames: int,
    sample_rate: float,
    phase_freq: float,
    *,
    scan_offset: int = 0,
    bandpass_half_width: float = BANDPASS_HALF_WIDTH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    window_sample_time = np.arange(samples_per_frame * n_frames) / sample_rate
    phase_signal_quantised_sin = quantise_2bit(np.sin(2 * math.pi * phase_freq * window_sample_time))
    phase_signal_quantised_cos = quantise_2bit(np.cos(2 * math.pi * phase_freq * window_sample_time))

    phase_signal_quantised_sin = np.roll(phase_signal_quantised_sin, scan_offset)
    phase_signal_quantised_cos = np.roll(phase_signal_quantised_cos, scan_offset)

    phase_baseband = abs(phase_freq) % sample_rate
    if phase_baseband > sample_rate / 2:
        phase_baseband = sample_rate - phase_baseband
    phase_bandpass_sos = scipy.signal.butter(
        4,
        [phase_baseband - bandpass_half_width, phase_baseband + bandpass_half_width],
        btype="bandpass",
        fs=sample_rate,
        output="sos",
    )
    return window_sample_time, phase_signal_quantised_sin, phase_signal_quantised_cos, phase_bandpass_sos


def create_noisy_sine_data_offset_phase(
    samples_per_frame: int,
    offset_phase: float,
    n_window_samples: int,
    sample_rate: float,
    *,
    scan_offset: int = 0,
) -> np.ndarray:
    window_sample_time = np.arange(n_window_samples) / sample_rate
    data = np.sin(2 * math.pi * PHASE_FREQ * window_sample_time + math.radians(offset_phase))
    data + (np.random.normal(0, 3, data.size) - 1.5)
    if scan_offset:
        data = np.roll(data, scan_offset)
    data = quantise_2bit(data).reshape(-1, 1)[:samples_per_frame]

    assert data.shape == (samples_per_frame, 1)
    return data


def create_dirac_data(samples_per_frame: int, index: int) -> np.ndarray:
    """Create a dirac data array use as payload for a VDIF frame."""
    data = np.zeros((samples_per_frame, 1), dtype=NP_DTYPE) - 1
    data[index] = 1
    assert data.shape == (samples_per_frame, 1)
    return data


def make_vtp_packet(
    seq_id: int,
    frame_nr: int,
    seconds: int,
    thread_id: int,
    samples_per_frame: int,
    ref_epoch: int = FAKE_DATA_REF_EPOCH,
    data: np.ndarray | None = None,
) -> bytes:
    """Build a VTP packet (8-byte seq header + VDIF frame)."""
    if data is None:
        data = np.zeros((samples_per_frame, 1), dtype=NP_DTYPE)
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
    def __init__(self, samples_per_frame: int, data_fnc=None):
        self.seq_id = 0
        self.frame_nr = 0
        self.seconds = 0
        self.thread_id = 0
        self.samples_per_frame = samples_per_frame
        self.data_fnc = data_fnc

    def recv_into(self, buf: Buffer) -> int:
        mv = memoryview(buf).cast("B")
        if self.thread_id > 3:
            self.thread_id = 0
            self.frame_nr += 1
        if self.frame_nr >= 3200:  # TODO: parametrize frames per second
            self.frame_nr = 0
            self.seconds += 1
            logger.debug("Generating second %d", self.seconds)

        packet = make_vtp_packet(
            self.seq_id,
            self.frame_nr,
            self.seconds,
            self.thread_id,
            self.samples_per_frame,
            data=self.data_fnc(self.samples_per_frame),
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
        samples_per_frame = 20000
        receiver = TiedArrayResampledVoltageReceiver(
            stream_name="tied-array-resampled-voltage",
            interface_address=args.interface,
            bandwidth=int(64e6),
            samples_per_frame=samples_per_frame,
            multicast_groups=[Endpoint("127.0.0.1", 5001)],
            scale_factor_timestamp=64e6,
            sock=FakeSocket(samples_per_frame),
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
    lines = [[None] * 8, [None] * 8]
    bandpass_n_channels = 32768
    target_fft_n_channels = args.n_channels or 20000
    samples_per_frame = receiver.n_samples_per_frame
    frequency_sample_size_frames = math.ceil(target_fft_n_channels * 2 / samples_per_frame)
    logger.info("Samples per frame: %s", samples_per_frame)
    logger.info("Frame rate: %s", receiver.frame_rate)
    # One real sample per output sample: rate is frames/s times samples/frame,
    # which is the tied-array-resampled-voltage bandwidth from the client.
    sample_rate = receiver.frame_rate * samples_per_frame
    logger.info("Sample rate: %s", sample_rate)
    (
        window_sample_time,
        phase_signal_quantised_sin,
        phase_signal_quantised_cos,
        phase_bandpass_sos,
    ) = generate_time_data(
        samples_per_frame, frequency_sample_size_frames, sample_rate, PHASE_FREQ, scan_offset=SCAN_OFFSET
    )
    n_fft = window_sample_time.size
    logger.info("N FFT: %s", n_fft)
    if args.fake_data:
        # sock=FakeSocket(data_fnc=lambda x: create_dirac_data(x, 1)),
        # sock=FakeSocket(data_fnc=create_noisy_sine_data),
        receiver.sock.data_fnc = lambda n_samples: create_noisy_sine_data_offset_phase(
            n_samples,
            -90,
            phase_signal_quantised_sin.size,
            sample_rate,
            scan_offset=SCAN_OFFSET,
        )
    timedata = np.zeros((4, n_fft))
    narrowband_decimation = 8
    band = Band("L", adc_sample_rate=1712e6, centre_frequency=1284e6)
    nb_bandwidth = band.adc_sample_rate * 0.5 / narrowband_decimation
    pass_channels = get_pass_channels(narrowband_decimation, bandpass_n_channels)
    centre = band.centre_frequency
    baseband = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    x_by_time = window_sample_time

    # Bin 0 is the outer edge of a half, Nyquist is the centre.
    x_lower_all = centre - receiver.bandwidth / 2 + baseband
    x_upper_all = centre + receiver.bandwidth / 2 - baseband
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
        lines[0][i] = axs[0][i].plot(x_by_thread[i], np.zeros(x_by_thread[i].shape))[0]
        axs[0][i].set_ylabel(f"{i}'s magnitude (dB)")
        axs[0][i].set_xlabel(f"{fig_order[i]} half frequency (Hz)")

        lines[0][i + 4] = axs[0][i + 4].plot(x_by_time, np.zeros(x_by_time.shape))[0]
        axs[0][i + 4].set_ylabel(f"{i}'s time domain voltage after bandpass filter")
        axs[0][i + 4].set_xlabel("time (s)")

        lines[1][i] = axs[1][i].plot(x_by_thread[i], np.zeros(x_by_thread[i].shape))[0]
        axs[1][i].set_ylabel(f"{i}'s phase (rad)")
        axs[1][i].set_xlabel(f"{fig_order[i]} half frequency (Hz)")

    axs[1][4].set_title("sin reference")
    axs[1][4].set_xlabel("time (s)")
    lines[1][4] = axs[1][4].plot(x_by_time, phase_signal_quantised_sin)[0]
    axs[1][5].set_title("cos reference")
    axs[1][5].set_xlabel("time (s)")
    lines[1][5] = axs[1][5].plot(x_by_time, phase_signal_quantised_cos)[0]

    axs[1][6].set_title("sin multplication")
    axs[1][6].set_xlabel("time (s)")
    lines[1][6] = axs[1][6].plot(x_by_time, np.zeros(x_by_time.shape))[0]
    axs[1][7].set_title("cos multplication")
    axs[1][7].set_xlabel("time (s)")
    lines[1][7] = axs[1][7].plot(x_by_time, np.zeros(x_by_time.shape))[0]

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
    counter = 0
    filled = 0
    frame_samples = receiver.n_samples_per_frame
    if args.fake_data and frequency_sample_size_frames > 1:
        raise ValueError("Frequency sample size frames must be 1 for fake data")

    min_timestamp = None
    if args.fake_data:
        vdif_time = (
            VDIFTimestamp(
                seconds=0, ref_epoch=FAKE_DATA_REF_EPOCH, frame_nr=0, frame_rate=receiver.frame_rate
            ).timestamp
            + TimeDelta(1 / sample_rate, format="sec", scale="tai") * 5
        )
        min_timestamp = round(
            TimeConverter(receiver.sync_time, receiver.scale_factor_timestamp).unix_to_adc(vdif_time.unix)
        )

    while True:
        chunk = await anext(receiver.complete_framesets(min_timestamp=min_timestamp))
        if receiver.steady_state_timestamp == 0:
            logger.info("Steady state timestamp is zero")
        skip = samples_before_sync(
            chunk.timestamp.timestamp, receiver.sync_time_for(min_timestamp), sample_rate, frame_samples
        )
        take = min(frame_samples - skip, n_fft - filled)
        if skip and take:
            logger.info("Skipping %d samples before sync in frame %d", skip, chunk.timestamp.linear)
        if take > 0:
            for frame in chunk.frames:
                vdif_frame = BasebandVDIFFrame.fromfile(io.BytesIO(frame.raw_frame))
                samples = np.reshape(vdif_frame.data, -1)[skip : skip + take]
                timedata[frame.thread_id, filled : filled + take] = samples
            filled += take
            # A short leading fragment does not count as a frame on its own.
            counter = filled // frame_samples
        if counter == frequency_sample_size_frames:
            logger.info("Generating Image")
            logger.debug("Min Timestamp: %s", min_timestamp)

            # do a fourier transform of the time domain data here and plot the magnitude and phase
            for i in range(4):
                fft_data = np.fft.rfft(timedata[i])[bins_by_thread[i]]
                if CUTOFF:
                    threshold = np.max(np.abs(fft_data)) / 10
                    fft_data[np.abs(fft_data) < threshold] = 0
                # as dBm
                magnitude = 10 * np.log10(np.abs(fft_data))
                phase = np.angle(fft_data)
                # get the phase via time domain data instead? https://dsp.stackexchange.com/questions/75907/signal-processing-techniques-for-accurate-phase-measurement
                lines[0][i].set_data(x_by_thread[i], magnitude)
                lines[1][i].set_data(x_by_thread[i], phase)
                filtered = scipy.signal.sosfiltfilt(phase_bandpass_sos, timedata[i])
                lines[0][i + 4].set_data(x_by_time, filtered)
                if i == 0:
                    _q = filtered * phase_signal_quantised_sin
                    lines[1][6].set_data(x_by_time, _q)
                    axs[1][6].relim()
                    axs[1][6].autoscale_view()

                    _i = filtered * phase_signal_quantised_cos
                    lines[1][7].set_data(x_by_time, _i)
                    axs[1][7].relim()
                    axs[1][7].autoscale_view()

                    theta = np.atan2(np.sum(_i), np.sum(_q))
                    logger.info("Theta for thread %s: %.2f deg", i, np.degrees(theta))
                    logger.info("----------------------------------")
                axs[0][i].relim()
                axs[0][i].autoscale_view()
                axs[1][i].relim()
                axs[1][i].autoscale_view()
                axs[0][i + 4].relim()
                axs[0][i + 4].autoscale_view()
            # plt.suptitle(f"VDIF FFTs ({PHASE_FREQ} phase is {', '.join(f'{s_p:.2f} rad' for s_p in s_phase)})")
            plt.suptitle(f"VDIF FFTs ({PHASE_FREQ} phase is in phase with the synthetic signal)")

            if not args.interactive:
                plt.savefig("rep.png")
            logger.info("Wrote second %d", chunk.timestamp.seconds)
            counter = 0
            filled = 0
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
                await asyncio.sleep(1)


if __name__ == "__main__":
    main()
