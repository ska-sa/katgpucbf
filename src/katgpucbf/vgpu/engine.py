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

"""Engine class, which does all the actual processing."""

import asyncio
import logging
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from fractions import Fraction

import aiokatcp
import baseband
import cupy as cp
import cupyx
import katcbf_vlbi_resample.cupy_bridge
import katcbf_vlbi_resample.parameters
import katcbf_vlbi_resample.polarisation
import katcbf_vlbi_resample.power
import katcbf_vlbi_resample.rechunk
import katcbf_vlbi_resample.resample
import katcbf_vlbi_resample.stream
import katcbf_vlbi_resample.utils
import katcbf_vlbi_resample.vdif_writer
import numpy as np
import spead2.recv.asyncio
import xarray as xr
from astropy.time import Time

from .. import COMPLEX, N_POLS
from .. import recv as base_recv
from ..monitor import Monitor
from ..recv import RECV_SENSOR_TIMEOUT_CHUNKS, RECV_SENSOR_TIMEOUT_MIN
from ..ringbuffer import ChunkRingbuffer
from ..utils import Engine, TimeConverter
from . import N_SIDEBANDS, recv

logger = logging.getLogger(__name__)


class RecvStream:
    """Wrap the incoming data stream into a :class:`.katcbf_vlbi_resample.stream.Stream`."""

    def __init__(
        self,
        layout: recv.Layout,
        time_converter: TimeConverter,
        stream_group: spead2.recv.ChunkStreamRingGroup,
        sensors: aiokatcp.SensorSet,
        pol_labels: tuple[str, str],
    ) -> None:
        self._layout = layout
        self._time_converter = time_converter
        self._stream_group = stream_group
        self._sensors = sensors
        self._pol_labels = pol_labels
        self._samples_between_spectra = layout.heap_timestamp_step // layout.n_spectra_per_heap
        # Properties required by the Stream protocol
        self.channels = layout.n_channels
        self.is_cupy = True
        self.time_base = Time(time_converter.sync_time, scale="utc", format="unix")
        self.time_scale = Fraction(self._samples_between_spectra) / Fraction(time_converter.adc_sample_rate)

    async def __aiter__(self) -> AsyncIterator[xr.DataArray]:
        for stream in self._stream_group:
            stream.start()
        data_ringbuffer = self._stream_group.data_ringbuffer
        assert isinstance(data_ringbuffer, spead2.recv.asyncio.ChunkRingbuffer)
        last_chunk_id: int | None = None
        async for chunk in recv.iter_chunks(
            data_ringbuffer,
            self._layout,
            self._sensors,
            self._time_converter,
            [label[-1] for label in self._pol_labels],
        ):
            with chunk:
                # TODO: need to do something with the presence flags
                # TODO: pipeline these transfers (but keeping in mind
                # that we need to recycle the chunk only when the transfer
                # is complete).
                data = cp.asarray(chunk.data, blocking=False)
                await katcbf_vlbi_resample.utils.stream_future(None)
                # There are two time axes. Transpose to place them together, then flatten
                # over them.
                # (N_POLS, layout.n_batches_per_chunk, layout.n_channels, layout.n_spectra_per_heap, COMPLEX),
                data = data.transpose(0, 1, 3, 2, 4)
                data = data.reshape(N_POLS, -1, self.channels, COMPLEX)
                # Convert Gaussian integers to complex
                data = cp.ascontiguousarray(data.astype(np.float32)).view(np.complex64)[..., 0]
                arr = xr.DataArray(
                    data,
                    dims=("pol", "time", "channel"),
                    coords={"pol": list(self._pol_labels)},
                    attrs={"time_bias": chunk.timestamp // self._samples_between_spectra},
                )
                # TODO (NGC-1689): need to properly handle missing data in
                # katcbf-vlbi-resample. This is a quick hack to keep things
                # running by injecting zero data into
                while last_chunk_id is not None and last_chunk_id < chunk.chunk_id - 1:
                    last_chunk_id += 1
                    zero_arr = xr.zeros_like(arr)
                    timestamp = last_chunk_id * self._layout.chunk_timestamp_step
                    zero_arr.attrs["time_bias"] = timestamp // self._samples_between_spectra
                    yield zero_arr
                last_chunk_id = chunk.chunk_id
                yield arr


class RecordPower(katcbf_vlbi_resample.power.RecordPower):
    """Record power levels to sensors."""

    def __init__(self, *args, sensors: aiokatcp.SensorSet, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sensors = sensors

    def record_rms(self, start: int, length: int, rms: xr.DataArray) -> None:  # noqa: D102
        end = start + length
        end_time = self.time_base + katcbf_vlbi_resample.utils.fraction_to_time_delta(end * self.time_scale)
        end_time_unix = float(end_time.unix)
        power = rms**2
        for pol in power.coords["pol"].values:
            for sideband in power.coords["sideband"].values:
                channel = ["lsb", "usb"].index(sideband)
                sensor = self.sensors[f"{pol}{channel}.mean-power"]
                sensor.set_value(power.sel(pol=pol, sideband=sideband).item(), timestamp=end_time_unix)


@dataclass
class RecvConfig:
    """Container for all the configuration for receiving data."""

    sync_time: float
    adc_sample_rate: float
    n_channels: int
    n_channels_per_substream: int
    n_spectra_per_heap: int
    n_samples_between_spectra: int
    n_batches_per_chunk: int
    sample_bits: int
    srcs: list[list[tuple[str, int]]]
    interface: str | None
    ibv: bool
    affinity: int
    comp_vector: int
    buffer: int
    pols: tuple[str, str]

    @property
    def pol_labels(self) -> list[str]:
        """Incoming polarisations without any ± prefix."""
        return [pol[-1] for pol in self.pols]

    def __post_init__(self) -> None:
        self.layout = recv.Layout(
            sample_bits=self.sample_bits,
            n_channels=self.n_channels,
            n_channels_per_substream=self.n_channels_per_substream,
            n_spectra_per_heap=self.n_spectra_per_heap,
            n_batches_per_chunk=self.n_batches_per_chunk,
            heap_timestamp_step=self.n_samples_between_spectra * self.n_spectra_per_heap,
        )
        self.time_converter = TimeConverter(self.sync_time, self.adc_sample_rate)


@dataclass
class SendConfig:
    """Container for all the configuration for sending data."""

    pols: tuple[str, str]
    bandwidth: float
    n_samples_per_frame: int
    station: str


@dataclass
class CaptureConfig:
    """Container for all the configuration needed to run a capture session."""

    recv_config: RecvConfig
    send_config: SendConfig
    fir_taps: int
    hilbert_taps: int
    passband: float
    threshold: float
    power_int_time: int

    def __post_init__(self) -> None:
        self.pol_matrix = katcbf_vlbi_resample.polarisation.from_linear(self.send_config.pols)
        self.pol_matrix @= katcbf_vlbi_resample.polarisation.to_linear(self.recv_config.pols)
        self.resample_parameters = katcbf_vlbi_resample.parameters.ResampleParameters(
            fir_taps=self.fir_taps,
            hilbert_taps=self.hilbert_taps,
            passband=self.passband,
        )
        self.threads = [
            {"sideband": sideband, "pol": pol} for sideband in ["lsb", "usb"] for pol in self.send_config.pols
        ]


class _CaptureSession:
    """Manage the lifetime of actions between ``?capture-start`` and ``?capture-stop``."""

    def __init__(self, config: CaptureConfig, engine: Engine, monitor: Monitor) -> None:
        recv_chunks = 4  # TODO: may need tuning?
        data_ringbuffer = ChunkRingbuffer(recv_chunks, name="recv_data_ringbuffer", task_name="run", monitor=monitor)
        free_ringbuffer = spead2.recv.ChunkRingbuffer(recv_chunks)
        layout = config.recv_config.layout
        dtype = np.dtype(f"int{layout.sample_bits}")
        recv_group = recv.make_stream_group(
            layout, data_ringbuffer, free_ringbuffer, config.recv_config.affinity, config.recv_config.pol_labels
        )
        for _ in range(recv_chunks):
            chunk = recv.Chunk(
                present=np.empty(
                    (N_POLS, layout.n_batches_per_chunk, layout.n_pol_substreams),
                    np.uint8,
                ),
                data=cupyx.empty_pinned(
                    (N_POLS, layout.n_batches_per_chunk, layout.n_channels, layout.n_spectra_per_heap, COMPLEX),
                    dtype,
                ),
                sink=recv_group,
            )
            chunk.recycle()  # Make available to the stream

        for i, stream in enumerate(recv_group):
            base_recv.add_reader(
                stream,
                src=config.recv_config.srcs[i],
                interface=config.recv_config.interface,
                ibv=config.recv_config.ibv,
                comp_vector=config.recv_config.comp_vector,
                buffer_size=config.recv_config.buffer // len(recv_group),
            )

        self.config = config
        self._recv_group = recv_group
        self._sensors = engine.sensors
        self._capture_task = asyncio.create_task(self._capture(), name="capture")
        engine.add_service_task(self._capture_task, wait_on_stop=True)

    def _process_frameset(self, frameset: baseband.vdif.VDIFFrameSet) -> None:
        """Handle a received frameset.

        This function is only here temporarily so that it can be mocked out
        by unit tests to intercept the framesets. It can be removed once
        the infrastructure for transmitting framesets is in place.

        .. todo:: Remove this method once no longer needed by unit tests.
        """
        logger.debug("Received frameset: %s +%s", frameset["seconds"], frameset["frame_nr"])

    def _capture_complete(self) -> None:
        """Handle the end of all processing.

        This method exists only to mock from unit tests.

        .. todo:: Remove this method once no longer needed by unit tests.
        """
        pass

    async def _capture(self) -> None:
        """Do all the primary work of the engine.

        This is an asyncio task that runs as a service task of the device server.
        """
        # Copy some references just to make the code shorter
        config = self.config
        recv_config = config.recv_config
        send_config = config.send_config

        it: katcbf_vlbi_resample.stream.Stream[xr.DataArray] = RecvStream(
            recv_config.layout,
            recv_config.time_converter,
            self._recv_group,
            self._sensors,
            recv_config.pols,
        )
        it = katcbf_vlbi_resample.cupy_bridge.AsCupy(it)
        it = katcbf_vlbi_resample.resample.IFFT(it)
        it = katcbf_vlbi_resample.polarisation.ConvertPolarisation(
            it, config.pol_matrix, recv_config.pols, send_config.pols
        )
        it = katcbf_vlbi_resample.resample.Resample(send_config.bandwidth, 0.0, config.resample_parameters, it)
        it = katcbf_vlbi_resample.rechunk.Rechunk.align_utc_seconds(it)
        it_rms: katcbf_vlbi_resample.stream.Stream[xr.Dataset] = katcbf_vlbi_resample.power.MeasurePower(it)
        it_rms = RecordPower(it_rms, sensors=self._sensors)
        it = katcbf_vlbi_resample.power.NormalisePower(
            it_rms, baseband.base.encoding.TWO_BIT_1_SIGMA / config.threshold
        )
        it = katcbf_vlbi_resample.vdif_writer.VDIFEncode2Bit(it, samples_per_frame=send_config.n_samples_per_frame)
        it = katcbf_vlbi_resample.cupy_bridge.AsNumpy(it)
        frameset_it = katcbf_vlbi_resample.vdif_writer.VDIFFormatter(
            it, config.threads, station=send_config.station, samples_per_frame=send_config.n_samples_per_frame
        )
        async for frameset in frameset_it:
            self._process_frameset(frameset)
        self._capture_complete()

    async def stop(self) -> None:
        """Stop the capture."""
        self._recv_group.stop()
        await self._capture_task


class VEngine(Engine):
    """Top-level class running the whole thing."""

    # TODO: VERSION means interface version, rather than software version. It
    # will need to wait on a proper ICD for a release.
    VERSION = "katgpucbf-vgpu-icd-0.1"

    def __init__(
        self,
        *,
        katcp_host: str,
        katcp_port: int,
        config: CaptureConfig,
        monitor: Monitor,
    ) -> None:
        super().__init__(katcp_host, katcp_port)

        self.config = config
        self.monitor = monitor

        recv_config = config.recv_config
        send_config = config.send_config
        recv_sensor_timeout = max(
            RECV_SENSOR_TIMEOUT_MIN,
            RECV_SENSOR_TIMEOUT_CHUNKS * recv_config.layout.chunk_timestamp_step / recv_config.adc_sample_rate,
        )
        self._populate_sensors(self.sensors, recv_config.pol_labels, send_config.pols, recv_sensor_timeout)
        self._capture: _CaptureSession | None = None

    def _populate_sensors(
        self,
        sensors: aiokatcp.SensorSet,
        recv_pol_labels: Sequence[str],
        send_pols: Sequence[str],
        recv_sensor_timeout: float,
    ) -> None:
        """Define the sensors for the engine."""
        for pol in send_pols:
            for channel in range(N_SIDEBANDS):
                sensors.add(
                    aiokatcp.Sensor(
                        float,
                        f"{pol}{channel}.mean-power",
                        "Mean power over the previous interval of length power-int-time",
                    )
                )
        sensors.add(
            aiokatcp.Sensor(
                float,
                "delay",
                "Delay introduced by processing",
                units="s",
                default=0.0,
                initial_status=aiokatcp.Sensor.Status.NOMINAL,
            )
        )
        prefixes = [f"{pol}." for pol in recv_pol_labels]
        for sensor in base_recv.make_sensors(recv_sensor_timeout, prefixes).values():
            sensors.add(sensor)

    async def on_stop(self) -> None:  # noqa: D102
        if self._capture is not None:
            await self._stop_capture()
        await super().on_stop()

    async def request_vlbi_delay(self, ctx: aiokatcp.RequestContext, delay: float) -> None:
        """Set the delay applied to the stream, in second."""
        # TODO: will need to be rounded/quantised
        self.sensors["delay"].value = delay

    async def request_capture_start(self, ctx: aiokatcp.RequestContext, timestamp: int = 0) -> None:
        """Start capturing and emitting data.

        Parameters
        ----------
        timestamp
            Minimum ADC timestamp at which to enable emitting.
        """
        if self._capture is not None:
            raise aiokatcp.FailReply("a capture is already in progress")
        # TODO: use timestamp and delay
        self._capture = _CaptureSession(self.config, self, self.monitor)

    async def _stop_capture(self) -> None:
        assert self._capture is not None
        try:
            await self._capture.stop()
        finally:
            self._capture = None

    async def request_capture_stop(self, ctx: aiokatcp.RequestContext) -> None:
        """Stop capturing and emitting data."""
        if self._capture is None:
            raise aiokatcp.FailReply("no capture in progress")
        await self._stop_capture()
