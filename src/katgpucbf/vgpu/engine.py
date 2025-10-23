################################################################################
# Copyright (c) 2025, National Research Foundation (SARAO)
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
from collections.abc import Sequence

import aiokatcp
import cupyx
import numpy as np
import spead2.recv.asyncio

from .. import COMPLEX, N_POLS, RECV_TASK_NAME
from .. import recv as base_recv
from ..monitor import Monitor
from ..recv import RECV_SENSOR_TIMEOUT_CHUNKS, RECV_SENSOR_TIMEOUT_MIN
from ..ringbuffer import ChunkRingbuffer
from ..utils import Engine, TimeConverter
from . import N_SIDEBANDS, recv


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
        sync_time: float,
        adc_sample_rate: float,
        n_channels: int,
        n_channels_per_substream: int,
        n_spectra_per_heap: int,
        n_samples_between_spectra: int,
        n_batches_per_chunk: int,
        sample_bits: int,
        srcs: list[list[tuple[str, int]]],
        recv_interface: str | None,
        recv_ibv: bool,
        recv_affinity: int,
        recv_comp_vector: int,
        recv_buffer: int,
        recv_pols: tuple[str, str],
        send_pols: tuple[str, str],
        monitor: Monitor,
    ) -> None:
        super().__init__(katcp_host, katcp_port)

        self._srcs = srcs
        self._recv_interface = recv_interface
        self._recv_ibv = recv_ibv
        self._recv_comp_vector = recv_comp_vector
        self._recv_buffer = recv_buffer
        self.recv_pols = recv_pols
        self.recv_layout = recv.Layout(
            sample_bits=sample_bits,
            n_channels=n_channels,
            n_channels_per_substream=n_channels_per_substream,
            n_spectra_per_heap=n_spectra_per_heap,
            n_batches_per_chunk=n_batches_per_chunk,
            heap_timestamp_step=n_samples_between_spectra * n_spectra_per_heap,
        )
        self.recv_time_converter = TimeConverter(sync_time, adc_sample_rate)

        self.send_pols = send_pols

        recv_sensor_timeout = max(
            RECV_SENSOR_TIMEOUT_MIN,
            RECV_SENSOR_TIMEOUT_CHUNKS * self.recv_layout.chunk_timestamp_step / adc_sample_rate,
        )
        self._populate_sensors(self.sensors, self.recv_pol_labels, self.send_pols, recv_sensor_timeout)
        self._init_recv(recv_affinity, monitor)

    @property
    def recv_pol_labels(self) -> list[str]:
        """Incoming polarisations without any ± prefix."""
        return [pol[-1] for pol in self.recv_pols]

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
        prefixes = [f"{pol}." for pol in self.recv_pol_labels]
        for sensor in base_recv.make_sensors(recv_sensor_timeout, prefixes).values():
            sensors.add(sensor)

    def _init_recv(self, recv_affinity: int, monitor: Monitor) -> None:
        """Initialise the receive side of the engine."""
        recv_chunks = 4  # TODO: may need tuning?
        data_ringbuffer = ChunkRingbuffer(
            recv_chunks, name="recv_data_ringbuffer", task_name="run_receive", monitor=monitor
        )
        free_ringbuffer = spead2.recv.ChunkRingbuffer(recv_chunks)
        layout = self.recv_layout
        dtype = np.dtype(f"int{layout.sample_bits}")
        self._recv_group = recv.make_stream_group(
            layout, data_ringbuffer, free_ringbuffer, recv_affinity, self.recv_pol_labels
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
                sink=self._recv_group,
            )
            chunk.recycle()  # Adds to free_ringbuffer

    async def _run_receive(self) -> None:
        """Receive data from the tied-array-channelised-voltage streams."""
        for stream in self._recv_group:
            stream.start()
        data_ringbuffer = self._recv_group.data_ringbuffer
        assert isinstance(data_ringbuffer, spead2.recv.asyncio.ChunkRingbuffer)
        async for chunk in recv.iter_chunks(
            data_ringbuffer, self.recv_layout, self.sensors, self.recv_time_converter, self.recv_pol_labels
        ):
            with chunk:
                pass
        # TODO

    async def start(self) -> None:
        """Start the engine."""
        for i, stream in enumerate(self._recv_group):
            base_recv.add_reader(
                stream,
                src=self._srcs[i],
                interface=self._recv_interface,
                ibv=self._recv_ibv,
                comp_vector=self._recv_comp_vector,
                buffer=self._recv_buffer // len(self._recv_group),
            )

        recv_task = asyncio.create_task(self._run_receive(), name=RECV_TASK_NAME)
        self.add_service_task(recv_task, wait_on_stop=True)
        await super().start()

    async def on_stop(self) -> None:  # noqa: D102
        self._recv_group.stop()
        await super().on_stop()

    async def request_vlbi_delay(self, ctx: aiokatcp.RequestContext, delay: float) -> None:
        """Set the delay applied to the stream, in second."""
        # TODO: will need to be rounded/quantised
        self.sensors["delay"].value = delay

    async def request_capture_start(self, ctx: aiokatcp.RequestContext) -> None:
        """Start capturing and emitting data."""
        pass

    async def request_capture_stop(self, ctx: aiokatcp.RequestContext) -> None:
        """Stop capturing and emitting data."""
        pass
