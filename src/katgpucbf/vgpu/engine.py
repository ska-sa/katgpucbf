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

import aiokatcp

from ..utils import Engine


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
        recv_pols: tuple[str, str],
        send_pols: tuple[str, str],
    ) -> None:
        super().__init__(katcp_host, katcp_port)
        self.recv_pols = recv_pols
        self.send_pols = send_pols
        self._populate_sensors(self.sensors, send_pols)

    def _populate_sensors(self, sensors: aiokatcp.SensorSet, send_pols: tuple[str, str]) -> None:
        """Define the sensors for the engine."""
        for pol in self.send_pols:
            for channel in range(2):
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
