################################################################################
# Copyright (c) 2020-2022, National Research Foundation (SARAO)
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

"""A collection of utility functions for katgpucbf."""

import ipaddress
import logging
import signal
from asyncio import get_event_loop
from enum import Enum
from typing import TypeVar

import aiokatcp
from katsdptelstate.endpoint import endpoint_list_parser

_T = TypeVar("_T")

logger = logging.getLogger(__name__)


def add_signal_handlers(server: aiokatcp.DeviceServer) -> None:
    """Arrange for clean shutdown on SIGINT (Ctrl-C) or SIGTERM."""
    signums = [signal.SIGINT, signal.SIGTERM]

    def handler():
        # Remove the handlers so that if it fails to shut down, the next
        # attempt will try harder.
        logger.info("Received signal, shutting down")
        for signum in signums:
            loop.remove_signal_handler(signum)
        server.halt()

    loop = get_event_loop()
    for signum in signums:
        loop.add_signal_handler(signum, handler)


def parse_source(value: str) -> list[tuple[str, int]] | str:
    """Parse a string into a list of IP endpoints."""
    try:
        endpoints = endpoint_list_parser(7148)(value)
        for endpoint in endpoints:
            ipaddress.IPv4Address(endpoint.host)  # Raises if invalid syntax
        return [(ep.host, ep.port) for ep in endpoints]
    except ValueError:
        return value


class DeviceStatus(Enum):
    """Discrete `device-status` readings.

    Doesn't follow the convention of :class:`aiokatcp.Sensor.Status` because
    it's useful for `UNKNOWN` to have the highest (or worst) value.
    """

    OK = 1
    DEGRADED = 2
    FAIL = 3


class DeviceStatusSensor(aiokatcp.AggregateSensor[DeviceStatus]):
    """Summary sensor for quickly ascertaining device status.

    This takes its value from the worst status of its target set of sensors, so
    it's quick to identify if there's something wrong, or if everything is good.
    """

    def __init__(self, target: aiokatcp.SensorSet) -> None:
        super().__init__(
            target=target, sensor_type=DeviceStatus, name="device-status", description="Overall engine health"
        )

    def update_aggregate(
        self,
        updated_sensor: aiokatcp.Sensor[_T] | None,
        reading: aiokatcp.Reading[_T] | None,
        old_reading: aiokatcp.Reading[_T] | None,
    ) -> aiokatcp.Reading[DeviceStatus] | None:  # noqa: D102
        # For device status it's far simpler just to re-calculate everything
        # each time, than to try and maintain state.
        if reading is not None and old_reading is not None and reading.status == old_reading.status:
            return None  # Sensor didn't change state, so no change in overall device status
        worst_status: aiokatcp.Sensor.Status = aiokatcp.Sensor.Status.NOMINAL
        for sensor in self.target.values():
            if self.filter_aggregate(sensor):
                worst_status = max(worst_status, sensor.status)

        if worst_status == aiokatcp.Sensor.Status.NOMINAL:
            return aiokatcp.Reading(sensor.timestamp, aiokatcp.Sensor.Status.NOMINAL, DeviceStatus.OK)
        if worst_status == aiokatcp.Sensor.Status.WARN:
            return aiokatcp.Reading(sensor.timestamp, aiokatcp.Sensor.Status.WARN, DeviceStatus.DEGRADED)
        return aiokatcp.Reading(sensor.timestamp, aiokatcp.Sensor.Status.ERROR, DeviceStatus.FAIL)
