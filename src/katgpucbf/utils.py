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
from collections import Counter
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
    """Discrete `device-status` readings."""

    OK = 1
    DEGRADED = 2
    FAIL = 3


class DeviceStatusSensor(aiokatcp.SimpleAggregateSensor[DeviceStatus]):
    """Summary sensor for quickly ascertaining device status.

    This takes its value from the worst status of its target set of sensors, so
    it's quick to identify if there's something wrong, or if everything is good.
    """

    def __init__(self, target: aiokatcp.SensorSet) -> None:
        # We count the number of sensors with each possible status
        self._counts: Counter[aiokatcp.Sensor.Status] = Counter()
        super().__init__(
            target=target, sensor_type=DeviceStatus, name="device-status", description="Overall engine health"
        )

    def update_aggregate(
        self,
        updated_sensor: aiokatcp.Sensor[_T] | None,
        reading: aiokatcp.Reading[_T] | None,
        old_reading: aiokatcp.Reading[_T] | None,
    ) -> aiokatcp.Reading[DeviceStatus] | None:  # noqa: D102
        if reading is not None and old_reading is not None and reading.status == old_reading.status:
            return None  # Sensor didn't change state, so no change in overall device status
        return super().update_aggregate(updated_sensor, reading, old_reading)

    def aggregate_add(self, sensor: aiokatcp.Sensor[_T], reading: aiokatcp.Reading[_T]) -> bool:  # noqa: D102
        self._counts[reading.status] += 1
        return True

    def aggregate_remove(self, sensor: aiokatcp.Sensor[_T], reading: aiokatcp.Reading[_T]) -> bool:  # noqa: D102
        self._counts[reading.status] -= 1
        return True

    def aggregate_compute(self) -> tuple[aiokatcp.Sensor.Status, DeviceStatus]:  # noqa: D102
        worst_status = max(
            (status for status, count in self._counts.items() if count > 0), default=aiokatcp.Sensor.Status.NOMINAL
        )
        if worst_status <= aiokatcp.Sensor.Status.NOMINAL:  # NOMINAL or UNKNOWN
            return (aiokatcp.Sensor.Status.NOMINAL, DeviceStatus.OK)
        # We won't return FAIL because if the device is unusable, we probably
        # won't be able to.
        return (aiokatcp.Sensor.Status.WARN, DeviceStatus.DEGRADED)


class TimeConverter:
    """Convert times between UNIX timestamps and ADC sample counts.

    Note that because UNIX timestamps are handled as 64-bit floats, they are
    only accurate to roughly microsecond precision, and will not round-trip
    precisely.

    Parameters
    ----------
    sync_epoch
        UNIX timestamp corresponding to ADC timestamp 0
    adc_sample_rate
        Number of ADC samples per second

    .. todo::

       This does not yet handle leap-seconds correctly.
    """

    def __init__(self, sync_epoch: float, adc_sample_rate: float) -> None:
        self.sync_epoch = sync_epoch
        self.adc_sample_rate = adc_sample_rate

    def unix_to_adc(self, timestamp: float) -> float:
        """Convert a UNIX timestamp to an ADC sample count."""
        return (timestamp - self.sync_epoch) * self.adc_sample_rate

    def adc_to_unix(self, samples: float) -> float:
        """Convert an ADC sample count to a UNIX timstamp."""
        return samples / self.adc_sample_rate + self.sync_epoch
