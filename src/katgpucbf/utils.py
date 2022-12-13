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

import asyncio
import ipaddress
import logging
import signal
import weakref
from collections import Counter
from enum import Enum
from typing import TypeVar

import aiokatcp

from . import TIME_SYNC_TASK_NAME
from .endpoint import endpoint_list_parser

_T = TypeVar("_T")

# Sensor status threshold. These are mostly thumb-sucks.
TIME_ESTERROR_WARN = 1e-3
TIME_ESTERROR_ERROR = 5e-3  # CBF-REQ-0203 specifies 5ms
# maxerror is an over-estimate (it makes very conservative assumptions
# about network asymmetry and skew in the clock source). Use conservative
# thresholds to avoid warnings when there isn't a stratum 1 time source
# in the same data centre. Experimentally, maximum error is < 10ms
# when synchronising between Cape Town and the Karoo.
TIME_MAXERROR_WARN = 10e-3
TIME_MAXERROR_ERROR = 0.1

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

    loop = asyncio.get_running_loop()
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

    def __init__(
        self, target: aiokatcp.SensorSet, name: str = "device-status", description: str = "Overall engine health"
    ) -> None:
        # We count the number of sensors with each possible status
        self._counts: Counter[aiokatcp.Sensor.Status] = Counter()
        super().__init__(
            target=target,
            sensor_type=DeviceStatus,
            name=name,
            description=description,
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

    def filter_aggregate(self, sensor: aiokatcp.Sensor) -> bool:  # noqa: D102
        # Filter other aggregate sensors out. We don't need them because the
        # underlying (normal) sensors are incorporated.
        return not isinstance(sensor, aiokatcp.AggregateSensor)


class TimeoutSensorStatusObserver:
    """Change the status of a sensor if it doesn't receive an update for a given time.

    Do not directly attach or detach this observer from the sensor (it does
    this internally). It is not necessary to retain a reference to the object
    unless you wish to interact with it later (for example, by calling
    :meth:`cancel`).

    It must be constructed while there is a running event loop.
    """

    def __init__(self, sensor: aiokatcp.Sensor, timeout: float, new_status: aiokatcp.Sensor.Status) -> None:
        loop = asyncio.get_running_loop()
        self._sensor = weakref.ref(sensor, self._cleanup)
        self._new_status = new_status
        self._timeout = timeout
        self._cb_handle: asyncio.TimerHandle | None = None  # Callback to change status after timeout
        if sensor.status != new_status:
            self._cb_handle = loop.call_later(timeout, self._change_status)
        sensor.attach(self)

    def _cancel_cb(self) -> None:
        """Cancel the callback handle, if any."""
        if self._cb_handle is not None:
            self._cb_handle.cancel()
            self._cb_handle = None

    def __call__(self, sensor: aiokatcp.Sensor, reading: aiokatcp.Reading) -> None:
        """Sensor update callback (do not call directly)."""
        # Cancel the countdown, and start a new one if appropriate.
        self._cancel_cb()
        if reading.status != self._new_status:
            self._cb_handle = asyncio.get_running_loop().call_later(self._timeout, self._change_status)

    def _change_status(self) -> None:
        """Update the status of the sensor when the timeout expires."""
        self._cb_handle = None
        sensor = self._sensor()
        # Check that the sensor wasn't deleted already. It's unlikely, because _cleanup will
        # cancel the callback, but potentially there are race conditions.
        if sensor is not None:
            logger.debug("Changing sensor status of %s to %s after timeout", sensor.name, self._new_status)
            timestamp = sensor.timestamp + self._timeout
            sensor.set_value(sensor.value, status=self._new_status, timestamp=timestamp)

    def _cleanup(self, weak_sensor: weakref.ReferenceType) -> None:
        """Cancel the callback if the sensor is garbage collected.

        This allows the observer to be garbage collected immediately; otherwise
        it can only be collected once the timeout fires, because the event loop
        holds a reference.
        """
        self._cancel_cb()

    def cancel(self) -> None:
        """Detach from the sensor and make no further updates to it."""
        self._cancel_cb()
        sensor = self._sensor()
        if sensor is not None:
            sensor.detach(self)


def _time_esterror_status(seconds: float) -> aiokatcp.Sensor.Status:
    if 0.0 <= seconds < TIME_ESTERROR_WARN:
        return aiokatcp.Sensor.Status.NOMINAL
    elif seconds < TIME_ESTERROR_ERROR:
        return aiokatcp.Sensor.Status.WARN
    else:
        return aiokatcp.Sensor.Status.ERROR


def _time_maxerror_status(seconds: float) -> aiokatcp.Sensor.Status:
    if 0.0 <= seconds < TIME_MAXERROR_WARN:
        return aiokatcp.Sensor.Status.NOMINAL
    elif seconds < TIME_MAXERROR_ERROR:
        return aiokatcp.Sensor.Status.WARN
    else:
        return aiokatcp.Sensor.Status.ERROR


def _time_state_status(value: aiokatcp.ClockState) -> aiokatcp.Sensor.Status:
    if value == aiokatcp.ClockState.OK:
        return aiokatcp.Sensor.Status.NOMINAL
    elif value == aiokatcp.ClockState.ERROR:
        return aiokatcp.Sensor.Status.ERROR
    else:
        # Some form of leap second adjustment
        return aiokatcp.Sensor.Status.WARN


def add_time_sync_sensors(sensors: aiokatcp.SensorSet) -> asyncio.Task:
    """Add a number of sensors to a device server to track time synchronisation.

    This must be called with an event loop running. It returns a task that
    keeps the sensors periodically updated.
    """
    mapping: dict[str, aiokatcp.Sensor] = {
        "esterror": aiokatcp.Sensor(
            float, "time.esterror", "Estimated time synchronisation error", units="s", status_func=_time_esterror_status
        ),
        "maxerror": aiokatcp.Sensor(
            float,
            "time.maxerror",
            "Upper bound on time synchronisation error",
            units="s",
            status_func=_time_maxerror_status,
        ),
        "state": aiokatcp.Sensor(
            aiokatcp.ClockState, "time.state", "Kernel clock state", status_func=_time_state_status
        ),
    }
    for sensor in mapping.values():
        sensors.add(sensor)

    synchronised_sensor = aiokatcp.Sensor(
        bool, "time.synchronised", "Whether the host clock is synchronised within tolerances"
    )
    sensors.add(synchronised_sensor)
    updater = aiokatcp.TimeSyncUpdater(mapping)

    async def run() -> None:
        while True:
            updater.update()
            good = all(sensor.status != aiokatcp.Sensor.Status.ERROR for sensor in mapping.values())
            synchronised_sensor.set_value(
                good,
                status=aiokatcp.Sensor.Status.NOMINAL if good else aiokatcp.Sensor.Status.ERROR,
                timestamp=mapping["state"].timestamp,
            )
            # 1 second is pretty arbitrary. It's probably faster than
            # necessary, but subscribers can always use a rate-limiting
            # sampling strategy to reduce the rate of updates.
            await asyncio.sleep(1)

    return asyncio.create_task(run(), name=TIME_SYNC_TASK_NAME)


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
