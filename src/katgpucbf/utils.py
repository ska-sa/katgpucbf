################################################################################
# Copyright (c) 2020-2025, National Research Foundation (SARAO)
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
import enum
import gc
import ipaddress
import logging
import signal
import time
import weakref
from collections import Counter
from collections.abc import Callable
from typing import TypeVar

import aiokatcp
import numpy as np
import prometheus_client
from katsdptelstate.endpoint import endpoint_list_parser

from . import MIN_SENSOR_UPDATE_PERIOD, TIME_SYNC_TASK_NAME
from .spead import DEFAULT_PORT

_E = TypeVar("_E", bound=enum.Enum)
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


class DitherType(enum.Enum):
    """Type of dithering to apply prior to quantisation."""

    NONE = 0  # Don't change this value: we rely on it being falsey
    UNIFORM = 1
    DEFAULT = 1  # Alias used to determine default when none is specified


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


def add_gc_stats() -> None:
    """Add Prometheus metrics for garbage collection timing.

    It is only safe to call this once.
    """
    gc_time = prometheus_client.Histogram(
        "python_gc_time_seconds",
        "Time spent in garbage collection",
        buckets=[0.0002, 0.0005, 0.001, 0.002, 0.005, 0.010, 0.020, 0.050, 0.100],
        labelnames=["generation"],
    )
    # Make all the metrics exist, before any GC calls happen
    for generation in range(3):
        gc_time.labels(str(generation))
    start_time = 0.0

    def callback(phase: str, info: dict) -> None:
        nonlocal start_time
        if phase == "start":
            start_time = time.monotonic()
        else:
            started = start_time  # Copy as early as possible, before any more GC can happen
            elapsed = time.monotonic() - started
            gc_time.labels(str(info["generation"])).observe(elapsed)

    gc.callbacks.append(callback)


def parse_enum(name: str, value: str, cls: type[_E]) -> _E:
    """Parse a command-line argument into an enum type."""
    table = {member.name.lower(): member for member in cls}
    try:
        return table[value]
    except KeyError:
        raise ValueError(f"Invalid {name} value {value} (valid values are {list(table.keys())})") from None


def parse_dither(value: str) -> DitherType:
    """Parse a string into a dither type."""
    # Note: this allows only the non-aliases, so excludes DEFAULT
    return parse_enum("dither", value, DitherType)


def parse_source(value: str) -> list[tuple[str, int]] | str:
    """Parse a string into a list of IP endpoints."""
    try:
        endpoints = endpoint_list_parser(DEFAULT_PORT)(value)
        for endpoint in endpoints:
            ipaddress.IPv4Address(endpoint.host)  # Raises if invalid syntax
        return [(ep.host, ep.port) for ep in endpoints]
    except ValueError:
        return value


def comma_split(
    base_type: Callable[[str], _T], count: int | None = None, allow_single=False
) -> Callable[[str], list[_T]]:
    """Return a function to split a comma-delimited str into a list of type _T.

    This function is used to parse lists of CPU core numbers, which come from
    the command-line as comma-separated strings, but are obviously more useful
    as a list of ints. It's generic enough that it could process lists of other
    types as well though if necessary.

    Parameters
    ----------
    base_type
        The base type of thing you expect in the list, e.g. `int`, `float`.
    count
        How many of them you expect to be in the list. `None` means the list
        could be any length.
    allow_single
        If true (defaults to false), allow a single value to be used when
        `count` is greater than 1. In this case, it will be repeated `count`
        times.
    """

    def func(value: str) -> list[_T]:  # noqa: D102
        parts = value.split(",")
        if parts == [""]:
            parts = []
        n = len(parts)
        if count is not None and n == 1 and allow_single:
            parts = parts * count
        elif count is not None and n != count:
            raise ValueError(f"Expected {count} comma-separated fields, received {n}")
        return [base_type(part) for part in parts]

    return func


class DeviceStatusSensor(aiokatcp.SimpleAggregateSensor[aiokatcp.DeviceStatus]):
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
            sensor_type=aiokatcp.DeviceStatus,
            name=name,
            description=description,
        )

    def update_aggregate(
        self,
        updated_sensor: aiokatcp.Sensor[_T] | None,
        reading: aiokatcp.Reading[_T] | None,
        old_reading: aiokatcp.Reading[_T] | None,
    ) -> aiokatcp.Reading[aiokatcp.DeviceStatus] | None:  # noqa: D102
        if reading is not None and old_reading is not None and reading.status == old_reading.status:
            return None  # Sensor didn't change state, so no change in overall device status
        return super().update_aggregate(updated_sensor, reading, old_reading)

    def aggregate_add(self, sensor: aiokatcp.Sensor[_T], reading: aiokatcp.Reading[_T]) -> bool:  # noqa: D102
        self._counts[reading.status] += 1
        return True

    def aggregate_remove(self, sensor: aiokatcp.Sensor[_T], reading: aiokatcp.Reading[_T]) -> bool:  # noqa: D102
        self._counts[reading.status] -= 1
        return True

    def aggregate_compute(self) -> tuple[aiokatcp.Sensor.Status, aiokatcp.DeviceStatus]:  # noqa: D102
        worst_status = max(
            (status for status, count in self._counts.items() if count > 0), default=aiokatcp.Sensor.Status.NOMINAL
        )
        if worst_status <= aiokatcp.Sensor.Status.NOMINAL:  # NOMINAL or UNKNOWN
            return (aiokatcp.Sensor.Status.NOMINAL, aiokatcp.DeviceStatus.OK)
        # We won't return FAIL because if the device is unusable, we probably
        # won't be able to.
        return (aiokatcp.Sensor.Status.WARN, aiokatcp.DeviceStatus.DEGRADED)

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
            await asyncio.sleep(MIN_SENSOR_UPDATE_PERIOD)

    return asyncio.create_task(run(), name=TIME_SYNC_TASK_NAME)


def steady_state_timestamp_sensor() -> aiokatcp.Sensor[int]:
    """Create ``steady-state-timestamp`` sensor."""
    return aiokatcp.Sensor(
        int,
        "steady-state-timestamp",
        "Heaps with this timestamp or greater are guaranteed to reflect the effects of previous katcp requests.",
        default=0,
        initial_status=aiokatcp.Sensor.Status.NOMINAL,
    )


class TimeConverter:
    """Convert times between UNIX timestamps and ADC sample counts.

    Note that because UNIX timestamps are handled as 64-bit floats, they are
    only accurate to roughly microsecond precision, and will not round-trip
    precisely.

    Parameters
    ----------
    sync_time
        UNIX timestamp corresponding to ADC timestamp 0
    adc_sample_rate
        Number of ADC samples per second

    .. todo::

       This does not yet handle leap-seconds correctly.
    """

    def __init__(self, sync_time: float, adc_sample_rate: float) -> None:
        self.sync_time = sync_time
        self.adc_sample_rate = adc_sample_rate

    def unix_to_adc(self, timestamp: float) -> float:
        """Convert a UNIX timestamp to an ADC sample count."""
        return (timestamp - self.sync_time) * self.adc_sample_rate

    def adc_to_unix(self, samples: float) -> float:
        """Convert an ADC sample count to a UNIX timstamp."""
        return samples / self.adc_sample_rate + self.sync_time


def gaussian_dtype(bits: int) -> np.dtype:
    """Get numpy dtype for a Gaussian (complex) integer.

    Parameters
    ----------
    bits
        Number of bits in each real component
    """
    assert bits in {4, 8, 16, 32}
    if bits == 4:
        # 1-byte, packed with real high, imaginary low. Using void rather
        # than e.g. uint8 avoids accidentally doing arithmetic on it.
        return np.dtype("V1")
    else:
        return np.dtype([("real", f"int{bits}"), ("imag", f"int{bits}")])
