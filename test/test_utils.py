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

"""Tests for :mod:`katcbfgpu.utils`."""

import asyncio
import gc
import weakref
from typing import AsyncGenerator, Generator

import aiokatcp
import async_solipsism
import pytest

from katgpucbf.utils import TimeConverter, TimeoutSensorStatusObserver


class TestTimeoutSensorStatus:
    """Tests for :func:`katgpucbf.utils.timeout_sensor_status`."""

    @pytest.fixture
    def event_loop(self) -> Generator[async_solipsism.EventLoop, None, None]:
        """Use async_solipsism event loop."""
        loop = async_solipsism.EventLoop()
        yield loop
        loop.close()

    @pytest.fixture
    def sensor(self) -> aiokatcp.Sensor:
        """Create a dummy sensor."""
        return aiokatcp.Sensor(int, "dummy", "Test sensor", default=123, initial_status=aiokatcp.Sensor.Status.NOMINAL)

    @pytest.fixture(autouse=True)
    async def observer(self, sensor: aiokatcp.Sensor) -> AsyncGenerator[TimeoutSensorStatusObserver, None]:
        """Attach a TimeoutSensorStatusObserver to the sensor."""
        # Note: needs to be an async fixture just so that the event loop is running
        observer = TimeoutSensorStatusObserver(sensor, 2.0, aiokatcp.Sensor.Status.ERROR)
        yield observer
        observer.cancel()

    async def test_initial_state(self, sensor: aiokatcp.Sensor) -> None:
        """Test that the observer doesn't immediately change anything."""
        assert sensor.value == 123
        assert sensor.status == aiokatcp.Sensor.Status.NOMINAL

    async def test_simple_timeout(self, sensor: aiokatcp.Sensor) -> None:
        """Test the simple case where the sensor is never updated."""
        start_timestamp = sensor.timestamp
        await asyncio.sleep(3)
        assert sensor.value == 123
        assert sensor.status == aiokatcp.Sensor.Status.ERROR
        assert sensor.timestamp == start_timestamp + 2.0
        # Make sure it doesn't get updated again after another 2 seconds
        await asyncio.sleep(2)
        assert sensor.timestamp == start_timestamp + 2.0

    async def test_delay(self, sensor: aiokatcp.Sensor) -> None:
        """Test that updating the sensor defers the status change."""
        start_timestamp = sensor.timestamp
        await asyncio.sleep(1)
        sensor.set_value(124, timestamp=start_timestamp + 1)
        await asyncio.sleep(1)
        sensor.set_value(125, timestamp=start_timestamp + 2)
        await asyncio.sleep(1.5)
        assert sensor.status == aiokatcp.Sensor.Status.NOMINAL
        assert sensor.timestamp == start_timestamp + 2
        await asyncio.sleep(1)
        # Should now have timed out
        assert sensor.value == 125
        assert sensor.status == aiokatcp.Sensor.Status.ERROR
        assert sensor.timestamp == start_timestamp + 4

    async def test_no_change(self, sensor: aiokatcp.Sensor) -> None:
        """Test that no update is made if the status is already correct."""
        sensor.set_value(124, status=aiokatcp.Sensor.Status.ERROR)
        timestamp = sensor.timestamp
        await asyncio.sleep(5)
        assert sensor.value == 124
        assert sensor.timestamp == timestamp
        assert sensor.status == aiokatcp.Sensor.Status.ERROR

    async def test_cancel(self, sensor: aiokatcp.Sensor, observer: TimeoutSensorStatusObserver) -> None:
        """Test that cancelling the observer prevents further updates."""
        await asyncio.sleep(1)
        observer.cancel()
        await asyncio.sleep(5)
        assert sensor.status == aiokatcp.Sensor.Status.NOMINAL

    async def test_garbage_collection(self) -> None:
        """Ensure that objects can be garbage collected immediately."""
        # Don't use the fixtures, because pytest probably holds refs.
        sensor = aiokatcp.Sensor(int, "dummy", "Test sensor")
        observer = TimeoutSensorStatusObserver(sensor, 2.0, aiokatcp.Sensor.Status.ERROR)
        weak_sensor = weakref.ref(sensor)
        weak_observer = weakref.ref(observer)
        del sensor
        del observer
        for _ in range(5):
            gc.collect()  # Some Python implementation need multiple passes to collect everything
        assert weak_sensor() is None
        assert weak_observer() is None


class TestTimeConverter:
    """Tests for :class:`katgpucbf.utils.TimeConverter`."""

    @pytest.fixture
    def time_converter(self) -> TimeConverter:  # noqa: D401
        """A time converter.

        It has power-of-two ADC sample count so that tests do not need to worry
        about rounding effects.
        """
        return TimeConverter(1234567890.0, 1048576.0)

    def test_unix_to_adc(self, time_converter: TimeConverter) -> None:
        """Test :meth:`.TimeConverter.unix_to_adc`."""
        assert time_converter.unix_to_adc(1234567890.0) == 0.0
        assert time_converter.unix_to_adc(1234567890.0 + 10.0) == 10485760.0

    def test_adc_to_unix(self, time_converter: TimeConverter) -> None:
        """Test :meth:`.TimeConverter.adc_to_unix`."""
        assert time_converter.adc_to_unix(0.0) == 1234567890.0
        assert time_converter.adc_to_unix(10485760.0) == 1234567890.0 + 10.0
