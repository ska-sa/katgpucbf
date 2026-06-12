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

"""Tests for :mod:`katcbfgpu.utils`."""

import asyncio
import gc
import logging
import weakref
from collections.abc import AsyncGenerator, AsyncIterable
from typing import override
from unittest import mock

import aiokatcp
import async_solipsism
import pytest
from aiokatcp import DeviceStatus

from katgpucbf.utils import (
    DeviceStatusSensor,
    DiscardingIterator,
    TimeConverter,
    TimeoutSensorStatusObserver,
)


@pytest.fixture
def counter() -> AsyncIterable[int]:
    """Asynchronous iterator that yields an integer a second for 8 seconds."""

    # We need to use a helper function because pytest_asyncio will apply
    # magic if we make the fixture function itself an async generator.
    async def gen() -> AsyncIterable[int]:
        for i in range(8):
            await asyncio.sleep(1.0)
            yield i
        await asyncio.sleep(1.0)

    return gen()


class SpyDiscardingIterator(DiscardingIterator[int]):
    """Override :meth:`discard` to record the values and times."""

    def __init__(self, base: AsyncIterable[int]) -> None:
        super().__init__(base)
        self.discarded: list[tuple[int, float]] = []

    @override
    def discard(self, item: int) -> None:
        self.discarded.append((item, asyncio.get_running_loop().time()))


# Note: function must be async because DiscardingIterator requires an event loop
@pytest.fixture
async def discarding_iterator(counter: AsyncIterable[int]) -> SpyDiscardingIterator:
    """Fixture for :class:`.DiscardingIterator` tests."""
    return SpyDiscardingIterator(counter)


class TestDiscardingIterator:
    """Test :class:`.DiscardingIterator`."""

    @pytest.fixture
    def event_loop_policy(self) -> async_solipsism.EventLoopPolicy:
        """Use async_solipsism event loop."""
        return async_solipsism.EventLoopPolicy()

    async def test_simple(self, discarding_iterator: SpyDiscardingIterator) -> None:
        """Test basic operations."""
        await asyncio.sleep(2.5)
        items = []
        loop = asyncio.get_running_loop()
        with discarding_iterator:
            async for item in discarding_iterator:
                items.append((item, loop.time()))
                if len(items) == 4:
                    break
        await asyncio.sleep(20)  # counter will run out
        # Check that attempting to iterate over the base iterator completes
        # will not block.
        with discarding_iterator:
            with pytest.raises(StopAsyncIteration):
                await anext(discarding_iterator)
        assert items == [(2, 3.0), (3, 4.0), (4, 5.0), (5, 6.0)]
        assert discarding_iterator.discarded == [
            (0, 1.0),
            (1, 2.0),
            (6, 7.0),
            (7, 8.0),
        ]

    async def test_interrupt(self, discarding_iterator: SpyDiscardingIterator) -> None:
        """Test that starting discarding will interrupt an active iteration."""
        loop = asyncio.get_event_loop()
        loop.call_at(3.5, discarding_iterator.start_discarding)
        items = []
        with discarding_iterator:
            async for item in discarding_iterator:
                items.append(item)
        assert items == [0, 1, 2]
        assert loop.time() == 3.5
        # Ensure that we cleanly switch to discarding mode
        await asyncio.sleep(2)
        assert discarding_iterator.discarded == [(3, 4.0), (4, 5.0)]

    async def test_block(self, discarding_iterator: SpyDiscardingIterator) -> None:
        """Test with a calling iterator that spends time elsewhere too."""
        await asyncio.sleep(2.5)
        items = []
        loop = asyncio.get_running_loop()
        with discarding_iterator:
            async for item in discarding_iterator:
                items.append((item, loop.time()))
                await asyncio.sleep(2)
                if len(items) == 4:
                    break
        await asyncio.sleep(20)  # counter will run out
        assert items == [(2, 3.0), (3, 5.0), (4, 7.0), (5, 9.0)]
        assert discarding_iterator.discarded == [
            (0, 1.0),
            (1, 2.0),
            (6, 11.0),
            (7, 12.0),
        ]

    async def test_iter_end(self, discarding_iterator: SpyDiscardingIterator) -> None:
        """Test case where we're in capturing mode at the end of the base iterator."""
        with discarding_iterator:
            items = [item async for item in discarding_iterator]
        assert items == list(range(8))

    async def test_iter_end_race(self, discarding_iterator: SpyDiscardingIterator) -> None:
        """Test case in which we switch to capturing mode just as the base iterator finishes."""
        await asyncio.sleep(8.5)
        with discarding_iterator:
            items = [item async for item in discarding_iterator]
        assert items == []
        assert discarding_iterator.discarded == [(i, i + 1) for i in range(8)]

    async def test_cancel(self, discarding_iterator: SpyDiscardingIterator) -> None:
        """Test robustness to cancellation of :meth:`!__anext__`."""

        async def iterate(items: list[int]) -> None:
            with discarding_iterator:
                async for item in discarding_iterator:
                    items.append(item)

        async with asyncio.TaskGroup() as tg:
            items: list[int] = []
            task = tg.create_task(iterate(items))
            await asyncio.sleep(2.5)
            task.cancel()
        await asyncio.sleep(20)
        assert items == [0, 1]
        assert discarding_iterator.discarded == [(i, i + 1) for i in range(2, 8)]

    async def test_crash(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test handling of an exception raised from the underlying iterator.

        The implementation is not currently very robust to this (it does not
        pass the exception on to the caller for example), but check that it at
        least logs the exception.
        """

        async def gen() -> AsyncIterable[int]:
            yield 0
            raise RuntimeError("crash test")

        it = SpyDiscardingIterator(gen())
        with caplog.at_level(logging.WARNING, "katgpucbf.utils"):
            await asyncio.sleep(1)
        assert "crash test" in caplog.text
        assert it.discarded == [(0, 0.0)]


class TestDeviceStatusSensor:
    """Test :class:`.DeviceStatusSensor`.

    This is a limited set of tests, because a lot of the functionality is inherited from aiokatcp.
    """

    @pytest.fixture
    def mock_time(self, mocker) -> mock.Mock:
        """Mock out time so that timestamps are predictable."""
        return mocker.patch("time.time", return_value=1234567890.0)

    @pytest.fixture
    def sensors(self, mock_time: mock.Mock) -> aiokatcp.SensorSet:
        """Create a collection of sensors."""
        sensors = aiokatcp.SensorSet()
        sensors.add(aiokatcp.Sensor(int, "sensor1", ""))
        sensors.add(aiokatcp.Sensor(float, "sensor2", "", initial_status=aiokatcp.Sensor.Status.NOMINAL))
        sensors.add(DeviceStatusSensor(sensors))
        return sensors

    def test_initial(self, sensors: aiokatcp.SensorSet) -> None:
        """Test initial reading of the sensor."""
        ds = sensors["device-status"]
        assert ds.reading == aiokatcp.Reading(1234567890.0, aiokatcp.Sensor.Status.NOMINAL, DeviceStatus.OK)

    def test_early_out(self, sensors: aiokatcp.SensorSet, mock_time: mock.Mock) -> None:
        """Update a sensor without changing status, and check that there is no change."""
        ds = sensors["device-status"]
        old = ds.reading
        sensors["sensor2"].set_value(123.0, timestamp=2345678901.0)
        assert ds.reading == old

    def test_status_change(self, sensors: aiokatcp.SensorSet, mock_time: mock.Mock) -> None:
        """Change the status of a sensor and check that device-status follows."""
        ds = sensors["device-status"]
        sensors["sensor1"].set_value(123, timestamp=2345678901.0, status=aiokatcp.Sensor.Status.WARN)
        assert ds.reading == aiokatcp.Reading(2345678901.0, aiokatcp.Sensor.Status.WARN, DeviceStatus.DEGRADED)
        sensors["sensor1"].set_value(234, timestamp=3456789012.0, status=aiokatcp.Sensor.Status.NOMINAL)
        assert ds.reading == aiokatcp.Reading(3456789012.0, aiokatcp.Sensor.Status.NOMINAL, DeviceStatus.OK)


class TestTimeoutSensorStatus:
    """Tests for :func:`katgpucbf.utils.timeout_sensor_status`."""

    @pytest.fixture
    def event_loop_policy(self) -> async_solipsism.EventLoopPolicy:
        """Use async_solipsism event loop."""
        return async_solipsism.EventLoopPolicy()

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
        assert sensor.reading == aiokatcp.Reading(start_timestamp + 2.0, aiokatcp.Sensor.Status.ERROR, 123)
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
        assert sensor.reading == aiokatcp.Reading(start_timestamp + 2, aiokatcp.Sensor.Status.NOMINAL, 125)
        await asyncio.sleep(1)
        # Should now have timed out
        assert sensor.reading == aiokatcp.Reading(start_timestamp + 4, aiokatcp.Sensor.Status.ERROR, 125)

    async def test_no_change(self, sensor: aiokatcp.Sensor) -> None:
        """Test that no update is made if the status is already correct."""
        sensor.set_value(124, status=aiokatcp.Sensor.Status.ERROR)
        timestamp = sensor.timestamp
        await asyncio.sleep(5)
        assert sensor.reading == aiokatcp.Reading(timestamp, aiokatcp.Sensor.Status.ERROR, 124)

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
    def time_converter(self) -> TimeConverter:
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
