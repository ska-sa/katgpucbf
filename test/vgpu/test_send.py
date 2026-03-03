################################################################################
# Copyright (c) 2026, National Research Foundation (SARAO)
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

"""Unit tests for :mod:`katgpucbf.vgpu.send`."""

import asyncio
from typing import override

import async_solipsism
import pytest

from katgpucbf.vgpu.send import PreciseTime, PreciseTimeDelta, RateLimiter


class TestPreciseTime:
    """Test :class:`.PreciseTime` and :class:`.PreciseTimeDelta`."""

    def test_round_trip(self) -> None:
        """Test that constructing and converting a time/timedelta round trips."""
        assert float(PreciseTime(1234.5)) == 1234.5
        assert float(PreciseTimeDelta(1234.5)) == 1234.5

    def test_add_sub(self) -> None:
        """Test adding and subtracting times/timedeltas."""
        assert float(PreciseTime(1234.5) + PreciseTimeDelta(1.5)) == 1236.0
        assert float(PreciseTimeDelta(1234.5) + PreciseTimeDelta(1.5)) == 1236.0
        assert float(PreciseTime(1234.5) - PreciseTimeDelta(2.0)) == 1232.5
        assert float(PreciseTimeDelta(1234.5) - PreciseTimeDelta(2.0)) == 1232.5
        assert float(PreciseTime(1234.5) - PreciseTime(1234.25)) == 0.25

    def test_compare(self) -> None:
        """Test comparisons between times/timedeltas."""
        assert PreciseTime(1234.5) < PreciseTime(1234.6)
        assert not (PreciseTime(1234.5) < PreciseTime(1234.5))
        assert not (PreciseTime(1234.5) < PreciseTime(1234.4))

        assert PreciseTime(1234.5) == PreciseTime(1234.5)
        assert PreciseTime(1234.5) != PreciseTime(1234.6)

    def test_type_mismatch(self) -> None:
        """Test various badly-defined combinations of types."""
        with pytest.raises(TypeError):
            PreciseTime(1234.5) + 3  # type: ignore[operator]
        with pytest.raises(TypeError):
            PreciseTime(1234.5) + PreciseTime(1.0)  # type: ignore[operator]
        with pytest.raises(TypeError):
            PreciseTimeDelta(1234.5) - PreciseTime(1.0)  # type: ignore[operator]
        with pytest.raises(TypeError):
            PreciseTime(1234.5) < PreciseTimeDelta(1234.5)  # type: ignore[operator] # noqa: B015

    def test_precision(self) -> None:
        """Test that PreciseTime can maintain precision over many additions."""
        start = PreciseTime(1234567890.0)
        step = PreciseTimeDelta(1e-9)
        cur = start
        for _ in range(100000):
            cur += step
        assert float(cur - start) == pytest.approx(1e-4, rel=1e-9)


class DummyRateLimiter(RateLimiter[int]):
    """Process integers and store the times they were processed."""

    def __init__(self, rate: float, burst_rate: float) -> None:
        super().__init__(rate, burst_rate)
        self.times: list[float] = []

    @override
    def item_size(self, item: int) -> int:
        return item

    @override
    async def _process_item(self, item: int) -> None:
        self.times.append(asyncio.get_running_loop().time())


class TestRateLimiter:
    """Test :class:`.RateLimiter`."""

    @pytest.fixture
    def event_loop_policy(self) -> async_solipsism.EventLoopPolicy:
        """Use async_solipsism event loop."""
        return async_solipsism.EventLoopPolicy()

    async def test(self) -> None:
        """Test :class:`.RateLimiter`."""
        # Use a TaskGroup so that we can schedule items to be sent at
        # precisely-controlled times, regardless of any sleeping that
        # the tasks do.
        limiter = DummyRateLimiter(10, 20)
        async with asyncio.TaskGroup() as tg:
            # Two back-to-back
            tg.create_task(limiter.send(1))
            tg.create_task(limiter.send(1))
            # Long gap so we have to start catching up
            await asyncio.sleep(0.7)
            # More back-to-back so we catch up then level out
            for _ in range(4):
                tg.create_task(limiter.send(4))
        assert limiter.times == pytest.approx(
            [
                0.0,
                0.1,
                0.7,
                0.9,  # Catchup
                1.1,  # Catchup
                1.4,  # Have caught up
            ]
        )
