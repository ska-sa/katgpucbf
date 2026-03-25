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

from katgpucbf.vgpu.send import RateLimiter


class DummyRateLimiter(RateLimiter[int]):
    """Process integers and store the times they were processed."""

    def __init__(self, rate: float, burst_rate: float) -> None:
        super().__init__(rate, burst_rate, 2)
        self.times: list[float] = []

    @override
    def item_size(self, item: int) -> int:
        return item

    @override
    async def _process_item(self, item: int) -> None:
        self.times.append(asyncio.get_running_loop().time())
        if item == 4:
            await asyncio.sleep(0.5)


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
            # Long gap so that we reset the reference point
            await asyncio.sleep(0.5)
            # More back-to-back, for which processing causes delays
            # (special case in DummyRateLimiter for item=4).
            for _ in range(3):
                tg.create_task(limiter.send(4))
            # Some more back-to-back for which we will be catching up
            for _ in range(5):
                tg.create_task(limiter.send(2))
        await limiter.join()
        assert limiter.times == pytest.approx(
            [
                0.0,
                0.1,
                0.5,
                1.0,
                1.5,
                2.0,
                2.1,
                2.2,
                2.3,
                2.5,  # All caught up now, revert to standard rate
            ]
        )
