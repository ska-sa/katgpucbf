################################################################################
# Copyright (c) 2023-2025, National Research Foundation (SARAO)
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

"""Common tools for benchmark_fgpu and benchmark_xbgpu."""

import argparse
import asyncio
import sys
import time
from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass

import aiohttp.client
import numpy as np
from prometheus_client.parser import text_string_to_metric_families
from scipy.special import expit

from noisy_search import noisy_search
from remote import Server

HEAPS_TOL = 0.05  #: Relative tolerance for number of heaps received
DEFAULT_IMAGE = "harbor.sdp.kat.ac.za/dpp/katgpucbf"
NOISE = 0.01  #: Minimum probability of an incorrect result from each trial
TOLERANCE = 0.001  #: Complement of confidence interval probability
#: Verbosity level at which individual test results are reported
VERBOSE_RESULTS = 1
#: Starting port for Prometheus metrics
PROMETHEUS_PORT_BASE = 7250


@dataclass
class Result:
    """Result of a single trial."""

    expected_heaps: float  #: Number of heaps expected, based on runtime and data rate
    heaps: int  #: Number of heaps received
    missing_heaps: int  #: Number of heaps reported as missing (gaps in incoming data)

    def good(self) -> bool:
        """Whether the trial was a success (kept up with the rate).

        If this is false, it does not mean that heaps were lost. If
        :attr:`missing_heaps` is 0, it could mean that data was not being sent
        at the correct rate, or that latency prevented the queries from
        being performed at the correct times.
        """
        if self.missing_heaps > 0:
            return False
        return (1.0 - HEAPS_TOL) * self.expected_heaps <= self.heaps <= (1.0 + HEAPS_TOL) * self.expected_heaps

    def message(self) -> str:
        """Human-readable description of the outcome."""
        if self.missing_heaps > 0:
            return f"Missing {self.missing_heaps} heaps"
        elif not self.good():
            return f"Expected ±{self.expected_heaps}, received {self.heaps}"
        else:
            return "Good"


class Benchmark(ABC):
    """Abstract base class for benchmarks.

    Subclasses provide the implementation details for a generator and a
    consumer.

    The argument parser must contain at least:
    - n (number of consumer processes)
    - verbose (verbosity level)
    - startup_time
    - runtime
    """

    def __init__(
        self,
        args: argparse.Namespace,
        generator_server: Server,
        consumer_server: Server,
        expected_heaps_scale: float,
        metric_prefix: str,
    ) -> None:
        self.args = args
        self.generator_server = generator_server
        self.consumer_server = consumer_server
        self.expected_heaps_scale = expected_heaps_scale
        self.metric_prefix = metric_prefix

    def verbose_results(self) -> bool:
        return self.args.verbose >= VERBOSE_RESULTS

    async def _heap_counts1(self, session: aiohttp.client.ClientSession, url: str) -> tuple[int, int]:
        heaps = 0
        missing_heaps = 0
        async with session.get(url) as resp:
            for family in text_string_to_metric_families(await resp.text()):
                for sample in family.samples:
                    if sample.name == f"{self.metric_prefix}_input_heaps_total":
                        heaps += int(sample.value)
                    elif sample.name == f"{self.metric_prefix}_input_missing_heaps_total":
                        missing_heaps += int(sample.value)
        return (heaps, missing_heaps)

    async def heap_counts(self, session: aiohttp.client.ClientSession) -> tuple[int, int]:
        """Query the number of heaps received and missing, for all n instances."""
        server = self.consumer_server
        async with asyncio.TaskGroup() as tg:
            tasks = [
                tg.create_task(
                    self._heap_counts1(session, f"http://{server.hostname}:{PROMETHEUS_PORT_BASE + i}/metrics")
                )
                for i in range(self.args.n)
            ]
        partials = [task.result() for task in tasks]
        heaps, missing_heaps = zip(*partials, strict=True)
        return sum(heaps), sum(missing_heaps)

    @abstractmethod
    async def run_producers(self, adc_sample_rate: float, sync_time: int) -> AbstractAsyncContextManager:
        pass

    @abstractmethod
    async def run_consumers(self, adc_sample_rate: float, sync_time: int) -> AbstractAsyncContextManager:
        pass

    async def process(self, adc_sample_rate: float) -> Result:
        """Perform a single trial on running engines."""
        async with aiohttp.client.ClientSession() as session:
            await asyncio.sleep(self.args.startup_time)  # Give a chance for startup losses
            async with asyncio.TaskGroup() as tg:
                # Start the sleep *before* entering heap_counts, so that time spent
                # during heap_counts is considered part of the sleep time.
                tg.create_task(asyncio.sleep(self.args.runtime))
                orig_heaps, orig_missing = await self.heap_counts(session)
            new_heaps, new_missing = await self.heap_counts(session)

        expected_heaps = self.args.runtime * self.args.n * adc_sample_rate * self.expected_heaps_scale
        return Result(
            expected_heaps=expected_heaps,
            heaps=new_heaps - orig_heaps,
            missing_heaps=new_missing - orig_missing,
        )

    async def trial(self, adc_sample_rate: float) -> Result:
        """Perform a single trial."""
        sync_time = int(time.time())
        async with await self.run_producers(adc_sample_rate, sync_time):
            async with await self.run_consumers(adc_sample_rate, sync_time):
                return await self.process(adc_sample_rate)
        raise AssertionError("should be unreachable")

    async def measure(self, adc_sample_rate: float) -> Result:
        """Perform a single trial, but repeat if no heaps were lost yet the wrong number were received.

        This also prints status information to stderr.
        """
        while True:
            if self.verbose_results():
                print(f"Testing {adc_sample_rate / 1e6} MHz... ", end="", flush=True, file=sys.stderr)
            result = await self.trial(adc_sample_rate)
            if not result.good() and result.missing_heaps == 0:
                if self.verbose_results():
                    print(f"{result.message()}, re-running", file=sys.stderr)
            else:
                if self.verbose_results():
                    print(result.message(), file=sys.stderr)
                return result

    async def calibrate(self, low: float, high: float, step: float, repeat: int) -> None:
        """Run multiple trials on all the possible rates."""
        rates = np.arange(low, high + 0.01 * step, step)
        successes = [0] * len(rates)
        for trial in range(repeat):
            for j, adc_sample_rate in enumerate(rates):
                sync_time = int(time.time())
                redo = True
                while redo:
                    redo = False
                    if self.verbose_results():
                        print(f"Testing {adc_sample_rate / 1e6} MHz... ", end="", flush=True, file=sys.stderr)
                    async with await self.run_producers(adc_sample_rate, sync_time):
                        async with await self.run_consumers(adc_sample_rate, sync_time):
                            result = await self.process(adc_sample_rate)
                    if result.good():
                        successes[j] += 1
                    elif result.missing_heaps == 0:
                        redo = True  # Unexpected number of heaps received
                    if self.verbose_results():
                        if redo:
                            print(f"{result.message()}, rerunning", flush=True, file=sys.stderr)
                        else:
                            print(f"{result.message()}, {successes[j]}/{trial + 1} passed", flush=True, file=sys.stderr)
        for success, adc_sample_rate in zip(successes, rates, strict=True):
            print(adc_sample_rate, success, repeat)

    async def oneshot(self, adc_sample_rate: float) -> None:
        """Measure at a single rate."""
        while True:
            result = await self.trial(adc_sample_rate)
            if not result.good() and result.missing_heaps == 0:
                if self.verbose_results():
                    print(f"{result.message()}, re-running", file=sys.stderr)
            else:
                print(result.message())
                break

    async def search(
        self, low: float, high: float, step: float, interval: float, max_comparisons: int, slope: float
    ) -> tuple[float, float]:
        """Search for the critical rate."""

        async def compare(adc_sample_rate: float) -> bool:
            return not (await self.measure(adc_sample_rate)).good()

        # The additional 0.01 * args.step is to ensure high is included rather
        # than excluded if the range is a multiple of step.
        rates = np.arange(low, high + 0.01 * step, step)
        low_result = await self.measure(rates[0])
        if not low_result.good():
            raise RuntimeError(f"failed on low: {low_result.message()}")
        high_result = await self.measure(rates[-1])
        if high_result.good():
            raise RuntimeError("succeeded on high")

        mid_rates = 0.5 * (rates[:-1] + rates[1:])  # Rates in the middle of the intervals
        mid_rates = np.r_[low, mid_rates, high]
        l_rates = np.log(rates)[:, np.newaxis]
        l_mid_rates = np.log(mid_rates)[np.newaxis, :]
        noise = expit((l_mid_rates - l_rates) * slope)
        # Don't allow probabilities to get too close to 0/1, as there are may be some
        # external factor that makes things go wrong even at very low/high rates.
        noise = np.clip(noise, NOISE, 1 - NOISE)

        result = await noisy_search(
            list(rates),
            noise,
            TOLERANCE,
            compare,
            max_interval=round(interval / step),
            max_comparisons=max_comparisons,
        )
        if result.low == -1:
            raise RuntimeError("lower bound is too high")
        elif result.high == len(rates):
            raise RuntimeError("upper bound is too low")
        else:
            return rates[result.low], rates[result.high]
