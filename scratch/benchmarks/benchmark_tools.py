################################################################################
# Copyright (c) 2023-2026, National Research Foundation (SARAO)
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
import logging
import sys
import time
from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from enum import Enum

import aiohttp.client
import numpy as np
from prometheus_client.parser import text_string_to_metric_families
from scipy.special import expit

from katgpucbf import DEFAULT_JONES_PER_BATCH

from noisy_search import NoisySearchResult, noisy_search
from remote import Server

HEAPS_TOL = 0.05  #: Relative tolerance for number of heaps received
DEFAULT_IMAGE = "harbor.sdp.kat.ac.za/dpp/katgpucbf"
NOISE = 0.01  #: Minimum probability of an incorrect result from each trial
TOLERANCE = 0.001  #: Complement of confidence interval probability
#: Verbosity level at which individual test results are reported
VERBOSE_RESULTS = 1
#: Starting port for Prometheus metrics
PROMETHEUS_PORT_BASE = 7250
THROTTLE_RETRIES = 3  #: Maximum number of times to retry a throttled trial
logger = logging.getLogger(__name__)


def add_common_benchmark_arguments(parser: argparse.ArgumentParser) -> None:
    """Add common command-line arguments shared by benchmark_fgpu and benchmark_xbgpu.

    Parameters
    ----------
    parser
        Argument parser to add arguments to.
    """
    parser.add_argument("-n", type=int, default=4, help="Number of engines [%(default)s]")
    parser.add_argument("--channels", type=int, default=1024, help="Channel count [%(default)s]")
    parser.add_argument(
        "--array-size",
        type=int,
        default=80,
        help="The number of antennas in the array [%(default)s]",
    )
    parser.add_argument(
        "--jones-per-batch",
        type=int,
        default=DEFAULT_JONES_PER_BATCH,
        metavar="SAMPLES",
        help="Jones vectors in each output batch [%(default)s]",
    )
    parser.add_argument("--low", type=float, default=1500e6, help="Minimum ADC sample rate to search [%(default)s]")
    parser.add_argument("--high", type=float, default=2200e6, help="Maximum ADC sample rate to search [%(default)s]")
    # For backwards compatibility, specifying --narrowband (without an argument) is
    # equivalent to specifying --narrowband=1.
    parser.add_argument(
        "--narrowband", type=int, default=0, const=1, nargs="?", help="Number of narrowband outputs [0]"
    )
    parser.add_argument(
        "--narrowband-decimation", type=int, default=8, help="Narrowband decimation factor [%(default)s]"
    )
    parser.add_argument(
        "--init-time", type=float, default=20.0, metavar="SECONDS", help="Time for engines to start [%(default)s]"
    )
    parser.add_argument(
        "--startup-time",
        type=float,
        default=1.0,
        metavar="SECONDS",
        help="Time to run before starting measurement [%(default)s]",
    )
    parser.add_argument(
        "--runtime", type=float, default=20.0, metavar="SECONDS", help="Time to let engine run [%(default)s]"
    )
    parser.add_argument("--image", type=str, default=DEFAULT_IMAGE, help="Docker image [%(default)s]")
    parser.add_argument("--no-pull", dest="pull", action="store_false", help="Do not pull Docker image")
    parser.add_argument("--servers", type=str, default="servers.toml", help="Server description file [%(default)s]")
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity [no]. Apply multiple times for greater effect.",
    )
    parser.add_argument("--oneshot", type=float, help="Run one test at the given sampling rate")
    parser.add_argument("--step", type=float, default=1e6, help="Step size between sample rates to test [%(default)s]")
    parser.add_argument("--interval", type=float, default=20e6, help="Target confidence interval [%(default)s]")
    parser.add_argument("--max-comparisons", type=int, default=40, help="Maximum comparisons to make [%(default)s]")
    parser.add_argument(
        "--calibrate", action="store_true", help="Run at multiple rates to calibrate expectations [%(default)s]"
    )
    parser.add_argument(
        "--calibrate-repeat", type=int, default=100, help="Number of times to run at each rate [%(default)s]"
    )


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

    def throttled(self) -> bool:
        """Whether the trial was throttled (kept up with the rate, but not at the correct data rate).

        This typically happens when the rate is too high for the network devices to keep up with.
        """
        return not self.good() and self.missing_heaps == 0 and self.heaps < (1.0 + HEAPS_TOL) * self.expected_heaps

    def message(self) -> str:
        """Human-readable description of the outcome."""
        if self.missing_heaps > 0:
            return f"Missing {self.missing_heaps} heaps"
        elif not self.good():
            return f"Expected ±{self.expected_heaps}, received {self.heaps}"
        else:
            return "Good"


class TrialState(Enum):
    """State of a benchmark trial."""

    SUCCESS = "success"
    THROTTLED = "throttled"
    FAILED = "failed"


@dataclass
class MeasureResult:
    """Result of a single measurement."""

    trail: Result
    throttled_adc: int
    state: TrialState

    def message(self) -> str:
        match self.state:
            case TrialState.SUCCESS:
                return self.trail.message()
            case TrialState.THROTTLED:
                return f"Throttled to {self.throttled_adc / 1e6} MHz"
            case TrialState.FAILED:
                return self.trail.message()


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
        slope: dict[int, float],
    ) -> None:
        self.args = args
        self.generator_server = generator_server
        self.consumer_server = consumer_server
        self.expected_heaps_scale = expected_heaps_scale
        self.metric_prefix = metric_prefix
        self.slope = slope

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
            # We throw away these results, but it causes `session` to
            # establish the HTTP connection, which helps minimise the latency
            # when we first request the results for real.
            await self.heap_counts(session)
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

    async def measure(self, adc_sample_rate: float) -> MeasureResult:
        """Perform a single trial.
        Returns a :class:`Result` describing the outcome of the trial.
        """

        if self.verbose_results():
            print(f"Testing {adc_sample_rate / 1e6} MHz... ", end="", flush=True, file=sys.stderr)

        state = TrialState.FAILED

        throttled_results = []
        for _ in range(THROTTLE_RETRIES):
            result = await self.trial(adc_sample_rate)
            if result.throttled():
                throttled_results.append(result)
            elif result.good():
                state = TrialState.SUCCESS
                break

        throttled_adc = 0
        if len(throttled_results) == THROTTLE_RETRIES:
            state = TrialState.THROTTLED
            percent_throttled = sum([result.heaps for result in throttled_results]) / sum(
                [result.expected_heaps for result in throttled_results]
            )
            throttled_adc = int(adc_sample_rate * percent_throttled)

        if self.verbose_results():
            print(result.message(), file=sys.stderr)
            if state == TrialState.THROTTLED:
                print(f" Throttled to {throttled_adc / 1e6} MHz", file=sys.stderr)
            print("\n", file=sys.stderr)

        return MeasureResult(trail=result, throttled_adc=throttled_adc, state=state)

    async def calibrate(self, low: float, high: float, step: float, repeat: int) -> str:
        """Run multiple trials on all the possible rates."""
        rates = np.arange(low, high + 0.01 * step, step).tolist()
        successes = [0] * len(rates)
        throttled = [0] * len(rates)
        for _ in range(repeat):
            for j, adc_sample_rate in enumerate(rates):
                if self.verbose_results():
                    print(f"Testing {adc_sample_rate / 1e6} MHz... ", end="", flush=True, file=sys.stderr)
                measurement = await self.measure(adc_sample_rate)
                if measurement.state == TrialState.SUCCESS:
                    successes[j] += 1
                elif measurement.state == TrialState.THROTTLED:
                    throttled[j] += 1
                elif measurement.state == TrialState.FAILED:
                    logger.error(f"No heaps received for {adc_sample_rate / 1e6} MHz")
                    raise RuntimeError(f"No heaps received for {adc_sample_rate / 1e6} MHz result is inconclusive")
        output = ""
        for success, adc_sample_rate, throttle in zip(successes, rates, throttled, strict=True):
            output += f"{adc_sample_rate} {success} {repeat} {throttle}\n"
        return output

    async def search(
        self, low: float, high: float, step: float, interval: float, max_comparisons: int, slope: float
    ) -> tuple[float, float]:
        """Search for the critical rate."""

        async def compare(adc_sample_rate: float, comparisons: int) -> bool | NoisySearchResult:
            measurement = await self.measure(adc_sample_rate)
            if measurement.state == TrialState.THROTTLED:
                return NoisySearchResult(
                    low=measurement.throttled_adc,
                    high=measurement.throttled_adc,
                    comparisons=comparisons,
                    confidence=100.0,
                )
            else:
                return measurement.state == TrialState.FAILED

        # The additional 0.01 * args.step is to ensure high is included rather
        # than excluded if the range is a multiple of step.
        rates = np.arange(low, high + 0.01 * step, step)
        low_result = await self.measure(rates[0])
        if low_result.state != TrialState.SUCCESS:
            raise RuntimeError(f"failed on low: {low_result.message()}")
        high_result = await self.measure(rates[-1])
        if high_result.state == TrialState.SUCCESS:
            raise RuntimeError(f"succeeded on high: {high_result.message()}")

        mid_rates = 0.5 * (rates[:-1] + rates[1:])  # Rates in the middle of the intervals
        mid_rates = np.r_[low, mid_rates, high]
        l_rates = np.log(rates)[:, np.newaxis]
        l_mid_rates = np.log(mid_rates)[np.newaxis, :]
        noise = expit((l_mid_rates - l_rates) * slope)
        # Don't allow probabilities to get too close to 0/1, as there may be some
        # external factor that makes things go wrong even at very low/high rates.
        noise = np.clip(noise, NOISE, 1 - NOISE)

        result = await noisy_search(
            rates.tolist(),
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

    async def run(self) -> None:
        if self.args.calibrate:
            result = await self.calibrate(self.args.low, self.args.high, self.args.step, self.args.calibrate_repeat)
        elif self.args.oneshot is not None:
            result = (await self.measure(self.args.oneshot)).message()
        else:
            slope = self.slope[min(self.args.n, max(self.slope.keys()))]
            low, high = await self.search(
                low=self.args.low,
                high=self.args.high,
                step=self.args.step,
                interval=self.args.interval,
                max_comparisons=self.args.max_comparisons,
                slope=slope,
            )
            result = f"\n{low / 1e6} MHz - {high / 1e6} MHz"
        print(result)
