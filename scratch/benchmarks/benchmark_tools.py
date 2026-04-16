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
import ipaddress
import logging
import sys
import time
from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from enum import Enum
from typing import override

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
UNSTABLE_RETRIES = 3  #: Maximum number of times to retry a throttled trial
MAXIMUM_RANGES = 2**20
logger = logging.getLogger(__name__)


def compress(addresses: list[ipaddress.IPv4Address | ipaddress.IPv6Address]) -> str:
    """Convert a list of contiguous IP addresses to a string of the form <addr>[+n].

    Raises
    ------
    ValueError
        If `addresses` is empty or non-contiguous
    """
    if not addresses:
        raise ValueError("addresses is empty")
    for i, addr in enumerate(addresses):
        if addr != addresses[0] + i:
            raise ValueError("addresses is not contiguous")
    if len(addresses) == 1:
        return f"{addresses[0]}"
    else:
        return f"{addresses[0]}+{len(addresses) - 1}"


class MulticastGroupAllocator:
    def __init__(self, groups: ipaddress.IPv4Network | ipaddress.IPv6Network) -> None:
        self._iter = groups.hosts()

    def as_list(self, n: int = 1) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
        try:
            # mypy thinks next returns _BaseAddress
            return [next(self._iter) for _ in range(n)]  # type: ignore
        except StopIteration:
            raise RuntimeError("ran out of multicast addresses") from None

    def __call__(self, n: int = 1) -> str:
        return compress(self.as_list(n))


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
    parser.add_argument(
        "--multicast-groups",
        type=ipaddress.ip_network,
        default=ipaddress.ip_network("239.192.128.0/20"),
        help="Multicast groups [%(default)s]",
    )


def process_common_benchmark_arguments(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Validate common command-line arguments shared by benchmark_fgpu and benchmark_xbgpu.

    Parameters
    ----------
    args
        Argument parser to validate arguments from.
    """
    if not args.narrowband:
        args.narrowband_decimation = 1  # Simplifies later logic
    if args.calibrate and args.oneshot is not None:
        parser.error("Cannot specify both --calibrate and --oneshot")
    if args.interval < args.step and args.calibrate is None and args.oneshot is None:
        parser.error("--interval must be greater than or equal to --step")
    total_steps = int(((args.high + 0.01 * args.step) - args.low) / args.step)
    if total_steps > MAXIMUM_RANGES:
        parser.error(
            f"range is too large: total steps:{int(total_steps)}"
            + f" maximum total steps: {MAXIMUM_RANGES}, reduce number of steps by increasing --step"
            + " or decrease the range by adjusting --low and --high."
        )


class ResultState(Enum):
    """State of a benchmark trial."""

    SUCCESS = "success"  # The trial was a success (kept up with the rate)
    THROTTLED = "throttled"  # the trial was throttled (kept up with the rate, but not at the correct data rate)
    FAILED = "failed"  # The trial failed (missing heaps)
    NO_HEAPS = "no heaps"  # The trial did not receive any heaps
    # The trial had more than the expected number of heaps counted outside allowed tolerance.
    TIMING_ERROR = "timing error"


@dataclass
class TrialResult:
    """Result of a single trial."""

    expected_heaps: float  #: Number of heaps expected, based on runtime and data rate
    heaps: int  #: Number of heaps received
    missing_heaps: int  #: Number of heaps reported as missing (gaps in incoming data)

    @property
    def state(self) -> ResultState:
        if self.missing_heaps > 0:
            return ResultState.FAILED
        elif (1.0 - HEAPS_TOL) * self.expected_heaps <= self.heaps <= (1.0 + HEAPS_TOL) * self.expected_heaps:
            return ResultState.SUCCESS
        elif 0 < self.heaps < (1.0 - HEAPS_TOL) * self.expected_heaps:
            return ResultState.THROTTLED
        elif self.heaps == 0:
            return ResultState.NO_HEAPS
        elif self.heaps > (1.0 + HEAPS_TOL) * self.expected_heaps:
            return ResultState.TIMING_ERROR
        else:
            return ResultState.FAILED

    def message(self) -> str:
        """Human-readable description of the outcome."""
        match self.state:
            case ResultState.SUCCESS:
                return "Good"
            case ResultState.FAILED:
                return f"Missing {self.missing_heaps} heaps"
            case ResultState.THROTTLED:
                return f"Throttled to {self.heaps / self.expected_heaps * 100:.1f}% of requested rate"
            case ResultState.TIMING_ERROR:
                return f"There were {self.heaps - self.expected_heaps} more heaps than expected"
            case ResultState.NO_HEAPS:
                return "No heaps"


@dataclass
class MeasuredTrials:
    """Result of a single measurement."""

    state: ResultState
    result: TrialResult

    def message(self) -> str:
        return self.result.message()


@dataclass
class ThrottledMeasurement(MeasuredTrials):
    """Result of a throttled trial."""

    def __init__(self, trials: list[TrialResult], adc_sample_rate: float):
        heap_received_high = max(trials, key=lambda x: x.heaps).heaps
        heap_received_low = min(trials, key=lambda x: x.heaps).heaps
        self.throttled_adc_high = adc_sample_rate * heap_received_high / trials[0].expected_heaps
        self.throttled_adc_low = adc_sample_rate * heap_received_low / trials[0].expected_heaps
        super().__init__(ResultState.THROTTLED, trials[0])

    @override
    def message(self) -> str:
        return f"Throttled to {self.throttled_adc_low / 1e6} ~ {self.throttled_adc_high / 1e6} MHz"


@dataclass
class TimingErrorMeasurement(MeasuredTrials):
    """Result of a complete list of timing error trials."""

    def __init__(self, trials: list[TrialResult]):
        self.heaps_median = np.median([trial.heaps for trial in trials])
        self.heaps_tolerated = trials[0].expected_heaps * (1.0 + HEAPS_TOL)
        super().__init__(ResultState.TIMING_ERROR, trials[0])

    @override
    def message(self) -> str:
        return f"Received heaps {self.heaps_median} are over expected tolerable number of heaps: {self.heaps_tolerated}"


def _find_measured_result(trials: list[TrialResult], adc_sample_rate: float) -> MeasuredTrials:
    if len(trials) == 1:
        return MeasuredTrials(state=trials[0].state, result=trials[0])
    if all(trial.state == ResultState.THROTTLED for trial in trials):
        return ThrottledMeasurement(trials, adc_sample_rate)
    elif all(trial.state == ResultState.TIMING_ERROR for trial in trials):
        return TimingErrorMeasurement(trials)
    else:
        return MeasuredTrials(state=ResultState.TIMING_ERROR, result=trials[0])


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
        logging.basicConfig(
            level=logging.DEBUG if args.verbose >= 3 else logging.INFO if args.verbose >= 2 else logging.WARNING
        )
        self.multicast_allocator = MulticastGroupAllocator(args.multicast_groups)

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

    def reset(self) -> None:
        self.multicast_allocator = MulticastGroupAllocator(self.args.multicast_groups)

    async def process(self, adc_sample_rate: float) -> TrialResult:
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
        return TrialResult(
            expected_heaps=expected_heaps,
            heaps=new_heaps - orig_heaps,
            missing_heaps=new_missing - orig_missing,
        )

    async def trial(self, adc_sample_rate: float) -> TrialResult:
        """Perform a single trial."""

        sync_time = int(time.time())
        self.reset()
        async with await self.run_producers(adc_sample_rate, sync_time):
            async with await self.run_consumers(adc_sample_rate, sync_time):
                return await self.process(adc_sample_rate)

    async def measure(self, adc_sample_rate: float) -> MeasuredTrials:
        """Perform a single trial.
        Returns a :class:`MeasureResult` describing the outcome of the trial.
        """

        if self.verbose_results():
            print(f"Testing {adc_sample_rate / 1e6} MHz... ", end="", flush=True, file=sys.stderr)

        results: list[TrialResult] = []
        for _ in range(UNSTABLE_RETRIES):
            result = await self.trial(adc_sample_rate)
            results.append(result)
            if not (result.state == ResultState.THROTTLED or result.state == ResultState.TIMING_ERROR):
                break

        measured_result = _find_measured_result(results, adc_sample_rate)

        if self.verbose_results():
            print(measured_result.message(), file=sys.stderr)

        return measured_result

    async def calibrate(self, low: float, high: float, step: float, repeat: int) -> str:
        """Run multiple trials on all the possible rates."""
        rates = np.arange(low, high + 0.01 * step, step).tolist()
        successes = [0] * len(rates)
        throttled = [0] * len(rates)
        for trial in range(repeat):
            for j, adc_sample_rate in enumerate(rates):
                if self.verbose_results():
                    print(f"Testing {adc_sample_rate / 1e6} MHz... ", end="", flush=True, file=sys.stderr)
                measurement = await self.measure(adc_sample_rate)
                match measurement.state:
                    case ResultState.SUCCESS:
                        successes[j] += 1
                    case ResultState.THROTTLED:
                        throttled[j] += 1
                    case ResultState.FAILED:
                        pass
                    case _:
                        raise RuntimeError("Measurement failed: " + measurement.message())
                if self.verbose_results():
                    print(
                        f"{measurement.message()}, {successes[j]}/{trial + 1} passed, {throttled[j]} throttled\n",
                    )
        output = ""
        for success, adc_sample_rate, throttle in zip(successes, rates, throttled, strict=True):
            output += f"{adc_sample_rate} {success} {repeat} {throttle}\n"
        return output

    def _throttled_result(
        self, measurement: ThrottledMeasurement, comparisons: int, rates: np.ndarray
    ) -> NoisySearchResult:
        low = int((np.abs(rates - measurement.throttled_adc_low)).argmin())
        high = int((np.abs(rates - measurement.throttled_adc_high)).argmin())

        return NoisySearchResult(
            low=low,
            high=high,
            comparisons=comparisons,
            confidence=1.0,
        )

    async def _search(
        self,
        low: float,
        high: float,
        step: float,
        interval: float,
        max_comparisons: int,
        slope: float,
        rates: np.ndarray,
    ) -> NoisySearchResult:
        """Search for the critical rate."""

        async def compare(adc_sample_rate: float, comparisons: int) -> bool | NoisySearchResult:
            measurement = await self.measure(adc_sample_rate)
            if isinstance(measurement, ThrottledMeasurement):
                return self._throttled_result(measurement, comparisons, rates)
            if measurement.state == ResultState.SUCCESS:
                return False
            if measurement.state == ResultState.FAILED:
                return True
            raise RuntimeError("Measurement failed: " + measurement.message())

        low_result = await self.measure(rates[0])
        if low_result.state != ResultState.SUCCESS:
            raise RuntimeError(f"failed on low: {low_result.message()}")
        high_result = await self.measure(rates[-1])
        if high_result.state == ResultState.SUCCESS:
            raise RuntimeError(f"succeeded on high: {high_result.message()}")
        if isinstance(high_result, ThrottledMeasurement):
            return self._throttled_result(high_result, max_comparisons, rates)

        mid_rates = 0.5 * (rates[:-1] + rates[1:])  # Rates in the middle of the intervals
        mid_rates = np.r_[low, mid_rates, high]
        l_rates = np.log(rates)[:, np.newaxis]
        l_mid_rates = np.log(mid_rates)[np.newaxis, :]
        noise = expit((l_mid_rates - l_rates) * slope)
        # Don't allow probabilities to get too close to 0/1, as there may be some
        # external factor that makes things go wrong even at very low/high rates.
        noise = np.clip(noise, NOISE, 1 - NOISE)

        return await noisy_search(
            rates.tolist(),
            noise,
            TOLERANCE,
            compare,
            max_interval=round(interval / step),
            max_comparisons=max_comparisons,
        )

    async def search(
        self, low: float, high: float, step: float, interval: float, max_comparisons: int, slope: float
    ) -> tuple[float, float]:
        # The additional 0.01 * args.step is to ensure high is included rather
        # than excluded if the range is a multiple of step.
        rates = np.arange(low, high + 0.01 * step, step)

        result = await self._search(low, high, step, interval, max_comparisons, slope, rates)
        if result.low == -1:
            raise RuntimeError("lower bound is too high")
        elif result.high == len(rates) - 1:
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
