#!/usr/bin/env python3

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

"""Benchmark critical rate of fgpu.

See :doc:`benchmarking`.
"""

import argparse
import asyncio
import functools
import sys
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass

import aiohttp.client
import asyncssh
import numpy as np
from prometheus_client.parser import text_string_to_metric_families
from scipy.special import expit

from noisy_search import noisy_search
from remote import Server, ServerInfo, run_tasks, servers_from_toml

HEAPS_TOL = 0.05  #: Relative tolerance for number of heaps received
N_POLS = 2
DEFAULT_IMAGE = "harbor.sdp.kat.ac.za/dpp/katgpucbf"
NOISE = 0.01  #: Minimum probability of an incorrect result from each trial
TOLERANCE = 0.001  #: Complement of confidence interval probability
#: Verbosity level at which individual test results are reported
VERBOSE_RESULTS = 1


def dsim_factory(
    server: Server,
    server_info: ServerInfo,
    conn: asyncssh.SSHClientConnection,
    index: int,
    *,
    adc_sample_rate: float,
    single_pol: bool,
    sync_time: int,
    args: argparse.Namespace,
) -> str:
    """Generate command to run dsim."""
    ncpus = len(server_info.cpus)
    step = ncpus // args.n
    if single_pol:
        step //= 2
    my_cpus = server_info.cpus[index * step : (index + 1) * step]
    if args.n == 1 or not single_pol:
        interface = server.interfaces[index % len(server.interfaces)]
    else:
        # For larger n, send the two pols over the same interface
        # (because fgpu_factory expects them to arrive on the same interface)
        interface = server.interfaces[index // 2 % len(server.interfaces)]
    katcp_port = 7140 + index
    prometheus_port = 7250 + index
    name = f"feng-dsim-{index}"
    if single_pol:
        addresses = f"239.102.{index // 2}.{64 + index % 2 * 8}+7:7148"
    else:
        addresses = f"239.102.{index}.64+7:7148 239.102.{index}.72+7:7148"
    command = (
        "docker run "
        f"--name={name} --cap-add=SYS_NICE --net=host "
        + "".join(f"--device={dev}:{dev} " for dev in server_info.infiniband_devices)
        + f"--ulimit=memlock=-1 --rm {args.image} "
        f"taskset -c {my_cpus[0]} "
        f"dsim --affinity={my_cpus[1]} "
        "--ibv "
        f"--interface={interface} "
        f"--adc-sample-rate={adc_sample_rate} "
        f"--heap-samples={args.dig_heap_samples} "
        "--ttl=2 "
        "--period=16777216 "  # Speeds things up
        f"--katcp-port={katcp_port} "
        f"--prometheus-port={prometheus_port} "
        f"--sync-time={sync_time} "
        f"--first-id={index if single_pol else 2 * index} "
        f"{addresses} "
    )
    if args.dig_sample_bits is not None:
        command += f"--sample-bits={args.dig_sample_bits} "
    return command


def fgpu_factory(
    server: Server,
    server_info: ServerInfo,
    conn: asyncssh.SSHClientConnection,
    index: int,
    *,
    adc_sample_rate: float,
    sync_time: int,
    args: argparse.Namespace,
) -> str:
    """Generate command to run fgpu."""
    n = args.n
    # When we run > 4, we assume we have enough RAM (GPU and host) that we
    # don't need to scale buffers down to tiny amounts.
    scaling_n = min(n, 4)
    step = len(server_info.cpus) // n
    hstep = step // 2
    qstep = step // 4
    my_cpus = server_info.cpus[index * step : (index + 1) * step]
    recv_chunk_samples = 2**27 // scaling_n
    send_chunk_jones = recv_chunk_samples // 4
    if n == 1:
        interface = ",".join(server.interfaces[:2])
        recv_affinity = f"0,1,{qstep},{qstep + 1}"
        send_affinity = f"{2 * qstep}"
        other_affinity = f"{3 * qstep}"
    elif n == 2:
        interface = server.interfaces[index % len(server.interfaces)]
        recv_affinity = f"{my_cpus[0]},{my_cpus[hstep]}"
        send_affinity = f"{my_cpus[qstep]}"
        other_affinity = f"{my_cpus[hstep + qstep]}"
    else:
        interface = server.interfaces[index % len(server.interfaces)]
        recv_affinity = f"{my_cpus[0]}"
        send_affinity = f"{my_cpus[hstep]}"
        other_affinity = f"{my_cpus[hstep + 1]}"
    gpu = server.gpus[index % len(server.gpus)]

    katcp_port = 7140 + index
    prometheus_port = 7250 + index
    name = f"fgpu-{index}"
    wideband_kwargs = {
        "name": "wideband",
        "channels": args.channels,
        "dst": f"239.102.{200 + index}.0+{args.xb - 1}:7148",
    }
    if args.jones_per_batch is not None:
        wideband_kwargs["jones_per_batch"] = args.jones_per_batch
    wideband_arg = ",".join(f"{key}={value}" for key, value in wideband_kwargs.items())
    command = (
        "docker run "
        f"--name={name} --cap-add=SYS_NICE --runtime=nvidia --gpus=device={gpu} --net=host "
        f"-e NVIDIA_MOFED=enabled --ulimit=memlock=-1 --rm "
        f" {' '.join(args.fgpu_docker_arg)} {args.image} "
        f"schedrr taskset -c {other_affinity} fgpu "
        f"--recv-packet-samples={args.dig_heap_samples} "
        f"--recv-chunk-samples={recv_chunk_samples} --send-chunk-jones={send_chunk_jones} "
        f"--recv-buffer={256 * 1024 * 1024 // scaling_n} "
        f"--recv-interface={interface} --recv-ibv "
        f"--send-interface={interface} --send-ibv "
        f"--recv-affinity={recv_affinity} --recv-comp-vector={recv_affinity} "
        f"--send-affinity={send_affinity} --send-comp-vector={send_affinity} "
        f"--adc-sample-rate={adc_sample_rate} "
        f"--katcp-port={katcp_port} "
        f"--prometheus-port={prometheus_port} "
        f"--sync-time={sync_time} "
        f"--feng-id={index} "
        f"{'--use-vkgdr' if args.use_vkgdr else ''} "
        f"--wideband={wideband_arg} "
        f"239.102.{index}.64+15:7148 "
    )
    if args.narrowband:
        narrowband_kwargs = {
            "name": "narrowband",
            "channels": args.narrowband_channels,
            "decimation": args.narrowband_decimation,
            "centre_frequency": adc_sample_rate / 4,
            "dst": f"239.102.{216 + index}.0+{args.xb // args.narrowband_decimation - 1}:7148",
        }
        if args.jones_per_batch is not None:
            narrowband_kwargs["jones_per_batch"] = args.jones_per_batch
        narrowband_arg = ",".join(f"{key}={value}" for key, value in narrowband_kwargs.items())
        command += f"--narrowband={narrowband_arg} "
    for arg in ["array_size", "dig_sample_bits"]:
        value = getattr(args, arg)
        if value is not None:
            dashed = arg.replace("_", "-")
            command += f"--{dashed}={value} "
    for extra in args.extra:
        command += f"{extra} "
    return command


async def run_dsims(
    adc_sample_rate: float,
    sync_time: int,
    args: argparse.Namespace,
) -> AsyncExitStack:
    """Run all the dsims.

    The result must be used as a context manager. Exiting the context manager
    will shut down the tasks.
    """
    single_pol = False
    n = args.n
    if n <= 2:
        n *= 2
        single_pol = True
    factory = functools.partial(
        dsim_factory,
        adc_sample_rate=adc_sample_rate,
        single_pol=single_pol,
        sync_time=sync_time,
        args=args,
    )
    return await run_tasks(args.dsim_server, n, factory, args.image, port_base=7140, verbose=args.verbose)


async def run_fgpus(
    adc_sample_rate: float,
    sync_time: int,
    args: argparse.Namespace,
) -> AsyncExitStack:
    """Run all the fgpu instances.

    The result must be used as a context manager. Exiting the context manager
    will shut down the tasks.
    """
    factory = functools.partial(
        fgpu_factory,
        adc_sample_rate=adc_sample_rate,
        sync_time=sync_time,
        args=args,
    )
    return await run_tasks(args.fgpu_server, args.n, factory, args.image, port_base=7140, verbose=args.verbose)


async def _heap_counts1(session: aiohttp.client.ClientSession, url: str) -> tuple[int, int]:
    heaps = 0
    missing_heaps = 0
    async with session.get(url) as resp:
        for family in text_string_to_metric_families(await resp.text()):
            for sample in family.samples:
                if sample.name == "fgpu_input_heaps_total":
                    heaps += int(sample.value)
                elif sample.name == "fgpu_input_missing_heaps_total":
                    missing_heaps += int(sample.value)
    return (heaps, missing_heaps)


async def heap_counts(session: aiohttp.client.ClientSession, server: Server, n: int) -> tuple[int, int]:
    """Query the number of heaps received and missing, for all n fgpu instances."""
    async with asyncio.TaskGroup() as tg:
        tasks = [
            tg.create_task(_heap_counts1(session, f"http://{server.hostname}:{7250 + i}/metrics")) for i in range(n)
        ]
    partials = [task.result() for task in tasks]
    heaps, missing_heaps = zip(*partials, strict=True)
    return sum(heaps), sum(missing_heaps)


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


async def process(
    adc_sample_rate: float,
    args: argparse.Namespace,
) -> Result:
    """Perform a single trial on running engines."""
    async with aiohttp.client.ClientSession() as session:
        await asyncio.sleep(args.startup_time)  # Give a chance for startup losses
        async with asyncio.TaskGroup() as tg:
            # Start the sleep *before* entering heap_counts, so that time spent
            # during heap_counts is considered part of the sleep time.
            tg.create_task(asyncio.sleep(args.runtime))
            orig_heaps, orig_missing = await heap_counts(session, args.fgpu_server, args.n)
        new_heaps, new_missing = await heap_counts(session, args.fgpu_server, args.n)

    expected_heaps = args.runtime * adc_sample_rate * args.n * N_POLS / args.dig_heap_samples
    return Result(
        expected_heaps=expected_heaps,
        heaps=new_heaps - orig_heaps,
        missing_heaps=new_missing - orig_missing,
    )


async def trial(adc_sample_rate: float, args: argparse.Namespace) -> Result:
    """Perform a single trial."""
    sync_time = int(time.time())
    async with await run_dsims(adc_sample_rate, sync_time, args):
        async with await run_fgpus(adc_sample_rate, sync_time, args):
            return await process(adc_sample_rate, args)
    raise AssertionError("should be unreachable")


async def calibrate(args: argparse.Namespace) -> None:
    """Run multiple trials on all the possible rates."""
    rates = np.arange(args.low, args.high + 0.01 * args.step, args.step)
    successes = [0] * len(rates)
    for trial in range(args.calibrate_repeat):
        for j, adc_sample_rate in enumerate(rates):
            sync_time = int(time.time())
            redo = True
            while redo:
                redo = False
                if args.verbose >= VERBOSE_RESULTS:
                    print(f"Testing {adc_sample_rate / 1e6} MHz... ", end="", flush=True, file=sys.stderr)
                async with await run_dsims(adc_sample_rate, sync_time, args):
                    async with await run_fgpus(adc_sample_rate, sync_time, args):
                        result = await process(adc_sample_rate, args)
                if result.good():
                    successes[j] += 1
                elif result.missing_heaps == 0:
                    redo = True  # Unexpected number of heaps received
                if args.verbose >= VERBOSE_RESULTS:
                    if redo:
                        print(f"{result.message()}, rerunning", flush=True, file=sys.stderr)
                    else:
                        print(f"{result.message()}, {successes[j]}/{trial + 1} passed", flush=True, file=sys.stderr)
    for success, adc_sample_rate in zip(successes, rates, strict=True):
        print(adc_sample_rate, success, args.calibrate_repeat)


async def search(args: argparse.Namespace) -> tuple[float, float]:
    """Search for the critical rate."""

    async def measure(adc_sample_rate: float) -> Result:
        while True:
            if args.verbose >= VERBOSE_RESULTS:
                print(f"Testing {adc_sample_rate / 1e6} MHz... ", end="", flush=True, file=sys.stderr)
            result = await trial(adc_sample_rate, args)
            if not result.good() and result.missing_heaps == 0:
                if args.verbose >= VERBOSE_RESULTS:
                    print(f"{result.message()}, re-running", file=sys.stderr)
            else:
                if args.verbose >= VERBOSE_RESULTS:
                    print(result.message(), file=sys.stderr)
                return result

    async def compare(adc_sample_rate: float) -> bool:
        return not (await measure(adc_sample_rate)).good()

    # The additional 0.01 * args.step is to ensure high is included rather
    # than excluded if the range is a multiple of step.
    rates = np.arange(args.low, args.high + 0.01 * args.step, args.step)
    low_result = await measure(rates[0])
    if not low_result.good():
        raise RuntimeError(f"failed on low: {low_result.message()}")
    high_result = await measure(rates[-1])
    if high_result.good():
        raise RuntimeError("succeeded on high")

    # Compute error estimate. The model is a logistic regression on the
    # log of the sample rate (log is used mainly to make the slope invariant
    # to the scale of the rates, rather than for the shape).
    # The magic numbers are determined from fit.py. For n > 4 we don't have
    # data, so just assume it is the same as for n = 4.
    slope = {
        1: -342.212919,
        2: -173.264274,
        4: -582.668296,
    }[min(args.n, 4)]
    mid_rates = 0.5 * (rates[:-1] + rates[1:])  # Rates in the middle of the intervals
    mid_rates = np.r_[args.low, mid_rates, args.high]
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
        max_interval=round(args.interval / args.step),
        max_comparisons=args.max_comparisons,
    )
    if result.low == -1:
        raise RuntimeError("lower bound is too high")
    elif result.high == len(rates):
        raise RuntimeError("upper bound is too low")
    else:
        return rates[result.low], rates[result.high]


async def oneshot(args: argparse.Namespace):
    """Measure at a single rate."""

    adc_sample_rate = args.oneshot
    while True:
        result = await trial(adc_sample_rate, args)
        if not result.good() and result.missing_heaps == 0:
            if args.verbose >= VERBOSE_RESULTS:
                print(f"{result.message()}, re-running", file=sys.stderr)
        else:
            print(result.message())
            break


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=4, help="Number of engines [%(default)s]")
    parser.add_argument("--channels", type=int, default=1024, help="Wideband channel count [%(default)s]")
    parser.add_argument(
        "--use-vkgdr",
        action="store_true",
        help="Assemble chunks directly in GPU memory (requires sufficient GPU BAR space)",
    )
    parser.add_argument(
        "--array-size",
        type=int,
        help="The number of antennas in the array.",
    )
    parser.add_argument(
        "--jones-per-batch",
        type=int,
        metavar="SAMPLES",
        help="Jones vectors in each output batch",
    )
    parser.add_argument(
        "--dig-heap-samples",
        type=int,
        metavar="SAMPLES",
        default=4096,
        help="Number of samples in each digitiser heap",
    )
    parser.add_argument(
        "--dig-sample-bits",
        type=int,
        metavar="BITS",
        help="Number of bits per digitised sample",
    )
    parser.add_argument("--narrowband", action="store_true", help="Enable a narrowband output [no]")
    parser.add_argument(
        "--narrowband-decimation", type=int, default=8, help="Narrowband decimation factor [%(default)s]"
    )
    parser.add_argument("--narrowband-channels", type=int, default=32768, help="Narrowband channels [%(default)s]")
    parser.add_argument("--xb", type=int, default=64, help="Number of XB-engines [%(default)s]")
    parser.add_argument("--fgpu-docker-arg", action="append", default=[], help="Add Docker argument for invoking fgpu")

    parser.add_argument(
        "--startup-time", type=float, default=1.0, help="Time to run before starting measurement [%(default)s]"
    )
    parser.add_argument("--runtime", type=float, default=20.0, help="Time to let engine run (s) [%(default)s]")
    parser.add_argument("--low", type=float, default=1500e6, help="Minimum ADC sample rate to search [%(default)s]")
    parser.add_argument("--high", type=float, default=2200e6, help="Maximum ADC sample rate to search [%(default)s]")
    parser.add_argument("--step", type=float, default=1e6, help="Step size between sample rates to test [%(default)s]")
    parser.add_argument("--interval", type=float, default=20e6, help="Target confidence interval [%(default)s]")
    parser.add_argument("--max-comparisons", type=int, default=40, help="Maximum comparisons to make [%(default)s]")
    parser.add_argument("--image", type=str, default=DEFAULT_IMAGE, help="Docker image [%(default)s]")
    parser.add_argument("--servers", type=str, default="servers.toml", help="Server description file [%(default)s]")
    parser.add_argument("--dsim-server", type=str, default="dsim", help="Server on which to run dsims [%(default)s]")
    parser.add_argument("--fgpu-server", type=str, default="fgpu", help="Server on which to run fgpu [%(default)s]")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity [no]")
    parser.add_argument(
        "--calibrate", action="store_true", help="Run at multiple rates to calibrate expectations [%(default)s]"
    )
    parser.add_argument(
        "--calibrate-repeat", type=int, default=100, help="Number of times to run at each rate [%(default)s]"
    )
    parser.add_argument("--oneshot", type=float, help="Run one test at the given sampling rate")
    parser.add_argument("extra", nargs="*", help="Remaining arguments are passed to fgpu")
    args = parser.parse_args()
    if args.calibrate and args.oneshot is not None:
        parser.error("Cannot specify both --calibrate and --oneshot")

    servers = servers_from_toml(args.servers)
    args.dsim_server = servers[args.dsim_server]
    args.fgpu_server = servers[args.fgpu_server]

    if args.calibrate:
        await calibrate(args)
    elif args.oneshot is not None:
        await oneshot(args)
    else:
        low, high = await search(args)
        print(f"\n{low / 1e6} MHz - {high / 1e6} MHz")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except asyncio.CancelledError:
        pass
