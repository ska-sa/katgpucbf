#!/usr/bin/env python3

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
from sighandler import add_sigint_handler

HEAPS_TOL = 0.05  #: Relative tolerance for number of heaps received
SAMPLES_PER_HEAP = 4096
N_POLS = 2
DEFAULT_IMAGE = "harbor.sdp.kat.ac.za/cbf/katgpucbf"
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
    ncpus_per_quadrant = server_info.ncpus // 4
    cpu_base = index * ncpus_per_quadrant
    interface = server.interfaces[index % len(server.interfaces)]
    katcp_port = 7140 + index
    prometheus_port = 7150 + index
    name = f"feng-dsim-{index}"
    if single_pol:
        addresses = ["239.102.0.64+7:7148", "239.102.0.72+7:7148"][index]
    else:
        addresses = f"239.102.{index}.64+7:7148 239.102.{index}.72+7:7148"
    command = (
        "docker run "
        f"--name={name} --cap-add=SYS_NICE --runtime=nvidia --net=host "
        f"-e NVIDIA_MOFED=enabled --ulimit=memlock=-1 --rm {args.image} "
        f"taskset -c {cpu_base} "
        f"dsim --affinity={cpu_base + 1} "
        "--ibv "
        f"--interface={interface} "
        f"--adc-sample-rate={adc_sample_rate} "
        "--ttl=2 "
        "--period=16777216 "  # Speeds things up
        f"--katcp-port={katcp_port} "
        f"--prometheus-port={prometheus_port} "
        f"--sync-time={sync_time} "
        f"--first-id={index} "
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
    ncpus_per_quadrant = server_info.ncpus // 4
    hstep = ncpus_per_quadrant // 2
    cpu_base = index * ncpus_per_quadrant
    if args.use_vkgdr:
        src_chunk_samples = 2**24 // n
        dst_chunk_jones = src_chunk_samples // 2
    else:
        src_chunk_samples = 2**27 // n
        dst_chunk_jones = src_chunk_samples // 4
    if n == 1:
        interface = ",".join(server.interfaces[:2])
        src_affinity = (
            f"0,1,2,3,{ncpus_per_quadrant},{ncpus_per_quadrant+1},{ncpus_per_quadrant+2},{ncpus_per_quadrant+3}"
        )
        dst_affinity = f"{2 * ncpus_per_quadrant}"
        other_affinity = f"{3 * ncpus_per_quadrant}"
    elif n == 2:
        interface = server.interfaces[index % len(server.interfaces)]
        cpu_base *= 2
        src_affinity = f"{cpu_base},{cpu_base + 1},{cpu_base + ncpus_per_quadrant},{cpu_base + ncpus_per_quadrant + 1}"
        dst_affinity = f"{cpu_base + hstep}"
        other_affinity = f"{cpu_base + ncpus_per_quadrant + hstep}"
    else:
        interface = server.interfaces[index % len(server.interfaces)]
        src_affinity = f"{cpu_base}"
        dst_affinity = f"{cpu_base + hstep}"
        other_affinity = f"{cpu_base + hstep + 1}"
    katcp_port = 7140 + index
    prometheus_port = 7150 + index
    name = f"fgpu-{index}"
    command = (
        "docker run "
        f"--name={name} --cap-add=SYS_NICE --runtime=nvidia --gpus=all --net=host "
        f"-e NVIDIA_MOFED=enabled --ulimit=memlock=-1 --rm "
        f" {' '.join(args.fgpu_docker_arg)} {args.image} "
        f"schedrr taskset -c {other_affinity} fgpu "
        f"--src-chunk-samples={src_chunk_samples} --dst-chunk-jones={dst_chunk_jones} "
        f"--src-buffer={256 * 1024 * 1024 // n} "
        f"--src-interface={interface} --src-ibv "
        f"--dst-interface={interface} --dst-ibv "
        f"--src-affinity={src_affinity} --src-comp-vector={src_affinity} "
        f"--dst-affinity={dst_affinity} --dst-comp-vector={dst_affinity} "
        f"--adc-sample-rate={adc_sample_rate} "
        f"--katcp-port={katcp_port} "
        f"--prometheus-port={prometheus_port} "
        f"--sync-epoch={sync_time} "
        f"--feng-id={index} "
        f"{'--use-vkgdr --max-delay-diff=65536' if args.use_vkgdr else ''} "
        f"--wideband=name=wideband,channels={args.channels},dst=239.102.{200 + index}.0+{args.xb - 1}:7148 "
        f"239.102.{index}.64+15:7148 "
    )
    if args.narrowband:
        command += (
            f"--narrowband=name=narrowband,channels={args.narrowband_channels},"
            f"decimation={args.narrowband_decimation},centre_frequency={adc_sample_rate / 4},"
            f"dst=239.102.{216 + index}.0+{args.xb // args.narrowband_decimation - 1}:7148 "
        )
    for arg in ["spectra_per_heap", "array_size", "dig_sample_bits"]:
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
    if n == 1:
        n = 2
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
    tasks = [
        asyncio.create_task(_heap_counts1(session, f"http://{server.hostname}:{7150 + i}/metrics")) for i in range(n)
    ]
    partials = await asyncio.gather(*tasks)
    heaps, missing_heaps = zip(*partials)
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
    n: int,
    startup_time: float,
    runtime: float,
    fgpu_server: Server,
) -> Result:
    """Perform a single trial on running engines."""
    async with aiohttp.client.ClientSession() as session:
        await asyncio.sleep(startup_time)  # Give a chance for startup losses
        sleeper = asyncio.create_task(asyncio.sleep(runtime))
        orig_heaps, orig_missing = await heap_counts(session, fgpu_server, n)
        await sleeper
        new_heaps, new_missing = await heap_counts(session, fgpu_server, n)

    expected_heaps = runtime * adc_sample_rate * n * N_POLS / SAMPLES_PER_HEAP
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
            return await process(adc_sample_rate, args.n, args.startup_time, args.runtime, args.fgpu_server)
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
                        result = await process(
                            adc_sample_rate, args.n, args.startup_time, args.runtime, args.fgpu_server
                        )
                if result.good():
                    successes[j] += 1
                elif result.missing_heaps == 0:
                    redo = True  # Unexpected number of heaps received
                if args.verbose >= VERBOSE_RESULTS:
                    if redo:
                        print(f"{result.message()}, rerunning", flush=True, file=sys.stderr)
                    else:
                        print(f"{result.message()}, {successes[j]}/{trial + 1} passed", flush=True, file=sys.stderr)
    for success, adc_sample_rate in zip(successes, rates):
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
    # The magic numbers are determined from fit.py.
    slope = {
        1: -342.212919,
        2: -400.0,  # TODO: need to measure
        4: -582.668296,
    }[args.n]
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


async def main():  # noqa: D103
    add_sigint_handler()
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
        "--spectra-per-heap",
        type=int,
        metavar="SPECTRA",
        help="Spectra in each output heap",
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
    parser.add_argument("extra", nargs="*", help="Remaining arguments are passed to fgpu")
    args = parser.parse_args()

    servers = servers_from_toml(args.servers)
    args.dsim_server = servers[args.dsim_server]
    args.fgpu_server = servers[args.fgpu_server]

    if args.calibrate:
        await calibrate(args)
    else:
        low, high = await search(args)
        print(f"\n{low / 1e6} MHz - {high / 1e6} MHz")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except asyncio.CancelledError:
        pass
