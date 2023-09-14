#!/usr/bin/env python3

import argparse
import asyncio
import functools
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass

import aiohttp.client
import asyncssh
import numpy as np
from prometheus_client.parser import text_string_to_metric_families

from noisy_search import noisy_search
from remote import Server, ServerInfo, run_tasks, servers_from_toml
from sighandler import add_sigint_handler

HEAPS_TOL = 0.05  #: Relative tolerance for number of heaps received
SAMPLES_PER_HEAP = 4096
N_POLS = 2
DEFAULT_IMAGE = "harbor.sdp.kat.ac.za/cbf/katgpucbf"
NOISE = 0.02  #: Probability of an incorrect result from each trial
#: Minimum relative difference for comparisons to be reliable i.e. for the
#: failure probability to be :const:`NOISE`.
FUZZ = 0.005
TOLERANCE = 0.001  #: Complement of confidence interval probability


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
        src_affinity = f"0,1,{ncpus_per_quadrant},{ncpus_per_quadrant+1}"
        dst_affinity = f"{2 * ncpus_per_quadrant}"
        other_affinity = f"{3 * ncpus_per_quadrant}"
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
        f"-e NVIDIA_MOFED=enabled --ulimit=memlock=-1 --rm {args.image} "
        f"schedrr taskset -c {other_affinity} fgpu "
        f"--src-chunk-samples={src_chunk_samples} --dst-chunk-jones={dst_chunk_jones} "
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
        f"--wideband=name=wideband,channels={args.channels},dst=239.102.{200 + index}.0+63:7148 "
        # "--narrowband=name=narrowband,channels=32768,decimation=8,"
        # f"ddc_taps=96,centre_frequency=200e6,dst=239.102.{216 + index}.0+7:7148 "
        f"239.102.{index}.64+15:7148 "
    )
    if args.narrowband:
        command += (
            f"--narrowband=name=narrowband,channels={args.narrowband_channels},"
            f"decimation={args.narrowband_decimation},centre_frequency={adc_sample_rate / 4},"
            f"dst=239.102.{216 + index}.0+7:7148 "
        )
    for arg in ["spectra_per_heap", "array_size", "dig_sample_bits"]:
        value = getattr(args, arg)
        if value is not None:
            dashed = arg.replace("_", "-")
            command += f"--{dashed}={value} "
    return command


async def run_dsims(
    adc_sample_rate: float,
    sync_time: int,
    args: argparse.Namespace,
) -> AsyncExitStack:
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
    tasks = [
        asyncio.create_task(_heap_counts1(session, f"http://{server.hostname}:{7150 + i}/metrics")) for i in range(n)
    ]
    partials = await asyncio.gather(*tasks)
    heaps, missing_heaps = zip(*partials)
    return sum(heaps), sum(missing_heaps)


@dataclass
class Result:
    expected_heaps: float
    heaps: int
    missing_heaps: int

    def good(self) -> bool:
        if self.missing_heaps > 0:
            return False
        return (1.0 - HEAPS_TOL) * self.expected_heaps <= self.heaps <= (1.0 + HEAPS_TOL) * self.expected_heaps

    def message(self) -> str:
        if self.missing_heaps > 0:
            return f"Missing {self.missing_heaps} heaps"
        elif not self.good():
            return f"Expected ±{self.expected_heaps}, received {self.heaps}"
        else:
            return "Good"


async def process(
    adc_sample_rate: float,
    n: int,
    runtime: float,
    fgpu_server: Server,
) -> Result:
    async with aiohttp.client.ClientSession() as session:
        await asyncio.sleep(1)  # Give a chance for startup losses
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
    sync_time = int(time.time())
    async with await run_fgpus(adc_sample_rate, sync_time, args):
        async with await run_dsims(adc_sample_rate, sync_time, args):
            return await process(adc_sample_rate, args.n, args.runtime, args.fgpu_server)
    raise AssertionError("should be unreachable")


async def calibrate(args: argparse.Namespace) -> None:
    for adc_sample_rate in np.arange(args.low, args.high + 0.01 * args.step, args.step):
        sync_time = int(time.time())
        async with await run_dsims(adc_sample_rate, sync_time, args):
            trials = 0
            successes = 0
            while trials < args.calibrate_repeat:
                async with await run_fgpus(adc_sample_rate, sync_time, args):
                    if args.verbose:
                        print(f"Testing {adc_sample_rate / 1e6} MHz... ", end="", flush=True)
                    result = await process(adc_sample_rate, args.n, args.runtime, args.fgpu_server)
                    if args.verbose:
                        print(result.message())
                    if result.good():
                        trials += 1
                        successes += 1
                    elif result.missing_heaps > 0:
                        # If missing == 0, we need to rerun the experiment
                        trials += 1
            print(adc_sample_rate, successes / trials)


async def search(args: argparse.Namespace) -> tuple[float, float]:
    async def measure(adc_sample_rate: float) -> Result:
        while True:
            print(f"Testing {adc_sample_rate / 1e6} MHz... ", end="", flush=True)
            result = await trial(adc_sample_rate, args)
            if not result.good() and result.missing_heaps == 0:
                print(f"{result.message()}, re-running")
            else:
                print(result.message())
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

    # Compute error estimate. This is a heuristic based on gut feel rather
    # than measurement. Assume that results are 1-NOISE reliable when at least
    # FUZZ away from the critical value, and vary logarithmically in between.
    mid_rates = 0.5 * (rates[:-1] + rates[1:])  # Rates in the middle of the intervals
    mid_rates = np.r_[args.low, mid_rates, args.high]
    l_rates = np.log(rates)[:, np.newaxis]
    l_mid_rates = np.log(mid_rates)[np.newaxis, :]
    noise = np.clip(0.5 + (l_rates - l_mid_rates) / np.log1p(FUZZ) * (0.5 - NOISE), NOISE, 1 - NOISE)

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


async def main():
    add_sigint_handler()
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=4, help="Number of engines [%(default)s]")
    parser.add_argument("--channels", type=int, default=32768, help="Wideband channel count [%(default)s]")
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
    parser.add_argument("-v", "--verbose", action="store_true", help="Emit stdout/stderr [no]")
    parser.add_argument(
        "--calibrate", action="store_true", help="Run at multiple rates to calibrate expectations [%(default)s]"
    )
    parser.add_argument(
        "--calibrate-repeat", type=int, default=100, help="Number of times to run at each rate [%(default)s]"
    )
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
