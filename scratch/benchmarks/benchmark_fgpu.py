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
from contextlib import AsyncExitStack

import asyncssh

from katgpucbf import N_POLS

from benchmark_tools import DEFAULT_IMAGE, PROMETHEUS_PORT_BASE, Benchmark
from remote import Server, ServerInfo, run_tasks, servers_from_toml

KATCP_PORT_BASE = 7140


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
    katcp_port = KATCP_PORT_BASE + index
    prometheus_port = PROMETHEUS_PORT_BASE + index
    name = f"feng-dsim-{index}"
    if single_pol:
        addresses = f"239.102.{index // 2}.{64 + index % 2 * 8}+7:7148"
    else:
        addresses = f"239.102.{index}.64+7:7148 239.102.{index}.72+7:7148"
    command = (
        "docker run "
        f"--name={name} --cap-add=SYS_NICE --net=host --stop-timeout=2 "
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

    katcp_port = KATCP_PORT_BASE + index
    prometheus_port = PROMETHEUS_PORT_BASE + index
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
        f"--name={name} --cap-add=SYS_NICE --runtime=nvidia --gpus=device={gpu} --net=host --stop-timeout=2 "
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
    for i in range(args.narrowband):
        narrowband_kwargs = {
            "name": f"narrowband{i}",
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


class FgpuBenchmark(Benchmark):
    def __init__(self, args: argparse.Namespace) -> None:
        servers = servers_from_toml(args.servers)
        super().__init__(
            args,
            generator_server=servers[args.dsim_server],
            consumer_server=servers[args.fgpu_server],
            expected_heaps_scale=N_POLS / args.dig_heap_samples,
            metric_prefix="fgpu",
        )

    async def run_producers(
        self,
        adc_sample_rate: float,
        sync_time: int,
    ) -> AsyncExitStack:
        """Run all the dsims.

        The result must be used as a context manager. Exiting the context manager
        will shut down the tasks.
        """
        single_pol = False
        n = self.args.n
        if n <= 2:
            n *= 2
            single_pol = True
        factory = functools.partial(
            dsim_factory,
            adc_sample_rate=adc_sample_rate,
            single_pol=single_pol,
            sync_time=sync_time,
            args=self.args,
        )
        return await run_tasks(
            self.generator_server,
            n,
            factory,
            self.args.image,
            port_base=KATCP_PORT_BASE,
            verbose=self.args.verbose,
            timeout=self.args.init_time,
        )

    async def run_consumers(
        self,
        adc_sample_rate: float,
        sync_time: int,
    ) -> AsyncExitStack:
        """Run all the fgpu instances.

        The result must be used as a context manager. Exiting the context manager
        will shut down the tasks.
        """
        factory = functools.partial(
            fgpu_factory,
            adc_sample_rate=adc_sample_rate,
            sync_time=sync_time,
            args=self.args,
        )
        return await run_tasks(
            self.consumer_server,
            self.args.n,
            factory,
            self.args.image,
            port_base=KATCP_PORT_BASE,
            verbose=self.args.verbose,
            timeout=self.args.init_time,
        )


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
    # For backwards compatibility, specifying --narrowband (without an argument) is
    # equivalent to specifying --narrowband=1.
    parser.add_argument(
        "--narrowband", type=int, default=0, const=1, nargs="?", help="Number of narrowband outputs [0]"
    )
    parser.add_argument(
        "--narrowband-decimation", type=int, default=8, help="Narrowband decimation factor [%(default)s]"
    )
    parser.add_argument("--narrowband-channels", type=int, default=32768, help="Narrowband channels [%(default)s]")
    parser.add_argument("--xb", type=int, default=64, help="Number of XB-engines [%(default)s]")
    parser.add_argument("--fgpu-docker-arg", action="append", default=[], help="Add Docker argument for invoking fgpu")

    parser.add_argument(
        "--init-time", type=float, default=20.0, metavar="SECONDS", help="Time for engines to start [%(default)s]"
    )
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

    benchmark = FgpuBenchmark(args)

    if args.calibrate:
        await benchmark.calibrate(args.low, args.high, args.step, args.calibrate_repeat)
    elif args.oneshot is not None:
        result = await benchmark.measure(args.oneshot)
        print(result.message())
    else:
        slope = {
            1: -342.212919,
            2: -173.264274,
            4: -582.668296,
        }[min(args.n, 4)]
        low, high = await benchmark.search(
            low=args.low,
            high=args.high,
            step=args.step,
            interval=args.interval,
            max_comparisons=args.max_comparisons,
            slope=slope,
        )
        print(f"\n{low / 1e6} MHz - {high / 1e6} MHz")


if __name__ == "__main__":
    asyncio.run(main())
