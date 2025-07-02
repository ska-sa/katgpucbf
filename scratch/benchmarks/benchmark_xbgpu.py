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

"""Benchmark critical rate of xbgpu.

See :doc:`benchmarking`.
"""

import argparse
import asyncio
import functools
import math
from contextlib import AsyncExitStack

import asyncssh

from katgpucbf import COMPLEX, DEFAULT_JONES_PER_BATCH, N_POLS

from benchmark_tools import DEFAULT_IMAGE, PROMETHEUS_PORT_BASE, Benchmark
from remote import Server, ServerInfo, run_tasks, servers_from_toml

SAMPLE_BITS = 8
KATCP_PORT_BASE = 7340


class StreamInfo:
    """Parameters describing the stream."""

    def __init__(self, args: argparse.Namespace, adc_sample_rate: float) -> None:
        self.channels_per_substream = args.channels // args.substreams
        self.bandwidth = adc_sample_rate / 2 / args.narrowband_decimation
        self.samples_between_spectra = 2 * args.channels * args.narrowband_decimation
        self.spectra_per_heap = args.jones_per_batch // args.channels


def fsim_factory(
    server: Server,
    server_info: ServerInfo,
    conn: asyncssh.SSHClientConnection,
    index: int,
    *,
    adc_sample_rate: float,
    sync_time: int,
    args: argparse.Namespace,
) -> str:
    """Generate command to run fsim."""
    ncpus = len(server_info.cpus)
    step = ncpus // args.n
    my_cpus = server_info.cpus[index * step : (index + 1) * step]
    interface = server.interfaces[index % len(server.interfaces)]
    prometheus_port = PROMETHEUS_PORT_BASE + index
    name = f"fsim-{index}"
    info = StreamInfo(args, adc_sample_rate)
    command = (
        "docker run --init "  # --init is needed because fsim doesn't catch SIGTERM itself
        f"--name={name} --cap-add=SYS_NICE --net=host "
        + "".join(f"--device={dev}:{dev} " for dev in server_info.infiniband_devices)
        + f"--ulimit=memlock=-1 --rm {args.image} "
        f"schedrr taskset -c {my_cpus[1]} fsim "
        "--ibv "
        f"--interface={interface} "
        f"--sync-time={sync_time} "
        f"--affinity={my_cpus[0]} "
        f"--main-affinity={my_cpus[1]} "
        f"--prometheus-port={prometheus_port} "
        f"--adc-sample-rate={adc_sample_rate} "
        f"--array-size={args.array_size} "
        f"--channels={args.channels} "
        f"--channels-per-substream={info.channels_per_substream} "
        f"--samples-between-spectra={info.samples_between_spectra} "
        f"--jones-per-batch={args.jones_per_batch} "
        f"239.102.199.{index}:7148 "
    )
    return command


def xbgpu_factory(
    server: Server,
    server_info: ServerInfo,
    conn: asyncssh.SSHClientConnection,
    index: int,
    *,
    adc_sample_rate: float,
    sync_time: int,
    args: argparse.Namespace,
) -> str:
    """Generate command to run xbgpu."""
    ncpus = len(server_info.cpus)
    step = ncpus // args.n
    # TODO: For non-power-of-two args.n, we should rather align to L3 caches
    my_cpus = server_info.cpus[index * step : (index + 1) * step]
    interface = server.interfaces[index % len(server.interfaces)]
    gpu = server.gpus[index % len(server.gpus)]
    katcp_port = KATCP_PORT_BASE + index
    prometheus_port = PROMETHEUS_PORT_BASE + index
    name = f"xbgpu-{index}"

    info = StreamInfo(args, adc_sample_rate)
    heap_time = info.samples_between_spectra * info.spectra_per_heap / adc_sample_rate
    threshold = max(1, round(args.int_time / heap_time))
    # Duplicate logic from katsdpcontroller's generator.py
    batch_size = (
        args.array_size * N_POLS * info.spectra_per_heap * info.channels_per_substream * COMPLEX * SAMPLE_BITS // 8
    )
    target_chunk_size = 64 * 1024**2
    batches_per_chunk = math.ceil(max(128 / info.spectra_per_heap, target_chunk_size / batch_size))

    command = (
        "docker run "
        "--stop-timeout=2 "
        f"--name={name} --cap-add=SYS_NICE --runtime=nvidia --gpus=device={gpu} --net=host "
        f"-e NVIDIA_MOFED=enabled --ulimit=memlock=-1 --rm {args.image} "
        f"schedrr taskset -c {my_cpus[1]} xbgpu "
        f"--katcp-port={katcp_port} "
        f"--prometheus-port={prometheus_port} "
        f"--adc-sample-rate={adc_sample_rate} "
        f"--bandwidth={info.bandwidth} "
        f"--array-size={args.array_size} "
        f"--channels={args.channels} "
        f"--channels-per-substream={info.channels_per_substream} "
        f"--samples-between-spectra={info.samples_between_spectra} "
        f"--jones-per-batch={args.jones_per_batch} "
        f"--sample-bits={SAMPLE_BITS} "
        f"--heaps-per-fengine-per-chunk={batches_per_chunk} "
        f"--sync-time={sync_time} "
        f"--recv-affinity={my_cpus[0]} "
        f"--recv-comp-vector={my_cpus[0]} "
        f"--recv-interface={interface} "
        f"--recv-ibv "
        f"--send-affinity={my_cpus[1]} "
        f"--send-comp-vector={my_cpus[1]} "
        f"--send-interface={interface} "
        f"--send-ibv "
        f"--send-enabled "
        f"239.102.199.{index}:7148 "
        f"--corrprod=name=corrprod,dst=239.102.198.{index},heap_accumulation_threshold={threshold} "
    )
    for i in range(args.beams):
        for j in range(N_POLS):
            idx = N_POLS * i + j
            command += f"--beam=name=beam{idx},dst=239.102.197.{index * args.beams * N_POLS + idx},pol={j} "
    return command


class XbgpuBenchmark(Benchmark):
    def __init__(self, args: argparse.Namespace) -> None:
        servers = servers_from_toml(args.servers)
        samples_between_spectra = 2 * args.channels * args.narrowband_decimation
        spectra_per_heap = args.jones_per_batch // args.channels
        heap_samples = samples_between_spectra * spectra_per_heap
        super().__init__(
            args,
            generator_server=servers[args.fsim_server],
            consumer_server=servers[args.xbgpu_server],
            expected_heaps_scale=args.array_size / heap_samples,
            metric_prefix="xbgpu",
        )

    async def run_producers(self, adc_sample_rate: float, sync_time: int) -> AsyncExitStack:
        factory = functools.partial(
            fsim_factory,
            adc_sample_rate=adc_sample_rate,
            sync_time=sync_time,
            args=self.args,
        )
        return await run_tasks(
            self.generator_server,
            self.args.n,
            factory,
            self.args.image,
            port_base=None,
            verbose=self.args.verbose,
            timeout=self.args.init_time,
        )

    async def run_consumers(self, adc_sample_rate: float, sync_time: int) -> AsyncExitStack:
        factory = functools.partial(
            xbgpu_factory,
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
    parser.add_argument("-n", type=int, required=True, help="Number of engines per host [%(default)s]")
    parser.add_argument("--channels", type=int, default=1024, help="Channel count [%(default)s]")
    parser.add_argument("--array-size", type=int, default=80, help="Number of antennas [%(default)s]")
    parser.add_argument("--substreams", type=int, default=64, help="Total number of engines [%(default)s]")
    parser.add_argument("--int-time", type=float, default=0.5, metavar="SECONDS", help="Integration time [%(default)s]")
    parser.add_argument("--narrowband", action="store_true", help="Measure narrowband output [false]")
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
    parser.add_argument("--beams", type=int, default=4, help="Number of dual-pol beams to produce [%(default)s]")
    parser.add_argument(
        "--jones-per-batch",
        type=int,
        default=DEFAULT_JONES_PER_BATCH,
        metavar="SAMPLES",
        help="Jones vectors in each output batch [%(default)s]",
    )
    parser.add_argument("--image", type=str, default=DEFAULT_IMAGE, help="Docker image [%(default)s]")
    parser.add_argument("--servers", type=str, default="servers.toml", help="Server description file [%(default)s]")
    parser.add_argument("--fsim-server", type=str, default="fsim", help="Server on which to run fsims [%(default)s]")
    parser.add_argument("--xbgpu-server", type=str, default="xbgpu", help="Server on which to run xbgpu [%(default)s]")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity [no]")
    parser.add_argument("--oneshot", type=float, help="Run one test at the given sampling rate")
    args = parser.parse_args()

    if not args.narrowband:
        args.narrowband_decimation = 1  # Simplifies later logic
    if args.channels % args.substreams:
        parser.error("--substreams must divide evenly into --channels")
    if not args.oneshot:
        parser.error("Only --oneshot mode is implemented so far")

    benchmark = XbgpuBenchmark(args)
    result = await benchmark.measure(args.oneshot)
    print(result.message())


if __name__ == "__main__":
    asyncio.run(main())
