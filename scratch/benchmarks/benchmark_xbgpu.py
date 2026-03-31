#!/usr/bin/env python3

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

"""Benchmark critical rate of xbgpu.

See :doc:`benchmarking`.
"""

import argparse
import asyncio
import functools
import ipaddress
import math
from contextlib import AsyncExitStack
from typing import override

import asyncssh

from katgpucbf import COMPLEX, N_POLS

from benchmark_tools import (
    PROMETHEUS_PORT_BASE,
    Benchmark,
    _address_at_index,
    _split_network,
    add_common_benchmark_arguments,
    process_common_benchmark_arguments,
)
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
    multicast_group: ipaddress.IPv4Network | ipaddress.IPv6Network,
) -> str:
    """Generate command to run fsim."""
    cores = server_info.allocate_cores(args.n, 2)[index]
    interface = server.interfaces[index % len(server.interfaces)]
    prometheus_port = PROMETHEUS_PORT_BASE + index
    name = f"fsim-{index}"
    info = StreamInfo(args, adc_sample_rate)
    command = (
        "docker run --init "  # --init is needed because fsim doesn't catch SIGTERM itself
        f"--name={name} --cap-add=SYS_NICE --net=host --stop-timeout=2 "
        + "".join(f"--device={dev}:{dev} " for dev in server_info.infiniband_devices)
        + f"--ulimit=memlock=-1 --rm {args.image} "
        f"schedrr taskset -c {cores[1]} fsim "
        "--ibv "
        f"--interface={interface} "
        f"--sync-time={sync_time} "
        f"--affinity={cores[0]} "
        f"--main-affinity={cores[1]} "
        f"--prometheus-port={prometheus_port} "
        f"--adc-sample-rate={adc_sample_rate} "
        f"--array-size={args.array_size} "
        f"--channels={args.channels} "
        f"--channels-per-substream={info.channels_per_substream} "
        f"--samples-between-spectra={info.samples_between_spectra} "
        f"--jones-per-batch={args.jones_per_batch} "
        f"{str(_address_at_index(multicast_group, index))}:7148 "
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
    multicast_group: ipaddress.IPv4Network | ipaddress.IPv6Network,
    fsim_multicast_group: ipaddress.IPv4Network | ipaddress.IPv6Network,
) -> str:
    """Generate command to run xbgpu."""
    cores = server_info.allocate_cores(args.n, 2)[index]
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
    beam_multicast_group, corrprod_multicast_group = _split_network(multicast_group)

    command = (
        "docker run "
        "--stop-timeout=2 "
        f"--name={name} --cap-add=SYS_NICE --runtime=nvidia --gpus=device={gpu} --net=host "
        f"-e NVIDIA_MOFED=enabled --ulimit=memlock=-1 --rm {args.image} "
        f"schedrr taskset -c {cores[1]} xbgpu "
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
        f"--recv-affinity={cores[0]} "
        f"--recv-comp-vector={cores[0]} "
        f"--recv-interface={interface} "
        f"--recv-ibv "
        f"--send-affinity={cores[1]} "
        f"--send-comp-vector={cores[1]} "
        f"--send-interface={interface} "
        f"--send-ibv "
        f"--send-enabled "
        f"{str(_address_at_index(fsim_multicast_group, index))}:7148 "
    )
    for i in range(args.beams):
        for j in range(N_POLS):
            idx = N_POLS * i + j
            beam_number = index * args.beams * N_POLS + idx
            command += f"--beam=name=beam{idx},dst={str(_address_at_index(beam_multicast_group, beam_number))},pol={j} "

    for i in range(args.corrprods):
        corrprod_number = index * args.corrprods + i
        command += f"--corrprod=name=corrprod{corrprod_number},"
        command += f"dst={str(_address_at_index(corrprod_multicast_group, corrprod_number))},"
        command += f"heap_accumulation_threshold={threshold} "
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
            slope={
                1: -451.368500,
                2: -429.814719,
            },
        )

    @override
    async def run_producers(self, adc_sample_rate: float, sync_time: int) -> AsyncExitStack:
        factory = functools.partial(
            fsim_factory,
            multicast_group=self.producer_multicast_group,
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
            pull=self.args.pull,
        )

    @override
    async def run_consumers(self, adc_sample_rate: float, sync_time: int) -> AsyncExitStack:
        factory = functools.partial(
            xbgpu_factory,
            multicast_group=self.consumer_multicast_group,
            fsim_multicast_group=self.producer_multicast_group,
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
            pull=self.args.pull,
        )


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--substreams", type=int, default=64, help="Total number of engines [%(default)s]")
    parser.add_argument("--int-time", type=float, default=0.5, metavar="SECONDS", help="Integration time [%(default)s]")
    parser.add_argument("--beams", type=int, default=4, help="Number of dual-pol beams to produce [%(default)s]")
    parser.add_argument(
        "--corrprods", type=int, default=1, help="Number of correlation products to produce [%(default)s]"
    )
    parser.add_argument("--fsim-server", type=str, default="fsim", help="Server on which to run fsims [%(default)s]")
    parser.add_argument("--xbgpu-server", type=str, default="xbgpu", help="Server on which to run xbgpu [%(default)s]")
    add_common_benchmark_arguments(parser)
    args = parser.parse_args()

    if args.channels % args.substreams:
        parser.error("--substreams must divide evenly into --channels")
    if args.jones_per_batch % args.channels or args.jones_per_batch // args.channels % 16:
        parser.error("spectra per heap (--jones_per_batch // --channels) must divide evenly and be a multiple of 16")
    if args.beams * N_POLS > 255:
        parser.error("total number of output beams must be less than 255 to fit range 239.102.197.0/24")
    if args.corrprods > 255:
        parser.error("total number of correlation products must be less than 255 to fit range 239.102.198.0/24")
    process_common_benchmark_arguments(args, parser)

    await XbgpuBenchmark(args).run()


if __name__ == "__main__":
    asyncio.run(main())
