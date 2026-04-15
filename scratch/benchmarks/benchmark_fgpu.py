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

"""Benchmark critical rate of fgpu.

See :doc:`benchmarking`.
"""

import argparse
import asyncio
import functools
import ipaddress
from collections.abc import Iterator
from contextlib import AsyncExitStack
from itertools import chain
from typing import override

from katgpucbf import N_POLS

from benchmark_tools import (
    PROMETHEUS_PORT_BASE,
    Benchmark,
    add_common_benchmark_arguments,
    compress,
    process_common_benchmark_arguments,
)
from remote import InsufficientCoresError, Server, ServerInfo, run_tasks, servers_from_toml

KATCP_PORT_BASE = 7140


class FgpuBenchmark(Benchmark):
    def __init__(self, args: argparse.Namespace) -> None:
        servers = servers_from_toml(args.servers)
        self.dsim_addresses: Iterator[list[ipaddress.IPv4Address | ipaddress.IPv6Address]] = iter([])
        super().__init__(
            args,
            generator_server=servers[args.dsim_server],
            consumer_server=servers[args.fgpu_server],
            expected_heaps_scale=N_POLS / args.dig_heap_samples,
            metric_prefix="fgpu",
            slope={
                1: -342.212919,
                2: -173.264274,
                4: -582.668296,
            },
        )

    def dsim_factory(
        self,
        server: Server,
        server_info: ServerInfo,
        index: int,
        *,
        adc_sample_rate: float,
        sync_time: int,
        single_pol: bool,
    ) -> str:
        """Generate command to run dsim.

        The multicast address(es) used for the dsims are appended to `self.dsim_addresses`.
        """
        # Use as many CPUs as we can to speed up startup. We need at least 3
        # (main thread, network thread and worker thread).
        n = self.args.n
        if single_pol:
            n *= 2

        cores_per_task = 3
        while True:
            try:
                server_info.allocate_cores(n, cores_per_task + 1)
            except InsufficientCoresError:
                break
            else:
                cores_per_task += 1
        cores = server_info.allocate_cores(n, cores_per_task)[index]
        if self.args.n == 1 or not single_pol:
            interface = server.interfaces[index % len(server.interfaces)]
        else:
            # For larger n, send the two pols over the same interface
            # (because fgpu_factory expects them to arrive on the same interface)
            interface = server.interfaces[index // 2 % len(server.interfaces)]
        katcp_port = KATCP_PORT_BASE + index
        prometheus_port = PROMETHEUS_PORT_BASE + index
        name = f"feng-dsim-{index}"
        if single_pol:
            addresses = [self.multicast_allocator.as_list(8)]
        else:
            addresses = [self.multicast_allocator.as_list(8), self.multicast_allocator.as_list(8)]
        self.dsim_addresses = chain(self.dsim_addresses, iter(addresses))
        addresses_str = " ".join(f"{compress(addrs)}:7148" for addrs in addresses)
        command = (
            "docker run "
            f"--name={name} --cap-add=SYS_NICE --net=host --stop-timeout=2 "
            + "".join(f"--device={dev}:{dev} " for dev in server_info.infiniband_devices)
            + f"--ulimit=memlock=-1 --rm {self.args.image} "
            f"taskset -c {','.join(str(core) for core in cores[2:])} "
            f"dsim --affinity={cores[0]} "
            f"--main-affinity={cores[1]} "
            "--ibv "
            f"--interface={interface} "
            f"--adc-sample-rate={adc_sample_rate} "
            f"--heap-samples={self.args.dig_heap_samples} "
            "--ttl=2 "
            "--period=16777216 "  # Speeds things up
            f"--katcp-port={katcp_port} "
            f"--prometheus-port={prometheus_port} "
            f"--sync-time={sync_time} "
            f"--first-id={index if single_pol else 2 * index} "
            f"{addresses_str} "
        )
        if self.args.dig_sample_bits is not None:
            command += f"--sample-bits={self.args.dig_sample_bits} "
        return command

    def fgpu_factory(
        self,
        server: Server,
        server_info: ServerInfo,
        index: int,
        *,
        adc_sample_rate: float,
        sync_time: int,
    ) -> str:
        """Generate command to run fgpu."""
        n = self.args.n
        # When we run > 4, we assume we have enough RAM (GPU and host) that we
        # don't need to scale buffers down to tiny amounts.
        scaling_n = min(n, 4)
        if n == 3:
            # Currently the chunk_jones must be a power of 2, this ensures that it is for n=3 and that it fits in memory
            scaling_n = 4

        recv_chunk_samples = 2**27 // scaling_n
        send_chunk_jones = recv_chunk_samples // 4
        if n == 1:
            interface = ",".join(server.interfaces[:2])
            recv_cores = 4
        else:
            interface = server.interfaces[index % len(server.interfaces)]
            recv_cores = 2

        narrowband_addresses_per_fgpu_dst = self.args.xb // self.args.narrowband_decimation
        cores = server_info.allocate_cores(n, recv_cores + 2)[index]
        recv_affinity = ",".join(str(core) for core in cores[:-2])
        send_affinity = str(cores[-2])
        other_affinity = str(cores[-1])
        gpu = server.gpus[index % len(server.gpus)]
        katcp_port = KATCP_PORT_BASE + index
        prometheus_port = PROMETHEUS_PORT_BASE + index
        name = f"fgpu-{index}"
        wideband_kwargs = {
            "name": "wideband",
            "channels": self.args.channels,
            "dst": f"{self.multicast_allocator(self.args.xb)}:7148",
        }

        if self.args.jones_per_batch is not None:
            wideband_kwargs["jones_per_batch"] = self.args.jones_per_batch
        wideband_arg = ",".join(f"{key}={value}" for key, value in wideband_kwargs.items())
        # Grab two polarisations of dsim addresses
        dsim = compress(next(self.dsim_addresses) + next(self.dsim_addresses))
        command = (
            "docker run "
            f"--name={name} --cap-add=SYS_NICE --runtime=nvidia --gpus=device={gpu} --net=host --stop-timeout=2 "
            f"-e NVIDIA_MOFED=enabled --ulimit=memlock=-1 --rm "
            f" {' '.join(self.args.fgpu_docker_arg)} {self.args.image} "
            f"schedrr taskset -c {other_affinity} fgpu "
            f"--recv-packet-samples={self.args.dig_heap_samples} "
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
            f"{'--use-vkgdr' if self.args.use_vkgdr else ''} "
            f"--wideband={wideband_arg} "
            f"{dsim}:7148 "
        )
        for i in range(self.args.narrowband):
            narrowband_kwargs = {
                "name": f"narrowband{i}",
                "channels": self.args.narrowband_channels,
                "decimation": self.args.narrowband_decimation,
                "centre_frequency": adc_sample_rate / 4,
                "dst": self.multicast_allocator(narrowband_addresses_per_fgpu_dst) + ":7148",
            }
            if self.args.jones_per_batch is not None:
                narrowband_kwargs["jones_per_batch"] = self.args.jones_per_batch
            narrowband_arg = ",".join(f"{key}={value}" for key, value in narrowband_kwargs.items())
            command += f"--narrowband={narrowband_arg} "
        for arg in ["array_size", "dig_sample_bits"]:
            value = getattr(self.args, arg)
            if value is not None:
                dashed = arg.replace("_", "-")
                command += f"--{dashed}={value} "
        for extra in self.args.extra:
            command += f"{extra} "
        return command

    @override
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
            self.dsim_factory,
            adc_sample_rate=adc_sample_rate,
            sync_time=sync_time,
            single_pol=single_pol,
        )
        return await run_tasks(
            self.generator_server,
            n,
            factory,
            self.args.image,
            port_base=KATCP_PORT_BASE,
            verbose=self.args.verbose,
            timeout=self.args.init_time,
            pull=self.args.pull,
        )

    @override
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
            self.fgpu_factory,
            adc_sample_rate=adc_sample_rate,
            sync_time=sync_time,
        )
        return await run_tasks(
            self.consumer_server,
            self.args.n,
            factory,
            image=self.args.image,
            port_base=KATCP_PORT_BASE,
            verbose=self.args.verbose,
            timeout=self.args.init_time,
            pull=self.args.pull,
        )


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use-vkgdr",
        action="store_true",
        help="Assemble chunks directly in GPU memory (requires sufficient GPU BAR space)",
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
    parser.add_argument("--narrowband-channels", type=int, default=32768, help="Narrowband channels [%(default)s]")
    parser.add_argument("--xb", type=int, default=64, help="Number of XB-engines [%(default)s]")
    parser.add_argument("--fgpu-docker-arg", action="append", default=[], help="Add Docker argument for invoking fgpu")
    parser.add_argument("--dsim-server", type=str, default="dsim", help="Server on which to run dsims [%(default)s]")
    parser.add_argument("--fgpu-server", type=str, default="fgpu", help="Server on which to run fgpu [%(default)s]")
    add_common_benchmark_arguments(parser)
    parser.add_argument("extra", nargs="*", help="Remaining arguments are passed to fgpu")
    args = parser.parse_args()

    process_common_benchmark_arguments(args, parser)

    await FgpuBenchmark(args).run()


if __name__ == "__main__":
    asyncio.run(main())
