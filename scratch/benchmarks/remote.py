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

"""Launch and manage remote tasks over SSH."""

import asyncio
import tomllib
from collections.abc import Callable
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Self

import asyncssh

VERBOSE_PASS_OUTPUT = 2  #: Verbosity level at which process output is passed through


@dataclass
class Server:
    """Static configuration of a server."""

    hostname: str
    username: str
    interfaces: list[str] = field(default_factory=list)
    gpus: list[str] = field(default_factory=lambda: ["0"])
    cpus: list[list[int]] = field(default_factory=list)


def servers_from_toml(filename: str) -> dict[str, "Server"]:
    """Parse a set of servers from a configuration file.

    The file must contain one TOML table per server. The table key is a short
    name for the server (which forms the dictionary key in the return value)
    and the table contents must correspond to the :class:`Server` class.

    There is currently no schema validation; if your config file is invalid,
    it might silently work, or it might raise a cryptic error.
    """
    with open(filename, "rb") as f:
        toml = tomllib.load(f)
    servers = {}
    for key, value in toml.items():
        servers[key] = Server(**value)
    return servers


class InsufficientCoresError(Exception):
    pass


@dataclass
class ServerInfo:
    """Dynamic information discovered about a server."""

    #: CPU cores grouped by L3 Cache
    cpus: list[list[int]]
    infiniband_devices: list[str]

    @classmethod
    async def factory(cls, conn: asyncssh.SSHClientConnection, override_cpus: list[list[int]] | None) -> Self:
        if not override_cpus:
            n_l3 = int((await conn.run("hwloc-calc -N L3Cache all")).stdout)  # type: ignore[arg-type]
            cpus = []
            for i in range(n_l3):
                l3_cpus = (await conn.run(f"hwloc-calc --physical-output -I PU L3Cache:{i}")).stdout
                assert isinstance(l3_cpus, str)
                cpus.append([int(cpu) for cpu in l3_cpus.strip().split(",")])
        else:
            cpus = override_cpus
        infiniband_devices_stdout = (await conn.run("find /dev/infiniband -type c -print0")).stdout
        assert isinstance(infiniband_devices_stdout, str)
        infiniband_devices = infiniband_devices_stdout.split("\0")
        if infiniband_devices and infiniband_devices[-1] == "":
            # split will also split on the \0 after the last entry, leaving an empty entry
            infiniband_devices.pop()
        return cls(cpus=cpus, infiniband_devices=infiniband_devices)

    def _allocate_cores(
        self, tasks: int, cores_per_task: int, l3_step: int, share: bool, split: bool
    ) -> list[list[int]]:
        """Make one attempt at :meth:`allocate_cores`.

        Only one in every `l3_step` L3 caches is used. L3 caches can be shared
        between tasks if `share` is true. If `split` is false, a task will not
        share with a previous task if that would cause it to split across L3 caches.
        """
        cpus = self.cpus[::l3_step]
        out = []
        buf: list[int] = []  # Buffer of available cores from an L3 cache
        for _ in range(tasks):
            task: list[int] = []
            while len(task) < cores_per_task:
                need = cores_per_task - len(task)
                if len(buf) < need:
                    if not cpus:
                        raise InsufficientCoresError(
                            f"could not allocate {tasks} tasks with {cores_per_task} cores each"
                        )
                    if not buf or (not split and len(cpus[0]) >= need):
                        buf = list(cpus[0])  # Copy it so we can safely delete from it
                        del cpus[0]
                # Note: Python allows need to be past the end of buf
                task += buf[:need]
                if share:
                    del buf[:need]
                else:
                    buf = []
            out.append(task)
        return out

    def allocate_cores(self, tasks: int, cores_per_task: int) -> list[list[int]]:
        """Assign `cores_per_task` to each of `tasks` tasks.

        This will:

        - Keep all the cores for a task in the same L3 cache, if possible.
        - Avoid sharing L3 caches between tasks, if possible.
        - Try to spread load across the range of L3 caches, rather than just
          using the low-numbered ones.

        These goals are only fully met when the topology is homogeneous i.e.,
        each L3 cache has the same number of cores.

        Raises
        ------
        InsufficientCoresError
            if the first goal cannot be met, or there are simply not enough cores.
        """
        # Try to avoid sharing L3 caches between tasks, and spread the caches out
        # as far as possible.
        for l3_step in range(len(self.cpus), 0, -1):
            try:
                return self._allocate_cores(tasks, cores_per_task, l3_step, False, False)
            except InsufficientCoresError:
                pass
        # If that didn't work, allow sharing, but try to avoid splitting tasks
        # across caches unnecessarily.
        try:
            return self._allocate_cores(tasks, cores_per_task, 1, False, True)
        except InsufficientCoresError:
            pass
        # Last chance: pack things as tightly as possible
        return self._allocate_cores(tasks, cores_per_task, 1, True, True)


async def kill_process(process: asyncssh.SSHClientProcess) -> None:
    """Kill a remote process and wait for it to die."""
    try:
        process.terminate()
        await process.wait(check=False, timeout=30)
    except TimeoutError:
        print("WARNING: kill timed out")
    except OSError as exc:
        print(f"WARNING: kill failed: {exc}")


async def wait_port(server: Server, port: int) -> None:
    """Wait until a particular port on a server is open."""
    while True:
        try:
            reader, writer = await asyncio.open_connection(server.hostname, port)
        except ConnectionRefusedError:
            await asyncio.sleep(1)
        else:
            writer.close()
            await writer.wait_closed()
            return


async def run_tasks(
    server: Server,
    n: int,
    factory: Callable[[Server, ServerInfo, asyncssh.SSHClientConnection, int], str],
    image: str,
    port_base: int | None,
    *,
    verbose: int,
    timeout: float = 20.0,
    pull: bool = True,
) -> AsyncExitStack:
    """Run a set of `n` tasks remotely on a server.

    This will not return until the tasks signal readiness by opening TCP
    ports. The return value is a context manager; closing it (with
    :meth:`contextlib.AsyncExitStack.aclose`) will terminate the processes.

    Parameters
    ----------
    server
        Place where tasks will be run.
    n
        Number of tasks to run.
    factory
        Callback to generate the command to run for each task. It takes the
        following parameters

        - server
        - discovered information about the server
        - asyncssh connection
        - index of the task (from 0 to `n` - 1)
    image
        Docker image to pull before running
    port_base
        Port number for the first task. The tasks must use consecutive ports. If
        None, the server is assumed to be immediately ready.
    verbose
        If at least :const:`VERBOSE_PASS_OUTPUT`, pass through the stdout and
        stderr of the tasks
    timeout
        Time to wait for the ports to be open after starting the tasks
    pull
        Whether to pull the docker image
    """
    async with AsyncExitStack() as stack:
        conn_options = asyncssh.SSHClientConnectionOptions(keepalive_interval="15s")
        conn = await stack.enter_async_context(
            asyncssh.connect(server.hostname, username=server.username, options=conn_options)
        )
        server_info = await ServerInfo.factory(conn=conn, override_cpus=server.cpus)
        if pull:
            await conn.run(f"docker pull {image}", check=True)
        procs: list[asyncssh.SSHClientProcess] = []
        for i in range(n):
            command = factory(server, server_info, conn, i)
            procs.append(
                await conn.create_process(
                    command=command,
                    stdin=asyncssh.DEVNULL,
                    stdout=asyncssh.DEVNULL if verbose < VERBOSE_PASS_OUTPUT else "/dev/stdout",
                    stderr=asyncssh.DEVNULL if verbose < VERBOSE_PASS_OUTPUT else "/dev/stderr",
                )
            )
            await stack.enter_async_context(procs[-1])
            stack.push_async_callback(kill_process, procs[-1])

        # Wait for the service to be ready by checking the katcp ports
        async with asyncio.timeout(timeout):
            for i in range(n):
                if port_base is not None:
                    # Wait until either the port is ready or the process dies
                    async with asyncio.TaskGroup() as tg:
                        tasks: list[asyncio.Future] = [
                            tg.create_task(wait_port(server, port_base + i)),
                            tg.create_task(procs[i].wait(check=True)),
                        ]
                        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                        for task in pending:
                            task.cancel()
                    if tasks[1] in done:
                        raise RuntimeError("process shut down before becoming ready")
        return stack.pop_all()
    raise AssertionError("should be unreachable")
