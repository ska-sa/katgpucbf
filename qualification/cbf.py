################################################################################
# Copyright (c) 2022-2025, National Research Foundation (SARAO)
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

"""Information about and control over CBF subarray products."""

import asyncio
import json
import logging
import re
import traceback
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict
from uuid import UUID, uuid4

import aiokatcp
import pytest

from katgpucbf import DIG_SAMPLE_BITS

from .host_config import HostConfigQuerier
from .reporter import Reporter, custom_report_log

if TYPE_CHECKING:
    # This is only imported for type checkers, because importing at runtime
    # would create a cyclic dependency.
    from .recv import BaselineCorrelationProductsReceiver, TiedArrayChannelisedVoltageReceiver

logger = logging.getLogger(__name__)
DEFAULT_MAX_DELAY = 1000000  # Around 0.5-1ms, depending on band. Increase if necessary


@dataclass
class CBFBase:
    """Information about a potential CBF that does not require it to exist yet."""

    name: str
    config: dict  # JSON dictionary used to configure the CBF
    mode_config: dict  # Configuration values used for MeerKAT mode string
    uuid: UUID

    async def close(self, master_controller_client: aiokatcp.Client | None) -> None:
        """Shut down all the connections and deconfigure the subarray product."""
        pass  # Derived classes can override


@dataclass
class FailedCBF(CBFBase):
    """A CBF that we attempted to create, but it failed."""

    exc: Exception


@dataclass
class CBFRemoteControl(CBFBase):
    """A container class for katcp clients needed by qualification tests."""

    product_controller_endpoint: tuple[str, int]
    product_controller_client: aiokatcp.Client
    dsim_names: list[str]  # Names to pass to ?dsim-signals requests
    sensor_watcher: aiokatcp.SensorWatcher
    # These are filled in by conftest.py.
    baseline_correlation_products_receiver: "BaselineCorrelationProductsReceiver | None" = None
    tied_array_channelised_voltage_receiver: "TiedArrayChannelisedVoltageReceiver | None" = None

    @property
    def init_sensors(self) -> aiokatcp.SensorSet:
        """Initial sensor values from the product controller.

        This is initialised once after starting the subarray product and not
        subsequently updated. Dynamic sensors should be queried directly.
        """
        return self.sensor_watcher.sensors

    @classmethod
    async def connect(cls, name: str, host: str, port: int, config: Mapping, mode_config: dict) -> "CBFRemoteControl":
        """Connect to a CBF's product controller.

        The function connects and gathers sufficient metadata in order for the
        user to know how to use the CBF for whatever testing needs to be
        done.
        """
        pcc = aiokatcp.Client(host, port)
        sensor_watcher = aiokatcp.SensorWatcher(pcc)
        pcc.add_sensor_watcher(sensor_watcher)
        await sensor_watcher.synced.wait()  # Implicitly waits for connection too
        # Remove it again: we only want the initial sensor values, not the full
        # firehose.
        pcc.remove_sensor_watcher(sensor_watcher)

        dsim_names = []
        for sensor_name in sensor_watcher.sensors:
            if match := re.fullmatch(r"(sim\.dsim\d+\.\d+\.0)\.port", sensor_name):
                dsim_names.append(match.group(1))
        dsim_names.sort()  # sorts by index

        logger.info("Sensors synchronised; %d dsims found", len(dsim_names))

        return CBFRemoteControl(
            name=name,
            product_controller_endpoint=(host, port),
            product_controller_client=pcc,
            dsim_names=list(dsim_names),
            config=dict(config),
            mode_config=mode_config,
            sensor_watcher=sensor_watcher,
            uuid=uuid4(),
        )

    async def steady_state_timestamp(self, *, max_delay: int = DEFAULT_MAX_DELAY) -> int:
        """Get a timestamp by which the system will be in a steady state.

        In other words, the effects of previous commands will be in place for
        data with this timestamp.

        Because delays affect timestamps, the caller must provide an upper
        bound on the delay of any F-engine. The default for this should be
        acceptable for most cases.
        """
        timestamp = 0
        _, informs = await self.product_controller_client.request("sensor-value", r"/.*steady-state-timestamp$/")
        for inform in informs:
            # In theory there could be multiple sensors per inform, but aiokatcp
            # never does this because timestamps are seldom shared.
            sensor_name = inform.arguments[2]  # Byte string
            sensor_value = int(inform.arguments[4])
            if sensor_name.startswith(b"sim."):
                # values returned from the dsim do not account for delay,
                # so need to be offset to get an output timestamp.
                sensor_value += max_delay
            timestamp = max(timestamp, sensor_value)
        logger.debug("steady_state_timestamp: %d", timestamp)
        return timestamp

    async def close(self, master_controller_client: aiokatcp.Client | None) -> None:
        """Shut down all the connections and deconfigure the subarray product."""
        if self.baseline_correlation_products_receiver is not None:
            self.baseline_correlation_products_receiver.stream_group.stop()
            self.baseline_correlation_products_receiver = None  # Break reference cycle
        if self.tied_array_channelised_voltage_receiver is not None:
            self.tied_array_channelised_voltage_receiver.stream_group.stop()
            self.tied_array_channelised_voltage_receiver = None  # Break reference cycle
        self.product_controller_client.close()
        await self.product_controller_client.wait_closed()
        if master_controller_client is not None:
            await master_controller_client.request("product-deconfigure", self.name)

    async def dsim_time(self, dsim_idx: int = 0) -> float:
        """Get the current UNIX time, as reported by a dsim.

        This helps make tests independent of the clock on the machine running
        the test; it depends only on the dsims to be synchronised with each other.
        """
        reply, _ = await self.product_controller_client.request("dsim-time", self.dsim_names[dsim_idx])
        return aiokatcp.decode(float, reply[0])

    async def dsim_gaussian(
        self, amplitude: float, pdf_report: Reporter | None = None, *, dsim_idx: int = 0, period: int | None = None
    ) -> None:
        """Configure a dsim with Gaussian noise.

        The identical signal is produced on both polarisations.

        Parameters
        ----------
        amplitude
            Standard deviation, in units of the LSB of the digitiser output
        pdf_report
            Reporter to which this process will be reported
        dsim_idx
            Index of the dsim to set
        period
            If specified, override the period of the dsim signal
        """
        if pdf_report is not None:
            pdf_report.step("Configure the D-sim with Gaussian noise.")
        dig_max = 2 ** (DIG_SAMPLE_BITS - 1) - 1
        amplitude /= dig_max  # Convert to be relative to full-scale
        signal = f"common=nodither(wgn({amplitude}));common;common;"
        if period is None:
            await self.product_controller_client.request("dsim-signals", self.dsim_names[dsim_idx], signal)
            suffix = ""
        else:
            await self.product_controller_client.request("dsim-signals", self.dsim_names[dsim_idx], signal, period)
            suffix = f" and period={period} samples"
        if pdf_report is not None:
            pdf_report.detail(f"Set D-sim with wgn amplitude={amplitude}{suffix}.")


class TaskDict(TypedDict):
    """Type annotation for dictionary describing tasks."""

    host: str
    interfaces: dict[str, str]
    version: str
    git_version: str


async def _get_git_version_conn(conn: aiokatcp.Client) -> str:
    """Query a katcp server for the katcp-device build-info."""
    _, informs = await conn.request("version-list")
    for inform in informs:
        if aiokatcp.decode(str, inform.arguments[0]) == "katcp-device":
            return aiokatcp.decode(str, inform.arguments[2])
    return "unknown"


async def _get_git_version(host: str, port: int) -> str:
    """Query a katcp server for the katcp-device build-info."""
    async with await aiokatcp.Client.connect(host, port) as conn:
        return await _get_git_version_conn(conn)


async def _report_cbf_config(
    pytestconfig: pytest.Config,
    host_config_querier: HostConfigQuerier,
    cbf: CBFRemoteControl,
    master_controller_client: aiokatcp.Client,
) -> None:
    async def get_task_details_multi[T](suffix: str, type: type[T]) -> dict[str, dict[tuple[str, ...], T]]:
        """Get values of a task-specific sensor for all tasks.

        The `suffix` is a regular expression, and may contain anonymous capture
        groups. The return value is a nested dictionary: the outer dictionary is
        indexed by the task name, the inner one by the capture group values.
        """
        regex = rf"^(.*)\.{suffix}$"
        r = re.compile(regex)
        _, informs = await cbf.product_controller_client.request("sensor-value", f"/{regex}/")
        result: dict[str, dict[tuple[str, ...], T]] = {}
        for inform in informs:
            if inform.arguments[3] == b"nominal":
                m = r.match(aiokatcp.decode(str, inform.arguments[2]))
                if m is None:
                    continue  # Should never happen, unless the product controller is buggy
                task = m[1]
                key = m.groups()[1:]  # All capture groups except the task name
                result.setdefault(task, {})[key] = aiokatcp.decode(type, inform.arguments[4])
        return result

    async def get_task_details[T](suffix: str, type: type[T]) -> dict[str, T]:
        """Get value of a task-specific sensor for all tasks."""
        raw = await get_task_details_multi(re.escape(suffix), type)
        return {key: value[()] for key, value in raw.items()}

    ports = await get_task_details("port", aiokatcp.Address)
    git_version_futures = {}
    async with asyncio.TaskGroup() as tg:
        for task_name, address in ports.items():
            assert address.port is not None
            git_version_futures[task_name] = tg.create_task(_get_git_version(str(address.host), address.port))

    versions = await get_task_details("version", str)
    hosts = await get_task_details("host", str)
    interfaces = await get_task_details_multi(r"interfaces\.([^.]+)\.name", str)
    tasks: dict[str, TaskDict] = {}
    for task_name, hostname in hosts.items():
        if task_name not in git_version_futures:
            raise RuntimeError(f"could not get katcp port for {task_name}: possible DNS failure")
        task_interfaces = interfaces.get(task_name, {})
        tasks[task_name] = {
            "host": hostname,
            "interfaces": {key[0]: value for key, value in task_interfaces.items()},  # Flatten 1-tuple key
            "version": versions[task_name],
            "git_version": git_version_futures[task_name].result(),
        }
    tasks["product_controller"] = {
        "host": await master_controller_client.sensor_value(f"{cbf.name}.host", str),
        "interfaces": {},
        "version": await master_controller_client.sensor_value(f"{cbf.name}.version", str),
        "git_version": await _get_git_version_conn(cbf.product_controller_client),
    }

    for task in tasks.values():
        host_config = host_config_querier.get_config(task["host"])
        if host_config is not None:
            logger.info("Logging host config for %s", task["host"])
            custom_report_log(
                pytestconfig, {"$report_type": "HostConfiguration", "hostname": task["host"], "config": host_config}
            )

    custom_report_log(
        pytestconfig,
        {
            "$report_type": "CBFConfiguration",
            "mode_config": cbf.mode_config,
            "uuid": str(cbf.uuid),
            "tasks": tasks,
        },
    )


class CBFCache:
    """Obtain a CBF with the given configuration.

    If there is already one running with the identical configuration, use it.
    Otherwise, shut down any existing one and start a new one.

    Parameters
    ----------
    host, port
        Endpoint for the master controller
    """

    def __init__(self, pytestconfig: pytest.Config) -> None:
        self._cbf: CBFBase | None = None
        self._master_controller_client: aiokatcp.Client | None = None
        self._pytestconfig = pytestconfig
        self._host_config_querier = HostConfigQuerier(pytestconfig.getini("prometheus_url"))

    async def _close_cbf(self) -> None:
        if self._cbf is not None:
            name = self._cbf.name
            logger.info("Tearing down CBF %s.", name)
            await self._cbf.close(self._master_controller_client)
            self._cbf = None

    async def _get_master_controller_client(self) -> aiokatcp.Client:
        if self._master_controller_client is not None:
            return self._master_controller_client

        try:
            host = self._pytestconfig.getini("master_controller_host")
            port = int(self._pytestconfig.getini("master_controller_port"))
            logger.debug("Connecting to master controller at %s:%d.", host, port)
            async with asyncio.timeout(10):
                self._master_controller_client = await aiokatcp.Client.connect(host, port)
            return self._master_controller_client
        except (ConnectionError, TimeoutError):
            logger.exception("unable to connect to master controller!")
            raise

    async def get_cbf(self, cbf_config: dict, cbf_mode_config: dict) -> CBFBase:
        """Get a :class:`CBFBase`, creating it if necessary."""
        if self._cbf is not None and self._cbf.config == cbf_config:
            return self._cbf

        product_name = self._pytestconfig.getini("product_name")
        try:
            await self._close_cbf()
            master_controller_client = await self._get_master_controller_client()
            try:
                reply, _ = await master_controller_client.request(
                    "product-configure", product_name, json.dumps(cbf_config)
                )
            except aiokatcp.FailReply:
                logger.exception("Something went wrong with starting the CBF!")
                raise

            product_controller_host = aiokatcp.decode(str, reply[1])
            product_controller_port = aiokatcp.decode(int, reply[2])
            logger.info(
                "CBF created, connecting to product controller at %s:%d",
                product_controller_host,
                product_controller_port,
            )
            self._cbf = await CBFRemoteControl.connect(
                product_name,
                product_controller_host,
                product_controller_port,
                cbf_config,
                cbf_mode_config,
            )
            await _report_cbf_config(self._pytestconfig, self._host_config_querier, self._cbf, master_controller_client)
        except Exception as exc:
            self._cbf = FailedCBF(
                name=product_name,
                config=cbf_config,
                mode_config=cbf_mode_config,
                uuid=uuid4(),
                exc=exc,
            )
            custom_report_log(
                self._pytestconfig,
                {
                    "$report_type": "CBFConfiguration",
                    "mode_config": self._cbf.mode_config,
                    "uuid": str(self._cbf.uuid),
                    "error": "".join(traceback.format_exception(self._cbf.exc)),
                },
            )
        return self._cbf

    async def close(self) -> None:
        """Shut down any running CBF and the master controller connection."""
        await self._close_cbf()
        if self._master_controller_client is not None:
            self._master_controller_client.close()
            await self._master_controller_client.wait_closed()
            self._master_controller_client = None
