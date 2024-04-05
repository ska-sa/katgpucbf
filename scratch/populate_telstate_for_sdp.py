#!/usr/bin/env python3

################################################################################
# Copyright (c) 2021-2023, National Research Foundation (SARAO)
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

import argparse
import asyncio
from collections.abc import Callable
from typing import Any

import aiokatcp
import numpy as np
from katsdptelstate import TelescopeState
from katsdptelstate.endpoint import endpoint_parser


class SensorConverter:
    """Read katcp sensors and sets values in telstate."""

    def __init__(self, client: aiokatcp.Client, telstate: TelescopeState, stream: str):
        self.client = client
        self.telstate_view = telstate.view(stream)
        self.stream = stream

    def set(self, key: str, value: Any) -> None:
        self.telstate_view[key] = value
        print(f"Set {self.stream}_{key} to {value}")

    async def __call__(
        self,
        katcp_type: type,
        sensor_name: str,
        telstate_name: str | None = None,
        convert: Callable[[Any], Any] | None = None,
    ) -> None:
        if telstate_name is None:
            telstate_name = sensor_name.replace("-", "_")
        full_sensor_name = f"{self.stream}-{sensor_name}"
        value = await self.client.sensor_value(full_sensor_name, katcp_type)
        if convert is not None:
            value = convert(value)
        self.set(telstate_name, value)
        return value


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "controller", type=endpoint_parser(None), metavar="HOST:PORT", help="Product controller endpoint"
    )
    args = parser.parse_args()

    client = await aiokatcp.Client.connect(args.controller.host, args.controller.port)
    async with client:
        telstate_endpoint, _ = await client.request("telstate-endpoint")
        if not telstate_endpoint:
            raise RuntimeError("No telstate endpoint available")
        telstate_endpoint = aiokatcp.decode(str, telstate_endpoint[0])
        print(f"Telstate is at {telstate_endpoint}")
        telstate = TelescopeState(telstate_endpoint)
        config = telstate["sdp_config"]
        for name, stream in config["outputs"].items():
            converter = SensorConverter(client, telstate, name)
            if "src_streams" in stream:
                converter.set("src_streams", stream["src_streams"])
            if stream["type"] == "gpucbf.baseline_correlation_products":
                await converter(int, "n-chans")
                await converter(int, "n-chans-per-substream")
                await converter(int, "n-accs")
                bls_ordering = await converter(str, "bls-ordering", convert=np.safe_eval)
                # No sensor for input-labels, so infer from bls_ordering
                input_labels = []
                for bls in bls_ordering:
                    for inp in bls:
                        if inp not in input_labels:
                            input_labels.append(inp)
                converter.set("input_labels", input_labels)
                await converter(float, "int-time")
                converter.set("instrument_dev_name", "dummy")
            elif stream["type"] == "gpucbf.antenna_channelised_voltage":
                await converter(float, "bandwidth")
                await converter(aiokatcp.Timestamp, "sync-time", convert=float)
                await converter(int, "n-samples-between-spectra", "ticks_between_spectra")
                await converter(float, "scale-factor-timestamp")
                converter.set("instrument_dev_name", "dummy")
                # TODO: get sky centre frequency from digitiser config?
                converter.set("center_freq", 1284e6)
            elif stream["type"] == "sim.dig.baseband_voltage" and name.endswith("h"):
                # Both pols are represented, but we're only interested in one per antenna.
                converter = SensorConverter(client, telstate, name[:-1])
                converter.set("observer", stream["antenna"])


if __name__ == "__main__":
    asyncio.run(main())
