#!/usr/bin/env python3

import argparse
import asyncio
from typing import Any, Callable, Optional, Type

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
        katcp_type: Type,
        sensor_name: str,
        telstate_name: Optional[str] = None,
        convert: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        if telstate_name is None:
            telstate_name = sensor_name.replace("-", "_")
        full_sensor_name = f"{self.stream}-{sensor_name}"
        reply, informs = await self.client.request("sensor-value", full_sensor_name)
        if len(informs) != 1:
            raise RuntimeError(f"Expected 1 sensor value for {full_sensor_name}, received {len(inform)}")
        value = aiokatcp.decode(katcp_type, informs[0].arguments[4])
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
            elif stream["type"] == "sim.dig.raw_antenna_voltage" and name.endswith("h"):
                # Both pols are represented, but we're only interested in one per antenna.
                converter = SensorConverter(client, telstate, name[:-1])
                converter.set("observer", stream["antenna"])


if __name__ == "__main__":
    asyncio.run(main())
