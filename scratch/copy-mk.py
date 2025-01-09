#!/usr/bin/env python3

################################################################################
# Copyright (c) 2024, National Research Foundation (SARAO)
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

"""Copy a running MK correlator.

This script makes use of sim_correlator.py in order to start a correlator, but
instead of a simulated one, it gets the parameters from a live MK correlator.
"""

import argparse
import asyncio
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import aiokatcp


@dataclass
class Subordinate:
    name: str
    control_port: int
    monitor_port: int
    digitiser_address: str
    n_antennas: int


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("cmc_host", help="Hostname / IP address of the CMC.")
    parser.add_argument("--cmc-port", type=int, default=7147, help="KATCP port of the CMC. [%(default)s]")
    parser.add_argument("--target", type=str, required=True, help="Which subordinate to copy.")
    parser.add_argument(
        "--develop",
        nargs="?",
        const=True,
        help="Pass development options in the config. Use comma separation, or omit the arg to enable all.",
    )
    parser.add_argument("--image-override", action="append", metavar="NAME:IMAGE:TAG", help="Override a single image")
    args = parser.parse_args(argv)
    return args


async def get_sensor_value(client: aiokatcp.Client, sensor_name: str, sensor_type: Callable):
    """Retrieve a sensor value from a katcp client."""
    try:
        _, informs = await client.request("sensor-value", sensor_name)
    except (aiokatcp.FailReply, ConnectionError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    return sensor_type(informs[0].arguments[4].decode())


async def async_main(args) -> int:
    cmc_client = await aiokatcp.Client.connect(args.cmc_host, args.cmc_port)

    try:
        _, informs = await cmc_client.request("subordinate-list")
    except (aiokatcp.FailReply, ConnectionError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if len(informs) == 0:
        print(f"No subarrays currently running on {args.cmc_host}")
        return 1

    subordinates = {}
    for inform in informs:
        name = inform.arguments[0].decode()
        control_port, monitor_port = inform.arguments[1].decode().split(",")
        control_port = int(control_port)
        monitor_port = int(monitor_port)
        mcast_groups = []
        for grp in inform.arguments[2:]:
            mcast_groups.append(grp.decode())
        n_antennas = len(mcast_groups) // 2
        for n, grp in enumerate(mcast_groups[1:]):
            if grp == mcast_groups[0]:
                # MK is not using a full power-of-2 size array, there are dummies from here onwards.
                print("Dummy antenna found starting at ", (n + 1) // 2)
                n_antennas = (n + 1) // 2
                break
        digitiser_address = mcast_groups[0].split("+")[0]

        subordinates[name] = Subordinate(name, control_port, monitor_port, digitiser_address, n_antennas)

    try:
        target = subordinates[args.target]
    except KeyError:
        print(f"{args.target} is not among the subordinates currently running. Options are: {subordinates.keys()}", file=sys.stderr)
        return 1

    # connect to the corr2_servlet itself to see the sync time, n_chans and bandwidth
    corr2_client = await aiokatcp.Client.connect(args.cmc_host, target.control_port)
    sync_time = await get_sensor_value(corr2_client, "sync-time", float)
    channels = await get_sensor_value(corr2_client, "antenna-channelised-voltage-n-chans", int)
    bandwidth = await get_sensor_value(corr2_client, "antenna-channelised-voltage-bandwidth", float)
    # TODO: this could probably be better, and doesn't support S-band yet.
    match bandwidth:
        case 544000000.0:
            band = "u"
        case 856000000.0:
            band = "l"
        case _:
            print(f"Unable to determine which band we are in. Bandwidth reported as {bandwidth}", file=sys.stderr)
            return 1

    out_kwargs = {
        "name": target.name.replace(".", "_"),
        "antennas": target.n_antennas,
        "channels": channels,
        "digitiser-address": mcast_groups[0].split("+")[0],
        "sync-time": sync_time,
        "band": band,
    }

    out_cmd = [
        "python",
        "sim_correlator.py",
        "cbf-mc.cbf.mkat.karoo.kat.ac.za",
    ]

    for k, v in out_kwargs.items():
        out_cmd.append(f"--{k}={v}")

    if args.develop is not None:
        out_cmd.append(f"--develop={args.develop}")

    if args.image_override is not None:
        for override in args.image_override:
            out_cmd.append(f"--image-override={override}")

    output = subprocess.run(out_cmd, capture_output=True)
    print(output)

    return output.returncode


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    sys.exit(main())
