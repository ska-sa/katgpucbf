#!/usr/bin/env python3

################################################################################
# Copyright (c) 2024-2025, National Research Foundation (SARAO)
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
import ast
import asyncio
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass

import aiokatcp

from katgpucbf.meerkat import BANDS


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
    parser.add_argument("--subordinate", type=str, required=True, help="Which subordinate to copy.")
    parser.add_argument(
        "--develop",
        nargs="?",
        const=True,
        help="Pass development options in the config. Use comma separation, or omit the arg to enable all.",
    )
    parser.add_argument("--image-override", action="append", metavar="NAME:IMAGE:TAG", help="Override a single image")
    parser.add_argument(
        "--controller", default="cbf-mc.cbf.mkat.karoo.kat.ac.za", help="Hostname of the CBF master controller"
    )
    args = parser.parse_args(argv)
    return args


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
                # MK is not using a full power-of-2 size array. There are dummies from here onwards.
                print(f"Dummy antenna found starting at {(n + 1) // 2}")
                n_antennas = (n + 1) // 2
                break
        digitiser_address = mcast_groups[0].split("+")[0]

        subordinates[name] = Subordinate(name, control_port, monitor_port, digitiser_address, n_antennas)

    try:
        target = subordinates[args.subordinate]
    except KeyError:
        print(
            f"{args.subordinate} is not among the subordinates currently running. "
            f"Options are: {list(subordinates.keys())}",
            file=sys.stderr,
        )
        return 1

    # connect to the corr2_servlet itself to see the sync time, n_chans and bandwidth
    corr2_client = await aiokatcp.Client.connect(args.cmc_host, target.control_port)
    sync_time = await corr2_client.sensor_value("sync-time", float)
    channels = await corr2_client.sensor_value("antenna-channelised-voltage-n-chans", int)
    sample_rate = await corr2_client.sensor_value("adc-sample-rate", float)
    # This returns a string representation of a list of tuples, with each item
    # in the format (input-label, input-index, LRU host, index-on-host).
    input_labelling_str = await corr2_client.sensor_value("input-labelling", str)
    input_labelling: list[tuple[str,]] = ast.literal_eval(input_labelling_str)
    input_labels = [label[0] for label in input_labelling]

    # This won't distinguish between the different kinds of S-band. It'll be wrong. I'm not quite
    # sure at this point how to match up the centre_frequency values in katgpucbf.meerkat.BANDS with
    # what MK actually reports, even in the L and UHF cases, so for the purposes of getting
    # something producing a result, I am leaving it as-is for the time being.
    band = next((k for k, v in BANDS.items() if v.adc_sample_rate == sample_rate), None)
    if not band:
        print(
            f"Unable to determine which band. Sample rate reported as {sample_rate}",
            file=sys.stderr,
        )
        return 1

    out_kwargs = {
        "name": target.name.replace(".", "_"),
        "antennas": target.n_antennas,
        "channels": channels,
        "digitiser-address": mcast_groups[0].split("+")[0],
        "sync-time": sync_time,
        "band": band,
        "develop": args.develop,
        # We don't want the dummy antennas for sim_correlator.py
        "input-labels": ",".join(input_labels[: 2 * target.n_antennas]),
    }

    out_cmd = [os.path.join(os.path.dirname(__file__), "sim_correlator.py")]
    out_cmd += [f"--{k}={v}" for k, v in out_kwargs.items() if v is not None]

    if args.image_override is not None:
        for override in args.image_override:
            out_cmd.append(f"--image-override={override}")

    out_cmd.append(args.controller)

    print(f"Executing: {out_cmd}\n")
    os.execv(out_cmd[0], out_cmd)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    sys.exit(main())
