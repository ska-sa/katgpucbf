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
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass

import aiokatcp
from katsdptelstate.endpoint import endpoint_parser

import katgpucbf.configure_tools
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
        "--controller",
        type=endpoint_parser(5001),
        default="cbf-mc.cbf.mkat.karoo.kat.ac.za",
        help="Endpoint of the CBF master controller",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print config only")
    katgpucbf.configure_tools.add_arguments(parser)
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
    adc_sample_rate = await corr2_client.sensor_value("adc-sample-rate", float)
    int_time = await corr2_client.sensor_value("baseline-correlation-products-int-time", float)
    # This returns a string representation of a list of tuples, with each item
    # in the format (input-label, input-index, LRU host, index-on-host).
    input_labelling_str = await corr2_client.sensor_value("input-labelling", str)
    input_labelling: list[tuple[str, int, str, int]] = ast.literal_eval(input_labelling_str)
    input_labels = [label[0] for label in input_labelling]
    # Trip off the dummy labels
    input_labels = input_labels[: 2 * target.n_antennas]

    # This won't distinguish between the different kinds of S-band. It'll be wrong. I'm not quite
    # sure at this point how to match up the centre_frequency values in katgpucbf.meerkat.BANDS with
    # what MK actually reports, even in the L and UHF cases, so for the purposes of getting
    # something producing a result, I am leaving it as-is for the time being.
    band = next((k for k, v in BANDS.items() if v.adc_sample_rate == adc_sample_rate), None)
    if not band:
        print(
            f"Unable to determine which band. Sample rate reported as {adc_sample_rate}",
            file=sys.stderr,
        )
        return 1

    config: dict = {
        "version": "4.5",
        "config": {},
        "inputs": {
            label: {
                "type": "dig.baseband_voltage",
                "sync_time": sync_time,
                "band": band,
                "adc_sample_rate": adc_sample_rate,
                "centre_frequency": BANDS[band].centre_frequency,  # TODO get from CAM
                "antenna": label[:-1],
                "url": f"spead://{addr}",
            }
            for label, addr in zip(input_labels, mcast_groups)
        },
        "outputs": {
            "antenna-channelised-voltage": {
                "type": "gpucbf.antenna_channelised_voltage",
                "src_streams": input_labels,
                "input_labels": input_labels,
                "n_chans": channels,
            },
            "baseline-correlation-products": {
                "type": "gpucbf.baseline_correlation_products",
                "src_streams": ["antenna-channelised-voltage"],
                "int_time": int_time,
            },
        },
    }

    katgpucbf.configure_tools.apply_arguments(config, args)

    if args.dry_run:
        json.dump(config, sys.stdout, indent=2)
    else:
        client = await aiokatcp.Client.connect(args.controller.host, args.controller.port)
        try:
            reply, _ = await client.request("product-configure", target.name.replace(".", "_"), json.dumps(config))
            pc_host = aiokatcp.decode(str, reply[1])
            pc_port = aiokatcp.decode(int, reply[2])
            print(f"Product controller is at {pc_host}:{pc_port}")
            client.close()
            await client.wait_closed()
            client = await aiokatcp.Client.connect(pc_host, pc_port)
            await client.request("capture-start", "baseline-correlation-products")
            print("baseline-correlation-products is enabled")
        except (aiokatcp.FailReply, ConnectionError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        finally:
            client.close()
            await client.wait_closed()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    sys.exit(main())
