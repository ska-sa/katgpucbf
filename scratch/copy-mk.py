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
import asyncio
import json
import sys
from collections.abc import Sequence
from typing import Any

import aiokatcp
from katportalclient import KATPortalClient
from katsdptelstate.endpoint import endpoint_parser

import katgpucbf.configure_tools


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--portal",
        default="http://portal.mkat.karoo.kat.ac.za/api/client/1",
        help="URL for katportal (including subarray number) [%(default)s]",
    )
    parser.add_argument(
        "--controller",
        type=endpoint_parser(5001),
        default="cbf-mc.cbf.mkat.karoo.kat.ac.za",
        help="Endpoint of the CBF master controller",
    )
    parser.add_argument("--name", help="Name of the subarray product")
    parser.add_argument("--dry-run", action="store_true", help="Print config only")
    katgpucbf.configure_tools.add_arguments(parser)
    args = parser.parse_args(argv)
    return args


async def async_main(args) -> int:
    portal_client = KATPortalClient(args.portal, None)
    # Get name of the CBF proxy e.g. "cbf_1"
    prefix = await portal_client.sensor_subarray_lookup("cbf", "")

    async def cbf_sensor_value(name: str) -> Any:
        sample = await portal_client.sensor_value(f"{prefix}_{name}")
        return sample.value

    input_labels = (await cbf_sensor_value("input_labels")).split(",")
    # Get the multicast groups. This is an annoying sensor because it's not
    # valid Python or JSON: it's a comma-separated list surrounded by brackets,
    # but the strings are not quoted.
    mcast_groups_str = await cbf_sensor_value("wide_antenna_channelised_voltage_source")
    mcast_groups = [group.strip() for group in mcast_groups_str[1:-1].split(",")]
    for n, group in enumerate(mcast_groups[1:], start=1):
        if group == mcast_groups[0]:
            # MK is not using a full power-of-2 size array. There are dummies from here onwards.
            print(f"Dummy antenna found starting at input {n}")
            input_labels = input_labels[:n]
            mcast_groups = mcast_groups[:n]
            break

    sync_time: float = await cbf_sensor_value("wide_sync_time")
    channels: int = await cbf_sensor_value("wide_antenna_channelised_voltage_n_chans")
    adc_sample_rate: float = await cbf_sensor_value("wide_adc_sample_rate")
    int_time: float = await cbf_sensor_value("wide_baseline_correlation_products_int_time")
    band: str = await cbf_sensor_value("band")
    centre_frequency: float = await cbf_sensor_value("delay_centre_frequency")
    if args.name is None:
        subarray_product_id: str = await cbf_sensor_value("subarray_product_id")
        args.name = f"{subarray_product_id}_wide"

    config: dict = {
        "version": "4.5",
        "config": {},
        "inputs": {
            label: {
                "type": "dig.baseband_voltage",
                "sync_time": sync_time,
                "band": band,
                "adc_sample_rate": adc_sample_rate,
                "centre_frequency": centre_frequency,
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
            reply, _ = await client.request("product-configure", args.name, json.dumps(config))
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
