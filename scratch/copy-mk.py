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

It also provides the option for real-time transfer of delays from a live MK
correlator to a MK+ CBF correlator.
"""

import argparse
import ast
import asyncio
import json
import re
import sys
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

import aiokatcp
from katportalclient import KATPortalClient, SensorNotFoundError
from katsdptelstate.endpoint import endpoint_parser

import katgpucbf.configure_tools

#: Scale factors between MK and MK+ wideband gains, based on number of channels
GAIN_SCALE = {
    1024: 5687.284403271646,
    4096: 3096.9068735303676,
    32768: 1147.5800238145268,
}


@dataclass
class SensorQueueItem:
    """Information gathered during sensor updates."""

    name: str
    value: Any


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--portal",
        default="http://portal.mkat.karoo.kat.ac.za/api/client/1",
        help="URL for katportal (including subarray number) [%(default)s]",
    )
    parser.add_argument(
        "--master-controller",
        type=endpoint_parser(5001),
        default="cbf-mc.cbf.mkat.karoo.kat.ac.za",
        help="Endpoint of the CBF master controller",
    )
    parser.add_argument("--name", help="Name of the subarray product to create")
    parser.add_argument(
        "--product-controller",
        type=endpoint_parser(None),
        help="Endpoint of the Product Controller already running",
    )
    parser.add_argument(
        "--transfer", action="store_true", help="Continuously transfer F-engine delays and gains from MK CBF to MK+ CBF"
    )
    parser.add_argument("--sdp", choices=["ingest"], default=None, help="Configure SDP as well")
    parser.add_argument("--dry-run", action="store_true", help="Print config only")
    katgpucbf.configure_tools.add_arguments(parser)
    args = parser.parse_args(argv)
    return args


def _assemble_regex(names: Sequence[str]) -> str:
    """Assemble a regex from a sequence of strings."""
    regex = "|".join(re.escape(name) for name in names)
    return f"^(?:{regex})$"  # Anchor the regex and return


async def single_sensor_value(client: KATPortalClient, name: str) -> Any:
    """Query the value of a single sensor from katportal."""
    regex = "^" + re.escape(name) + "$"  # Anchor and escape the regex
    return (await client.sensor_value(regex)).value


async def multi_sensor_values(client: KATPortalClient, names: Sequence[str]) -> Mapping[str, Any]:
    """Query multiple sensors from katportal.

    This is a workaround for the slow design of katportal, where every query
    iterates over all the sensors even if only a single one is required.

    The return value contains just the sensor value rather than a full sample.
    """
    if not names:
        return {}  # Would get an error trying to query an empty regex
    regex = _assemble_regex(names)
    samples = await client.sensor_values(regex)
    out = {}
    for name in names:
        if name not in samples:
            raise SensorNotFoundError(f"Value for sensor {name} not found")
        out[name] = samples[name].value
    return out


def sensor_callback(
    msg_dict: dict,
    sensor_queue: asyncio.Queue[SensorQueueItem],
) -> None:
    """Receive sensor updates from katportal.

    Populate the `sensor_queue` with an item containing the sensor update.
    """
    queue_item = SensorQueueItem(msg_dict["msg_data"]["name"], msg_dict["msg_data"]["value"])
    try:
        sensor_queue.put_nowait(queue_item)
    except asyncio.QueueFull:
        print(f"Sensor queue full. Dropping this update for {queue_item.name}.")


async def async_main(args) -> int:
    # This queue is used in the case of --transfer
    # but is needed by the callback below when instantiating
    # the portal client.
    sensor_queue: asyncio.Queue[SensorQueueItem] = asyncio.Queue(maxsize=500)
    portal_client = KATPortalClient(
        args.portal,
        partial(
            sensor_callback,
            sensor_queue=sensor_queue,
        ),
    )
    # Get name of the CBF proxy e.g. "cbf_1"
    prefix = await portal_client.sensor_subarray_lookup("cbf", None)
    cbf_sensor_names = [
        "input_labels",
        "wide_antenna_channelised_voltage_source",
        "wide_sync_time",
        "wide_scale_factor_timestamp",
        "wide_antenna_channelised_voltage_n_chans",
        "wide_adc_sample_rate",
        "wide_baseline_correlation_products_int_time",
        "band",
        "delay_centre_frequency",
        "subarray_product_id",
    ]
    sensor_names = [f"{prefix}_{name}" for name in cbf_sensor_names]
    if args.sdp is not None:
        sdp_prefix = await portal_client.sensor_subarray_lookup("sdp", None)
        sdp_subarray_product_id = await single_sensor_value(portal_client, f"{sdp_prefix}_subarray_product_ids")
        # This is the int time for the first ingest process, but they should
        # all be the same.
        sdp_int_time_name = f"{sdp_prefix}_spmc_{sdp_subarray_product_id}_ingest_sdp_l0_1_output_int_time"
        sensor_names.append(sdp_int_time_name)
    sensor_values = await multi_sensor_values(portal_client, sensor_names)

    def cbf_sensor_value(name: str) -> Any:
        return sensor_values[f"{prefix}_{name}"]

    input_labels: list[str] = cbf_sensor_value("input_labels").split(",")
    # Get the multicast groups. This is an annoying sensor because it's not
    # valid Python or JSON: it's a comma-separated list surrounded by brackets,
    # but the strings are not quoted.
    mcast_groups_str: str = cbf_sensor_value("wide_antenna_channelised_voltage_source")
    mcast_groups = [group.strip() for group in mcast_groups_str[1:-1].split(",")]
    for n, group in enumerate(mcast_groups[1:], start=1):
        if group == mcast_groups[0]:
            # MK is not using a full power-of-2 size array. There are dummies from here onwards.
            print(f"Dummy antenna found starting at input {n}")
            input_labels = input_labels[:n]
            mcast_groups = mcast_groups[:n]
            break

    sync_time: float = cbf_sensor_value("wide_sync_time")
    channels: int = cbf_sensor_value("wide_antenna_channelised_voltage_n_chans")
    adc_sample_rate: float = cbf_sensor_value("wide_adc_sample_rate")
    int_time: float = cbf_sensor_value("wide_baseline_correlation_products_int_time")
    band: str = cbf_sensor_value("band")
    centre_frequency: float = cbf_sensor_value("delay_centre_frequency")
    scale_factor_timestamp: float = cbf_sensor_value("wide_scale_factor_timestamp")
    if args.name is None:
        subarray_product_id: str = cbf_sensor_value("subarray_product_id")
        args.name = f"{subarray_product_id}_wide"

    # Fetch the positions of the antennas
    antennas = {label[:-1] for label in input_labels}
    observers = await multi_sensor_values(
        portal_client,
        [f"{antenna}_observer" for antenna in antennas],
    )

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
                "antenna": observers[f"{label[:-1]}_observer"],
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
    if args.sdp is not None:
        config["outputs"]["sdp_l0"] = {
            "type": "sdp.vis",
            "src_streams": ["baseline-correlation-products"],
            "output_int_time": sensor_values[sdp_int_time_name],
            "excise": False,
            "archive": False,  # This is just for testing, so do not archive
            "continuum_factor": 1,
        }

    katgpucbf.configure_tools.apply_arguments(config, args)

    client: aiokatcp.Client | None = None
    pc_host: str = ""
    pc_port: int = 0
    if args.dry_run:
        json.dump(config, sys.stdout, indent=2)
        return 0
    try:
        if args.product_controller:
            client = await aiokatcp.Client.connect(args.product_controller.host, args.product_controller.port)
            print(f"Using existing MK+ CBF correlator at: {args.product_controller}")
        else:
            client = await aiokatcp.Client.connect(args.master_controller.host, args.master_controller.port)
            reply, _ = await client.request("product-configure", args.name, json.dumps(config))
            pc_host = aiokatcp.decode(str, reply[1])
            pc_port = aiokatcp.decode(int, reply[2])
            print(f"Product controller is at {pc_host}:{pc_port}")
            client.close()
            await client.wait_closed()
            client = await aiokatcp.Client.connect(pc_host, pc_port)
            await client.request("capture-start", "baseline-correlation-products")
            print("baseline-correlation-products is enabled")

        if args.transfer:
            qualified_labels = [
                f"{prefix}_wide_antenna_channelised_voltage_{input_label}" for input_label in input_labels
            ]
            # Connect before subscribing
            await portal_client.connect()
            dynamic_sensor_names = [f"{qualified_label}_delay" for qualified_label in qualified_labels] + [
                f"{qualified_label}_eq" for qualified_label in qualified_labels
            ]
            dynamic_sensors_regex = _assemble_regex(dynamic_sensor_names)
            namespace = f"{args.name}_{uuid.uuid4()}"
            await portal_client.subscribe(namespace)
            status = await portal_client.set_sampling_strategies(namespace, dynamic_sensors_regex, "event")
            for sensor_name, result in sorted(status.items()):
                if not result["success"]:
                    print(f"Failed to set sampling strategy for {sensor_name}")

            # There is a separate delay sensor for each input, but the delays
            # need to be set with a single request. So we accumulate delays
            # into delay_args until we have a set that covers all inputs with
            # the same loadmcnt.
            delay_args: dict[str, str | None] = dict.fromkeys(input_labels)
            delay_ref_loadmcnt = 0
            while True:
                queue_item = await sensor_queue.get()
                if m := re.match(rf"{prefix}_wide_antenna_channelised_voltage_(.*)_delay", queue_item.name):
                    input_label = m.group(1)
                    # (loadmcnt <ADC sample count when model was loaded>, delay <in seconds>,
                    # delay-rate <unit-less or, seconds-per-second>, phase <radians>,
                    # phase-rate <radians per second>)
                    loadmcnt, delay, delay_rate, phase, phase_rate = ast.literal_eval(queue_item.value)
                    if delay_ref_loadmcnt == 0:
                        # This is the first delay update we're seeing
                        delay_ref_loadmcnt = loadmcnt
                    elif delay_ref_loadmcnt != loadmcnt:
                        delay_ref_loadmcnt = loadmcnt
                        print(f"{input_label}: New loadmcnt {delay_ref_loadmcnt}. Getting new delays.")
                        delay_args = dict.fromkeys(input_labels)  # Clear the stale values
                    delay_args[input_label] = f"{delay},{delay_rate}:{phase},{phase_rate}"
                    if all(delay_args_value is not None for delay_args_value in delay_args.values()):
                        unix_timestamp = loadmcnt / scale_factor_timestamp + sync_time
                        sorted_delay_args = [delay_args[label] for label in input_labels]
                        await client.request(
                            "delays",
                            "antenna-channelised-voltage",
                            unix_timestamp,
                            *sorted_delay_args,
                        )
                        delay_args = dict.fromkeys(input_labels)
                elif m := re.match(rf"{prefix}_wide_antenna_channelised_voltage_(.*)_eq", queue_item.name):
                    input_label = m.group(1)
                    gains = ast.literal_eval(queue_item.value)
                    # Scale to get incoherent gain
                    gains = [gain / GAIN_SCALE[channels] for gain in gains]
                    print(f"Updating gains for {input_label}")
                    await client.request(
                        "gain", "antenna-channelised-voltage", input_label, *[str(gain) for gain in gains]
                    )
    except (aiokatcp.FailReply, ConnectionError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        if portal_client.is_connected:
            portal_client.disconnect()
        client.close()
        await client.wait_closed()

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    sys.exit(main())
