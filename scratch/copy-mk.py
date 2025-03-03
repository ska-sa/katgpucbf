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
import re
import sys
import uuid
from collections.abc import Mapping, Sequence
from functools import partial
from typing import Any

import aiokatcp
from katportalclient import KATPortalClient, SensorNotFoundError
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
        "--master-controller",
        type=endpoint_parser(5001),
        default="cbf-mc.cbf.mkat.karoo.kat.ac.za",
        help="Endpoint of the CBF master controller",
    )
    parser.add_argument("--name", help="Name of the subarray product to create")
    parser.add_argument(
        "--transfer-delays", action="store_true", help="Continuously transfer F-engine delays from MK CBF to MK+ CBF"
    )
    parser.add_argument("--dry-run", action="store_true", help="Print config only")
    katgpucbf.configure_tools.add_arguments(parser)
    args = parser.parse_args(argv)
    return args


async def multi_sensor_values(client: KATPortalClient, names: Sequence[str]) -> Mapping[str, Any]:
    """Query multiple sensors from katportal.

    This is a workaround for the slow design of katportal, where every query
    iterates over all the sensors even if only a single one is required.

    The return value contains just the sensor value rather than a full sample.
    """
    regex = "|".join(re.escape(name) for name in names)
    if not regex:
        return {}  # Would get an error trying to query an empty regex
    regex = f"^(?:{regex})$"  # Anchor the regex
    samples = await client.sensor_values(regex)
    out = {}
    for name in names:
        if name not in samples:
            raise SensorNotFoundError(f"Value for sensor {name} not found")
        out[name] = samples[name].value
    return out


def delay_transfer_callback(
    msg_dict: dict,
    delays_queue: asyncio.Queue[tuple[int, str, str]],
) -> tuple:
    """Transfer F-engine delays from MK subarray to MK+ F-engines.

    It currently puts a tuple on the `delays_queue` for each sensor update
    in the format (loadmcnt, src_name, request_args). The `request_args`
    is a string formatted as:
    - delay,delay_rate:phase,phase_rate
    """
    # TODO: Some delay sensors are logging multiple entries to the queue
    # before others even get a chance. Not sure if this is the place to
    # control that. Either way, the queue is filling faster than it can
    # be consumed.
    delay_sensor_value = msg_dict["msg_data"]["value"]
    src_name: str = msg_dict["msg_data"]["name"].split("_")[-2]
    # (loadmcnt <ADC sample count when model was loaded>, delay <in seconds>,
    # delay-rate <unit-less or, seconds-per-second>, phase <radians>,
    # phase-rate <radians per second>)
    loadmcnt, delay, delay_rate, phase, phase_rate = delay_sensor_value.replace(" ", "").split(",")
    request_args = f"{delay},{delay_rate}:{phase},{phase_rate[:-1]}"
    request_tuple = (int(loadmcnt[1:]), src_name, request_args)

    if delays_queue.full():
        # TODO: Figure out a better approach
        # Get rid of older items first (?)
        _ = delays_queue.get_nowait()
    delays_queue.put_nowait(request_tuple)


def format_delay_request(
    delays_tuples: list[tuple[int, str, str]],
    sync_time: float,
    scale_factor_timestamp: float,
) -> tuple[float, str]:
    """Confirm there are enough arguments for a complete ?delays request.

    Additionally, convert the ADC mcnt to a UNIX timestamp as required by
    MK+ CBF F-engine ?delays request.

    Parameters
    ----------
    delays_tuples
        List of tuples in the format (loadmcnt, src_name, delay_request_args).
    sync_time
        The time at which the digitisers were synchronised. Seconds since
        Unix Epoch. Used to convert the loadmcnt to a unix timestamp.
    scale_factor_timestamp
        Factor by which to divide instrument timestamps to convert to unix
        seconds. Used to conver the loadmcnt to a unix timestamp.

    Returns
    -------
    unix_timestamp, coefficient_set
        Arguments required for ?delays request. Time at which delays
        will be applied. Coefficients for each input, in the format
        ``delay,delay_rate:phase,phase_rate``.
    """
    ref_loadmcnt = delays_tuples[0][0]
    filtered_values = list(filter(lambda delays_tuple: delays_tuple[0] == ref_loadmcnt, delays_tuples))
    unix_timestamp = ref_loadmcnt / scale_factor_timestamp + sync_time
    return unix_timestamp, " ".join(filtered_value[-1] for filtered_value in filtered_values)


async def async_main(args) -> int:
    # Perhaps best to just start with a client to get subarray info
    portal_client = KATPortalClient(args.portal, None)
    # Get name of the CBF proxy e.g. "cbf_1"
    prefix = await portal_client.sensor_subarray_lookup("cbf", None)
    cbf_sensor_names = [
        "input_labels",
        "wide_scale_factor_timestamp",
        "wide_antenna_channelised_voltage_source",
        "wide_sync_time",
        "wide_antenna_channelised_voltage_n_chans",
        "wide_adc_sample_rate",
        "wide_baseline_correlation_products_int_time",
        "band",
        "delay_centre_frequency",
        "subarray_product_id",
    ]
    cbf_sensor_values = await multi_sensor_values(
        portal_client,
        [f"{prefix}_{name}" for name in cbf_sensor_names],
    )

    def cbf_sensor_value(name: str) -> Any:
        return cbf_sensor_values[f"{prefix}_{name}"]

    input_labels: list[str] = cbf_sensor_value("input_labels").split(",")
    n_src_inputs = len(input_labels)
    # Create asyncio Queue for the delay-callback to push items onto
    delays_queue: asyncio.Queue[tuple[int, str, str]] = asyncio.Queue(maxsize=n_src_inputs * 8)

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

    katgpucbf.configure_tools.apply_arguments(config, args)

    client: aiokatcp.Client | None = None
    pc_host: str = ""
    pc_port: int = 0
    if args.dry_run:
        json.dump(config, sys.stdout, indent=2)
        return 1
    else:
        client = await aiokatcp.Client.connect(args.master_controller.host, args.master_controller.port)
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

    if args.transfer_delays:
        stream_names = [f"antenna-channelised-voltage-{input_label}" for input_label in input_labels]
        # New client because it needs to be initialised with the callback (?)
        callback_client = KATPortalClient(
            args.portal,
            partial(
                delay_transfer_callback,
                delays_queue=delays_queue,
            ),
        )
        # Connect before subscribing
        await callback_client.connect()
        delay_sensors_regex = (
            "^(?:"
            + "|".join(f"{prefix}_wide_{stream_name.replace("-", "_")}_delay" for stream_name in stream_names)
            + ")$"
        )
        # TODO: Check against results returned from the following to
        # ensure the `callback_client` is ready to proceed.
        namespace = f"{args.name}_{uuid.uuid4()}"
        await callback_client.subscribe(namespace)
        await callback_client.set_sampling_strategies(namespace, delay_sensors_regex, "event")

        def _reset_dict(dict_to_reset: dict[Any, Any], keys: list) -> dict[Any, None]:
            dict_to_reset = {new_key: None for new_key in keys}
            return dict_to_reset

        delay_args: dict[str, tuple | None] = {}
        _reset_dict(delay_args, input_labels)
        ref_loadmcnt = 0
        try:
            pc_client = await aiokatcp.Client.connect(pc_host, pc_port)
            while True:
                complete_requests = False
                while not complete_requests:
                    # delays_queue items are in the format
                    # - (loadmcnt, src_name, delay_request_args)
                    delays_queue_item = await delays_queue.get()
                    if ref_loadmcnt == 0:
                        # This is the first delay update we're seeing
                        # TODO: Potentially move this back outside the while
                        # so it doesn't need to be checked on every loop.
                        ref_loadmcnt = delays_queue_item[0]
                    if delays_queue_item[0] != ref_loadmcnt:
                        ref_loadmcnt = delays_queue_item[0]
                        print(f"{delays_queue_item[1]}: New loadmcnt {ref_loadmcnt}. Getting new delays.")
                        _reset_dict(delay_args, input_labels)

                    delay_args[delays_queue_item[1]] = delays_queue_item
                    complete_requests = all(delay_args_value is not None for delay_args_value in delay_args.values())

                unix_timestamp, coefficient_set = format_delay_request(
                    [item[-1] for item in sorted(delay_args.items())],
                    sync_time=sync_time,
                    scale_factor_timestamp=scale_factor_timestamp,
                )
                await pc_client.request("delays", aiokatcp.encode(unix_timestamp), aiokatcp.encode(coefficient_set))
                _reset_dict(delay_args, input_labels)
        finally:
            callback_client.disconnect()
            pc_client.close()
            await pc_client.wait_closed()

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    sys.exit(main())
