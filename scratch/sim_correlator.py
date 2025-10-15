#!/usr/bin/env python3

################################################################################
# Copyright (c) 2021-2025, National Research Foundation (SARAO)
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

"""Start a simulated correlator via the SDP master controller.

This script constructs a JSON dictionary to describe a correlator simulation,
and submits it via katcp to an SDP master controller. Alternatively, the
configuration can be written to file to be manually started later.
"""

import argparse
import asyncio
import contextlib
import ipaddress
import json
import re
import sys
from collections.abc import Sequence
from fractions import Fraction

import aiokatcp

import katgpucbf.configure_tools
from katgpucbf.main import comma_split
from katgpucbf.meerkat import BANDS


def parse_input_labels(value: str) -> list[str]:
    return value.split(",")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command-line arguments (which may be specified as a parameter)."""

    parser = argparse.ArgumentParser()
    parser.add_argument("controller", help="Hostname of the SDP master controller")
    parser.add_argument("--port", type=int, default=5001, help="TCP port of the SDP master controller [%(default)s]")
    parser.add_argument("--name", default="sim_correlator", help="Subarray product name [%(default)s]")
    parser.add_argument("--input-labels", type=parse_input_labels, help="Input labels for F-engine sources")
    parser.add_argument("-a", "--antennas", type=int, required=True, help="Number of antennas")
    parser.add_argument("-d", "--digitisers", type=int, help="Number of digitisers [#antennas]")
    parser.add_argument("-c", "--channels", type=int, required=True, help="Number of channels")
    parser.add_argument("-i", "--int-time", type=float, default=0.5, help="Integration time in seconds [%(default)s]")
    parser.add_argument(
        "--last-stage",
        choices=["d", "f", "xb", "v", "ingest"],
        default="v",
        help="Do not run any stages past the given one [%(default)s]",
    )
    parser.add_argument(
        "--digitiser-address",
        type=ipaddress.IPv4Address,
        metavar="ADDRESS",
        help="Starting IP address for external digitisers",
    )
    parser.add_argument("--sync-time", type=float, help="Digitiser sync time [current time]")
    parser.add_argument("--band", default="l", choices=BANDS.keys(), help="Band ID [%(default)s]")
    parser.add_argument("--adc-sample-rate", type=float, help="ADC sample rate in Hz [from --band]")
    parser.add_argument("--centre-frequency", type=float, help="Sky centre frequency in Hz [from --band]")
    parser.add_argument("--beams", type=int, default=0, help="Number of dual-polarisation wideband beams [%(default)s]")
    parser.add_argument("--narrowband", action="store_true", help="Enable a narrowband output [no]")
    parser.add_argument(
        "--narrowband-decimation", type=int, default=8, help="Narrowband decimation factor [%(default)s]"
    )
    parser.add_argument("--narrowband-channels", type=int, default=32768, help="Narrowband channels [%(default)s]")
    parser.add_argument(
        "--narrowband-centre-frequency", type=float, help="Narrow baseband centre frequency [centre of wideband]"
    )
    parser.add_argument(
        "--narrowband-beams", type=int, default=0, help="Number of dual-polarisation narrowband beams [%(default)s]"
    )
    parser.add_argument("--vlbi", action="store_true", help="Enable VLBI mode [no]")
    parser.add_argument(
        "--vlbi-recv-pols",
        type=comma_split(str, 2),
        metavar="[+-]P,[+-]P",
        required=True,
        help="Input polarisations for VLBI (±x, ±y, ±L or ±R)",
    )
    parser.add_argument(
        "--vlbi-send-station-id", type=str, default="me", help="VDIF Station ID for VLBI output [%(default)s]"
    )

    katgpucbf.configure_tools.add_arguments(parser)
    parser.add_argument(
        "-w", "--write", action="store_true", help="Write to file (give filename instead of the controller)"
    )
    args = parser.parse_args(argv)
    if args.adc_sample_rate is None:
        args.adc_sample_rate = BANDS[args.band].adc_sample_rate
    if args.centre_frequency is None:
        args.centre_frequency = BANDS[args.band].centre_frequency
    if args.narrowband_centre_frequency is None:
        args.narrowband_centre_frequency = args.adc_sample_rate / 4
    if args.digitisers is None:
        args.digitisers = args.antennas
    if args.digitiser_address is not None and args.sync_time is None:
        parser.error("--sync-time is required when specifying --digitiser-address")
    if args.input_labels is None:
        args.input_labels = [f"m{800 + i}{pol}" for i in range(args.antennas) for pol in ["v", "h"]]
    if args.vlbi and args.narrowband_beams == 0:
        args.narrowband_beams = 1
    # TODO: Move the recv-pols error-checking to a helper function
    for pol in args.vlbi_recv_pols:
        if not re.fullmatch(r"^[-+]?[xyLR]", pol):
            parser.error(f"{pol!r} is not a valid --vlbi-recv-pol value")
    if set(pol[-1] for pol in args.vlbi_recv_pols) not in [{"x", "y"}, {"L", "R"}]:
        parser.error("--vlbi-recv-pol is not an orthogonal polarisation basis")
    return args


def generate_digitisers(args: argparse.Namespace, config: dict) -> list[str]:
    """Populate configuration for digitisers.

    Returns
    -------
    dig_names
        Names of the digitiser streams
    """
    next_dig_ip = args.digitiser_address
    dig_names = []
    for ant_index in range(args.digitisers):
        number = 800 + ant_index  # Avoid confusion with real antennas
        for pol in ["h", "v"]:
            name = f"m{number}{pol}"
            dig_names.append(name)
            if args.digitiser_address is None:
                config["outputs"][name] = {
                    "type": "sim.dig.baseband_voltage",
                    "band": args.band[:1],
                    "adc_sample_rate": args.adc_sample_rate,
                    "centre_frequency": args.centre_frequency,
                    "antenna": f"m{number}, 0:0:0, 0:0:0, 0, 0",
                }
                if args.sync_time is not None:
                    config["outputs"][name]["sync_time"] = args.sync_time
            else:
                config["inputs"][name] = {
                    "type": "dig.baseband_voltage",
                    "sync_time": args.sync_time,
                    "band": args.band[:1],
                    "adc_sample_rate": args.adc_sample_rate,
                    "centre_frequency": args.centre_frequency,
                    "antenna": f"m{number}, 0:0:0, 0:0:0, 0, 0",
                    "url": f"spead://{next_dig_ip}+7:7148",
                }
                next_dig_ip += 8
    return dig_names


def generate_antenna_channelised_voltage(args: argparse.Namespace, outputs: dict, dig_names: list[str]) -> None:
    """Populate configuration for antenna-channelised-voltage streams."""
    outputs["antenna-channelised-voltage"] = {
        "type": "gpucbf.antenna_channelised_voltage",
        # Cycle through digitisers as necessary
        "src_streams": [dig_names[i % len(dig_names)] for i in range(2 * args.antennas)],
        "input_labels": args.input_labels,
        "n_chans": args.channels,
    }
    if args.narrowband or args.vlbi:
        outputs["narrow0-antenna-channelised-voltage"] = {
            **outputs["antenna-channelised-voltage"],  # Copy from wideband
            "n_chans": args.narrowband_channels,
            "narrowband": {
                "decimation_factor": args.narrowband_decimation,
                "centre_frequency": args.narrowband_centre_frequency,
            },
        }
        if args.vlbi:
            bandwidth_ratio = Fraction(107, 64)  # Fixed ratio for VLBI mode
            bandwidth = Fraction(args.adc_sample_rate) / Fraction(2) / Fraction(args.narrowband_decimation)
            pass_bandwidth = Fraction(bandwidth) / Fraction(bandwidth_ratio)
            # The JSON encodeer doesn't like the Fraction type, so convert to float
            # TODO: Clarify decimal point accuracy requirement
            outputs["narrow0-antenna-channelised-voltage"]["narrowband"]["vlbi"] = {
                "pass_bandwidth": float(pass_bandwidth)
            }


def generate_baseline_correlation_products(args: argparse.Namespace, outputs: dict) -> None:
    """Populate configuration for baseline-correlation-products streams."""
    outputs["baseline-correlation-products"] = {
        "type": "gpucbf.baseline_correlation_products",
        "src_streams": ["antenna-channelised-voltage"],
        "int_time": args.int_time,
    }
    if args.narrowband or args.vlbi:
        outputs["narrow0-baseline-correlation-products"] = {
            **outputs["baseline-correlation-products"],  # Copy from wideband
            "src_streams": ["narrow0-antenna-channelised-voltage"],
        }


def generate_tied_array_channelised_voltage(args: argparse.Namespace, outputs: dict) -> None:
    """Populate configuration for tied-array-channelised-voltage streams."""
    for i in range(args.beams):
        for pol_idx, pol_name in enumerate("xy"):
            outputs[f"tied-array-channelised-voltage-{i}{pol_name}"] = {
                "type": "gpucbf.tied_array_channelised_voltage",
                "src_streams": ["antenna-channelised-voltage"],
                "src_pol": pol_idx,
            }
    if args.narrowband or args.vlbi:
        for i in range(args.narrowband_beams):
            for pol_idx, pol_name in enumerate("xy"):
                outputs[f"narrow0-tied-array-channelised-voltage-{i}{pol_name}"] = {
                    "type": "gpucbf.tied_array_channelised_voltage",
                    "src_streams": ["narrow0-antenna-channelised-voltage"],
                    "src_pol": pol_idx,
                }


def generate_tied_array_resampled_voltage(args: argparse.Namespace, outputs: dict) -> None:
    """Populate configuration for tied-array-resampled-voltage streams."""
    outputs["tied-array-resampled-voltage"] = {
        "type": "gpucbf.tied-array-resampled-voltage",
        "src_streams": [
            f"narrow0-tied-array-channelised-voltage-{i}{pol_name}"
            for i in range(args.narrowband_beams)
            for pol_name in "xy"
        ],
        "n_chans": 2,
        "pols": args.vlbi_recv_pols,
        "station_id": args.vlbi_send_station_id,
    }


def generate_sdp(args: argparse.Namespace, outputs: dict) -> None:
    outputs["sdp_l0"] = {
        "type": "sdp.vis",
        "src_streams": ["baseline-correlation-products"],
        "output_int_time": args.int_time,
        "excise": False,
        "archive": False,
        "continuum_factor": 1,
    }
    if args.narrowband:
        outputs["sdp_l0_narrow0"] = {
            **outputs["sdp_l0"],  # Copy from wideband
            "src_streams": ["narrow0-baseline-correlation-products"],
        }


def generate_config(args: argparse.Namespace) -> dict:
    """Produce the configuration dict from the parsed command-line arguments."""
    config: dict = {
        "version": "4.6",
        "config": {"mirror_sensors": False},
        "inputs": {},
        "outputs": {},
    }
    katgpucbf.configure_tools.apply_arguments(config, args)

    dig_names = generate_digitisers(args, config)
    if args.last_stage == "d":
        return config

    generate_antenna_channelised_voltage(args, config["outputs"], dig_names)
    if args.last_stage == "f":
        return config

    generate_baseline_correlation_products(args, config["outputs"])
    generate_tied_array_channelised_voltage(args, config["outputs"])
    if args.last_stage == "xb":
        return config

    generate_tied_array_resampled_voltage(args, config["outputs"])
    if args.last_stage == "v":
        return config

    generate_sdp(args, config["outputs"])

    return config


async def issue_config(host: str, port: int, name: str, config: dict) -> int:
    """Connect to the product controller and issue ``?product-configure``).

    Returns
    -------
    exit_code
        Exit code for the process (0 on success,non-zero on error)
    """
    client = await aiokatcp.Client.connect(host, port)
    try:
        reply, _ = await client.request("product-configure", name, json.dumps(config))

        product_host = reply[1].decode()
        product_port = int(reply[2].decode())
        print(f"Product controller is at {product_host}:{product_port}")

        product_client = await aiokatcp.Client.connect(product_host, product_port)
        for output_name, output in config["outputs"].items():
            if output["type"] in {"gpucbf.baseline_correlation_products", "gpucbf.tied_array_channelised_voltage"}:
                print(f"Enabling {output_name} transmission...")
                await product_client.request("capture-start", output_name)
    except (aiokatcp.FailReply, ConnectionError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = generate_config(args)
    if args.write:
        with contextlib.ExitStack() as exit_stack:
            if args.controller == "-":
                f = sys.stdout
            else:
                f = exit_stack.enter_context(open(args.controller, "w"))
            json.dump(config, f, indent=4)
            f.write("\n")  # json.dump doesn't write a final newline
        return 0
    else:
        return asyncio.run(issue_config(args.controller, args.port, args.name, config))


if __name__ == "__main__":
    sys.exit(main())
