#!/usr/bin/env python3

"""Start a simulated correlator via the SDP master controller.

This script constructs a JSON dictionary to describe a correlator simulation,
and submits it via katcp to an SDP master controller. Alternatively, the
configuration can be written to file to be manually started later.
"""

import argparse
import asyncio
import contextlib
import json
import sys
from dataclasses import dataclass
from typing import Optional, Sequence

import aiokatcp


@dataclass
class Band:
    """Holds presets for a known band."""

    adc_sample_rate: float
    centre_frequency: float


BANDS = {
    "l": Band(adc_sample_rate=1712e6, centre_frequency=1284e6),
    "u": Band(adc_sample_rate=1088e6, centre_frequency=816e6),
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse the command-line arguments (which may be specified as a parameter)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("controller", help="Hostname of the SDP master controller")
    parser.add_argument("--port", type=int, default=5001, help="TCP port of the SDP master controller [%(default)s]")
    parser.add_argument("--name", default="sim_correlator", help="Subarray product name [%(default)s]")
    parser.add_argument("-a", "--antennas", type=int, required=True, help="Number of antennas")
    parser.add_argument("-c", "--channels", type=int, required=True, help="Number of channels")
    parser.add_argument("-i", "--int-time", type=float, help="Integration time in seconds [%(default)s]")
    parser.add_argument(
        "--last-stage",
        choices=["d", "f", "x"],
        default="x",
        help="Do not run any stages past the given one [%(default)s]",
    )
    parser.add_argument("--band", default="l", choices=["l", "u"], help="Band ID [%(default)s]")
    parser.add_argument("--adc-sample-rate", type=float, help="ADC sample rate in Hz [from --band]")
    parser.add_argument("--centre-frequency", type=float, help="Sky centre frequency in Hz [from --band]")
    parser.add_argument("--image-tag", help="Docker image tag (for all images)")
    parser.add_argument("--katgpucbf-image", help="Full path to katgpucbf image to use (overrides --image-tag)")
    parser.add_argument(
        "-w", "--write", action="store_true", help="Write to file (give filename instead of the controller)"
    )
    args = parser.parse_args(argv)
    if args.adc_sample_rate is None:
        args.adc_sample_rate = BANDS[args.band].adc_sample_rate
    if args.centre_frequency is None:
        args.centre_frequency = BANDS[args.band].centre_frequency
    return args


def generate_config(args: argparse.Namespace) -> dict:
    """Produce the configuration dict from the parsed command-line arguments."""
    config: dict = {
        "version": "3.1",
        "config": {},
        "outputs": {},
    }
    if args.image_tag is not None:
        config["config"]["image_tag"] = args.image_tag
    if args.katgpucbf_image is not None:
        config["config"]["image_overrides"] = {"katgpucbf": args.katgpucbf_image}
    for ant_index in range(args.antennas):
        number = 800 + ant_index  # Avoid confusion with real antennas
        for pol in ["v", "h"]:
            name = f"m{number}{pol}"
            config["outputs"][name] = {
                "type": "sim.dig.raw_antenna_voltage",
                "band": args.band,
                "adc_sample_rate": args.adc_sample_rate,
                "centre_frequency": args.centre_frequency,
                "antenna": f"m{number}, 0:0:0, 0:0:0, 0, 0",
            }
    if args.last_stage == "d":
        return config

    config["outputs"]["antenna_channelised_voltage"] = {
        "type": "gpucbf.antenna_channelised_voltage",
        "src_streams": list(config["outputs"]),
        "n_chans": args.channels,
    }
    if args.last_stage == "f":
        return config

    config["outputs"]["baseline_correlation_products"] = {
        "type": "gpucbf.baseline_correlation_products",
        "src_streams": ["antenna_channelised_voltage"],
        "int_time": 0.5,
    }
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
    except (aiokatcp.FailReply, ConnectionError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    host = reply[1].decode()
    port = int(reply[2].decode())
    print(f"Product controller is at {host}:{port}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
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
