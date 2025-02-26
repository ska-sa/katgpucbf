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

"""Utilities for writing development tools that start CBFs."""

import argparse


def parse_develop_options(develop_argument: str) -> dict:
    """Separate complex develop options into a dictionary."""
    out_dict = {}
    for item in develop_argument.split(","):
        # data_timeout option isn't boolean, unlike the rest
        if "data_timeout" in item:
            k, v = item.split("=")
            out_dict[k] = float(v)
        else:
            out_dict[item] = True
    return out_dict


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add common options to an argument parser."""
    parser.add_argument("--image-tag", help="Docker image tag (for all images)")
    parser.add_argument("--image-override", action="append", metavar="NAME:IMAGE:TAG", help="Override a single image")
    parser.add_argument(
        "--develop",
        nargs="?",
        const=True,
        help="Pass development options in the config. Use comma separation, or omit the arg to enable all.",
    )


def apply_arguments(config: dict, args: argparse.Namespace) -> None:
    """Apply arguments from :func:`add_arguments` to a product-configure dictionary."""
    config.setdefault("config", {})
    if args.image_tag is not None:
        config["config"]["image_tag"] = args.image_tag
    if args.image_override is not None:
        image_overrides = {}
        for override in args.image_override:
            name, image = override.split(":", 1)
            image_overrides[name] = image
        config["config"]["image_overrides"] = image_overrides
    if args.develop is not None:
        if args.develop is True or args.develop == "":
            # User passed --develop with no argument or --develop= with empty argument
            config["config"]["develop"] = True
        else:
            # User passed a comma-separated list of options
            config["config"]["develop"] = parse_develop_options(args.develop)
