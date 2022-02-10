#!/usr/bin/env python3
import argparse
import ast
import asyncio
import ipaddress
import logging
from typing import Callable, List, Optional, Sequence, Tuple, TypeVar, Union

import aiokatcp
import matplotlib
import matplotlib.pyplot as plt
import numba
import numpy as np
import scipy
import spead2
import spead2.recv
import spead2.recv.asyncio
from katsdpservices import get_interface_address
from katsdptelstate.endpoint import Endpoint, endpoint_list_parser, endpoint_parser
from numba import types
from spead2.numba import intp_to_voidptr
from spead2.recv.numba import chunk_place_data


async def get_product_controller_endpoint(mc_endpoint: Endpoint, product_name: str) -> Endpoint:
    """Get the katcp address for a named product controller from the master."""
    client = await aiokatcp.Client.connect(*mc_endpoint)
    async with client:
        return endpoint_parser(None)(await get_sensor_val(client, f"{product_name}.katcp-address"))


def parse_source(value: str) -> Union[List[Tuple[str, int]], str]:
    """Parse a string into a list of IP endpoints."""
    try:
        endpoints = endpoint_list_parser(7148)(value)
        for endpoint in endpoints:
            ipaddress.IPv4Address(endpoint.host)  # Raises if invalid syntax
        return [(ep.host, ep.port) for ep in endpoints]
    except ValueError:
        return


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mc-address",
        type=endpoint_parser(5001),
        default="lab5.sdp.kat.ac.za:5001",  # Naturally this applies only to our lab...
        help="Master controller to query for details about the product. [%(default)s]",
    )

    args = parser.parse_args()
    print(args)
    asyncio.run(async_main(args))


async def async_main(args: argparse.Namespace) -> None:
    logger = logging.getLogger(__name__)
    product_name = "baseline0"

    mc_address = "239.102.0.64:7148"
    src = parse_source(mc_address)
    host = src[0][0]
    port = src[0][1]

    # host, port = await get_product_controller_endpoint(args.mc_address, product_name)
    client = await aiokatcp.Client.connect(host, port)
    a = 1


if __name__ == "__main__":
    main()
