#!/usr/bin/env python3

################################################################################
# Copyright (c) 2025, National Research Foundation (SARAO)
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

"""Forcibly deconfigure a subarray product.

The configuration details are extracted from the same INI file used by pytest
for qualification testing.
"""

import argparse
import asyncio
import configparser

import aiokatcp


async def main():  # noqa: D103
    parser = argparse.ArgumentParser()
    parser.add_argument("inifile", help="pytest configuration file")
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.inifile)
    mc_host = config["pytest"]["master_controller_host"]
    mc_port = int(config["pytest"].get("master_controller_port", 5001))
    product_name = config["pytest"].get("product_name", "qualification_cbf")
    async with await aiokatcp.Client.connect(mc_host, mc_port) as client:
        reply, informs = await client.request("product-list")
        products = [aiokatcp.decode(str, inform.arguments[0]) for inform in informs]
        if products:
            print(f"Current products: {', '.join(products)}")
        else:
            print("No current products")
        if product_name in products:
            print(f"Destroying {product_name}")
            # True to force deconfiguration
            await client.request("product-deconfigure", product_name, True)


if __name__ == "__main__":
    asyncio.run(main())
