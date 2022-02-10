#!/usr/bin/env python3

# Copyright 2015, 2020 National Research Foundation (SARAO)
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import asyncio
import functools
import ipaddress
import logging
from typing import Callable, Final, List, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np
import spead2
import spead2.recv
import spead2.send
import spead2.send.asyncio
from katsdptelstate.endpoint import endpoint_list_parser

DIGITISER_ID_ID = 0x3101
DIGITISER_STATUS_ID = 0x3102
FENG_ID_ID = 0x4101
FENG_RAW_ID = 0x4300
FREQUENCY_ID = 0x4103
RAW_DATA_ID = 0x3300  # Digitiser data
XENG_RAW_ID = 0x1800
TIMESTAMP_ID = 0x1600
COMPLEX: Final = 2
N_POLS: Final = 2
PREAMBLE_SIZE = 72

#: SPEAD flavour used for all send streams
FLAVOUR = spead2.Flavour(4, 64, 48, 0)


def parse_source(value: str) -> Union[List[Tuple[str, int]], str]:
    """Parse a string into a list of IP endpoints."""
    try:
        endpoints = endpoint_list_parser(7148)(value)
        for endpoint in endpoints:
            ipaddress.IPv4Address(endpoint.host)  # Raises if invalid syntax
        return [(ep.host, ep.port) for ep in endpoints]
    except ValueError:
        return


# async def send(self, stream: "spead2.send.asyncio.AsyncStream", frames: int) -> None:
async def send(stream: "spead2.send.asyncio.AsyncStream", heap_to_send) -> None:
    futures = []
    futures.append(stream.async_send_heap(heap_to_send, spead2.send.GroupMode.ROUND_ROBIN))
    await asyncio.gather(*futures)


async def async_main() -> None:
    logging.basicConfig(level=logging.INFO)

    interface_ip = "10.100.44.1"
    buffer = 32 * 1024 * 1024
    comp_vector = -1
    src = "239.102.0.64:7148"
    # src = "127.0.0.1:7148"
    ttl = 1

    dest_ip_port = parse_source(src)

    # Create threadpool
    thread_pool = spead2.ThreadPool()

    adc_sample_rate = 1712e6
    send_rate_factor = 1.05
    packet_payload = 1024

    rate = N_POLS * adc_sample_rate * send_rate_factor
    config = spead2.send.StreamConfig(
        rate=rate,
        max_packet_size=packet_payload + PREAMBLE_SIZE,
        max_heaps=4,
    )

    stream: "spead2.send.asyncio.AsyncStream"

    stream = spead2.send.asyncio.UdpStream(
        thread_pool, [(dest_ip_port[0][0], dest_ip_port[0][1])], config, ttl=ttl, interface_address=interface_ip
    )

    # Create item group
    item_group = spead2.send.ItemGroup(flavour=FLAVOUR)

    # Add items to item group
    item_group.add_item(
        FREQUENCY_ID,
        "frequency",  # Misleading name, but it's what the ICD specifies
        "Value of first channel in collections stored here.",
        shape=[],
        format=[("u", FLAVOUR.heap_address_bits)],
    )

    item_group.add_item(
        TIMESTAMP_ID,
        "timestamp",
        "Timestamp provided by the MeerKAT digitisers and scaled to the digitiser sampling rate.",
        shape=[],
        format=[("u", FLAVOUR.heap_address_bits)],
    )

    n_ants = 4
    n_channels_per_stream = 16
    n_baselines = (n_ants + 1) * (n_ants) * 2
    heap_shape = (n_channels_per_stream, 1)
    item_group.add_item(
        RAW_DATA_ID,
        "deng_raw",
        "RAW DEng data.",
        shape=heap_shape,
        dtype=np.int32,
    )

    descriptor_heap = item_group.get_heap(descriptors="all", data="none")

    item_group["timestamp"].value = 100
    item_group["frequency"].value = 1000
    item_group["deng_raw"].value = np.ones(heap_shape)

    heap_to_send = item_group.get_heap(descriptors="none", data="all")
    heap_to_send.repeat_pointers = True

    del thread_pool

    # Send only descriptors
    await send(stream, descriptor_heap)

    # Send only data
    await send(stream, heap_to_send)


if __name__ == "__main__":
    asyncio.run(async_main())
