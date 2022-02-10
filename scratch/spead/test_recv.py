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

import ipaddress
import logging
from typing import Callable, List, Optional, Sequence, Tuple, TypeVar, Union

# import spead2
import spead2
import spead2.recv
from katsdptelstate.endpoint import endpoint_list_parser


def parse_source(value: str) -> Union[List[Tuple[str, int]], str]:
    """Parse a string into a list of IP endpoints."""
    try:
        endpoints = endpoint_list_parser(7148)(value)
        for endpoint in endpoints:
            ipaddress.IPv4Address(endpoint.host)  # Raises if invalid syntax
        return [(ep.host, ep.port) for ep in endpoints]
    except ValueError:
        return


logging.basicConfig(level=logging.INFO)

interface = "10.100.44.1"
buffer = 32 * 1024 * 1024
comp_vector = -1
src = "239.102.0.64:7148"

src = parse_source(src)

thread_pool = spead2.ThreadPool()

stream = spead2.recv.Stream(
    thread_pool, spead2.recv.StreamConfig(memory_allocator=spead2.MemoryPool(16384, 26214400, 12, 8))
)
del thread_pool
# if 0:
#     with open('junkspeadfile', 'rb') as f:
#         text = f.read()
#     stream.add_buffer_reader(text)
# else:
#     stream.add_udp_reader(8888)

# ibv_config = spead2.recv.UdpIbvConfig(
#     endpoints=src,
#     interface_address=interface,
#     buffer_size=buffer,
#     comp_vector=comp_vector,
# )
# stream.add_udp_ibv_reader(ibv_config)

stream.add_udp_reader(multicast_group="239.102.0.64", port=7148, interface_address=interface)

ig = spead2.ItemGroup()
num_heaps = 0
for heap in stream:
    print("Got heap", heap.cnt)
    items = ig.update(heap)
    # print(len(items.values))
    for item in items.values():
        print("heap")
        # print(heap.cnt, item.name, item.value)
    num_heaps += 1
stream.stop()
print("Received", num_heaps, "heaps")
