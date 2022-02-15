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
from tkinter.messagebox import NO
from typing import Callable, List, Optional, Sequence, Tuple, TypeVar, Union
from typing import Final
from numpy import uint16
import numpy as np
import numba

# import spead2
import spead2
import spead2.recv
from katsdptelstate.endpoint import endpoint_list_parser

import matplotlib.pyplot as plt
BYTE_BITS: Final = 8

def parse_source(value: str) -> Union[List[Tuple[str, int]], str]:
    """Parse a string into a list of IP endpoints."""
    try:
        endpoints = endpoint_list_parser(7148)(value)
        for endpoint in endpoints:
            ipaddress.IPv4Address(endpoint.host)  # Raises if invalid syntax
        return [(ep.host, ep.port) for ep in endpoints]
    except ValueError:
        return

@numba.njit
def unpackbits(packed_data):
    unpacked_data = []
    data_sample = np.int16(0)
    idx = 0

    for _ in range(len(packed_data)//5):
        tmp_40b_word = np.uint64(
            packed_data[idx] << (8*4) | 
            packed_data[idx+1] << (8*3)|
            packed_data[idx+2] << (8*2) |
            packed_data[idx+3] << 8 |
            packed_data[idx+4]
            )
        for _ in range(4):
            data_sample = (tmp_40b_word & 1098437885952) >> 30
            if data_sample > 511:
                data_sample = data_sample - 1024
            unpacked_data.append(np.int16(data_sample))
            tmp_40b_word = tmp_40b_word << 10

        idx += 5

    return unpacked_data

logging.basicConfig(level=logging.INFO)

interface = "10.100.44.1"
buffer = 32 * 1024 * 1024
comp_vector = -1
src = "239.103.0.64:7148"

src = parse_source(src)

thread_pool = spead2.ThreadPool()

stream = spead2.recv.Stream(
    thread_pool, spead2.recv.StreamConfig(memory_allocator=spead2.MemoryPool(16384, 26214400, 12, 8))
)
del thread_pool

ibv_config = spead2.recv.UdpIbvConfig(
    endpoints=src,
    interface_address=interface,
    buffer_size=buffer,
    comp_vector=comp_vector,
)
# stream.add_udp_ibv_reader(ibv_config)

stream.add_udp_reader(multicast_group="239.103.0.64", port=7148, interface_address=interface)

ig = spead2.ItemGroup()
num_heaps = 0
unpacked_data = None
for heap in stream:
    print("Got heap", heap.cnt)
    items = ig.update(heap)
    for item in items.values():
        print(heap.cnt, item.name, item.value)
        if item.name == "Raw Data":
            print('Captured Data')
            tmp = unpackbits(item.value)
            if unpacked_data is None:
                unpacked_data = tmp
            else:
                unpacked_data = np.concatenate([unpacked_data, tmp])
            # unpacked_data.append(tmp)

    num_heaps += 1
    if unpacked_data is not None:
       if len(unpacked_data) >= 20000:
           print(f'Heap:{num_heaps}')
           break

stream.stop()

if len(unpacked_data) > 0:
    # Plot unpacked data.
    plt.figure(1)
    plt.plot(unpacked_data)
    plt.show()

print("Received", num_heaps, "heaps")
print(f"data length is: {len(unpacked_data)}")