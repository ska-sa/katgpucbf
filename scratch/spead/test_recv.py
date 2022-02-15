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
def _packbits(data: np.ndarray, bits: int) -> np.ndarray:  # pragma: nocover
    # Note: needs lots of explicit casting to np.uint64, as otherwise
    # numba seems to want to infer double precision.
    out = np.zeros(data.size * bits // BYTE_BITS, np.uint8)
    buf = np.uint64(0)
    buf_size = 0
    mask = (np.uint64(1) << bits) - np.uint64(1)
    out_pos = 0
    for v in data:
        buf = (buf << bits) | (np.uint64(v) & mask)
        buf_size += bits
        while buf_size >= BYTE_BITS:
            out[out_pos] = buf >> (buf_size - BYTE_BITS)
            out_pos += 1
            buf_size -= BYTE_BITS
    return out

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


# Quick pack - unpack test. This is doesn'yt affect the spead processing.
# test_data = np.ones(64, np.uint16)
random_data = np.zeros(64, np.int8)

rng = np.random.default_rng(seed=2021)
random_data = rng.uniform(
        np.iinfo(random_data.dtype).min,
        np.iinfo(random_data.dtype).max,
        random_data.shape,
    ).astype(random_data.dtype)

packed_data = _packbits(random_data,10)
unpacked_data = unpackbits(packed_data)
np.testing.assert_array_equal(random_data, unpacked_data)

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

ibv_config = spead2.recv.UdpIbvConfig(
    endpoints=src,
    interface_address=interface,
    buffer_size=buffer,
    comp_vector=comp_vector,
)
# stream.add_udp_ibv_reader(ibv_config)

stream.add_udp_reader(multicast_group="239.102.0.64", port=7148, interface_address=interface)

ig = spead2.ItemGroup()
num_heaps = 0
unpacked_data = []
for heap in stream:
    print("Got heap", heap.cnt)
    items = ig.update(heap)
    for item in items.values():
        print(heap.cnt, item.name, item.value)
        if item.name == "Raw Data":
            print('Captured Data')
            tmp = unpackbits(item.value)
            unpacked_data.append(tmp)
    num_heaps += 1
    plt.figure(1)
    plt.plot(unpacked_data)
    plt.show()
stream.stop()
print("Received", num_heaps, "heaps")
print(f"data length is{len(unpacked_data)}")