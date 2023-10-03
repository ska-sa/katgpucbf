################################################################################
# Copyright (c) 2023, National Research Foundation (SARAO)
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

"""Module for sending tied array channelised voltage products onto the network."""

import asyncio
from typing import Callable, Final, Sequence

import katsdpsigproc.accel as accel
import numpy as np
import spead2
import spead2.send.asyncio
from katsdpsigproc.abc import AbstractContext
from katsdptelstate.endpoint import Endpoint

from .. import COMPLEX, DEFAULT_PACKET_PAYLOAD_BYTES
from ..spead import BF_RAW_ID, FLAVOUR, FREQUENCY_ID, IMMEDIATE_FORMAT, TIMESTAMP_ID
from .output import BOutput

# NOTE: ICD suggests `beng_out_bits_per_sample`,
# MK correlator doesn't make this configurable.
SEND_DTYPE = np.dtype(np.int8)


class Heap:
    """Hold all data for a heap.

    # TODO: This will likely need to be updated to the 'fgpu.send.Chunk' methodology
            - Especially to dictate 'Frames' within the Chunk
    """

    def __init__(
        self,
        context: AbstractContext,
        n_channels_per_substream: int,
        n_spectra_per_heap: int,
    ) -> None:
        self.buffer: Final = accel.HostArray(
            (n_channels_per_substream, n_spectra_per_heap, COMPLEX), SEND_DTYPE, context=context
        )


class BSend:
    """
    Class for turning tied array channelised voltage products into SPEAD heaps.

    NOTE: Each BSend shouldn't need its own *version* of a descriptor heap
    """

    descriptor_heap: spead2.send.Heap
    header_size: Final[int] = 64

    def __init__(
        self,
        output: BOutput,
        n_channels_per_substream: int,
        spectra_per_heap: int,
        send_rate_factor: float,
        channel_offset: int,
        context: AbstractContext,
        stream_factory: Callable[[spead2.send.StreamConfig, Sequence[np.ndarray]], "spead2.send.asyncio.AsyncStream"],
        packet_payload: int = DEFAULT_PACKET_PAYLOAD_BYTES,
        tx_enabled: bool = False,
    ) -> None:
        # Now that we've moved away from *multiple* BSend objects, towards
        # a single send_stream with multiple substreams, the BSend object
        # probably still needs to exist
        # - The `send` function will need to keep track of "which substream
        #   is enabled/disabled" Re: capture-{start, stop} <stream_name>
        self.enabled_stream_ids: list[int] = []

        self._heaps_queue: asyncio.Queue[Heap] = asyncio.Queue()
        buffers: list[accel.HostArray] = []

        # `n_heaps_to_send` is actually used to dictate the amount of buffers (in XSend)
        # So perhaps we need to change the number of buffers to be range(send_free_queue.maxsize)
        n_heaps_to_send = len(buffers) // spectra_per_heap

        for _ in range(n_heaps_to_send):
            heap = Heap(context, n_channels_per_substream, spectra_per_heap)
            self._heaps_queue.put_nowait(heap)
            buffers.append(heap.buffer)

        # Multicast stream parameters
        self.heap_payload_size_bytes = n_channels_per_substream * spectra_per_heap * COMPLEX * SEND_DTYPE.itemsize
        # Transport-agnostic stream information
        # Used in XSend to calculate `send_rate_bytes_per_second`, do we need it here?
        # packets_per_heap = math.ceil(self.heap_payload_size_bytes / packet_payload)
        # packet_header_overhead_bytes = packets_per_heap * BSend.header_size

        stream_config = spead2.send.StreamConfig(
            max_packet_size=packet_payload + BSend.header_size,
            max_heaps=10,  # TODO: Update this to be proper
            rate_method=spead2.send.RateMethod.AUTO,
            rate=0,  # TODO: Update to use `send_rate_bytes_per_second`, this sends as fast as possible
        )
        self.source_stream = stream_factory(stream_config, buffers)

    async def send_heap(self, heap: Heap) -> None:
        """Take in a buffer and send it as a SPEAD heap."""
        pass

    def enable_substream(self, stream_id: int, enable: bool = True) -> None:
        """Enable/Disable a substream's data transmission.

        :class:`.BSend` operates as a large single stream with multiple
        substreams. Each substream is its own data product and is required
        to be enabled/disabled independently.

        Parameters
        ----------
        stream_id
            ID of the substream, corresponds to the <beam-id><pol>
            convention, e.g. stream_id 3 has a stream_name ending in <1x>.
        enable
            Boolean indicating whether the `stream_id` should be enabled or
            disabled.
        """
        pass


def make_stream(
    *,
    endpoints: list[Endpoint],
    interface: str,
    ttl: int,
    use_ibv: bool,
    affinity: int,
    comp_vector: int,
    stream_config: spead2.send.StreamConfig,
    buffers: Sequence[np.ndarray],
) -> "spead2.send.asyncio.AsyncStream":
    """Create asynchronous SPEAD stream for transmission.

    This is architected to be a single send stream with multiple substreams,
    each corresponding to a tied-array-channelised-voltage output data product.
    The `endpoints` need not be a contiguous list of multicast addresses.
    """
    stream: spead2.send.asyncio.AsyncStream
    thread_pool = spead2.ThreadPool(1, [] if affinity < 0 else [affinity])

    if use_ibv:
        stream = spead2.send.asyncio.UdpIbvStream(
            thread_pool,
            stream_config,
            spead2.send.UdpIbvConfig(
                endpoints=[(ep.host, ep.port) for ep in endpoints],
                interface_address=interface,
                ttl=ttl,
                comp_vector=comp_vector,
                memory_regions=list(buffers),
            ),
        )
    else:
        stream = spead2.send.asyncio.UdpStream(
            thread_pool,
            [(ep.host, ep.port) for ep in endpoints],
            stream_config,
            interface_address=interface,
            ttl=ttl,
        )

    return stream


def make_descriptor_heap(
    *,
    n_channels_per_substream: int,
    spectra_per_heap: int,
) -> "spead2.send.Heap":
    """Create a descriptor heap for output B-engine data."""
    heap_data_shape = (n_channels_per_substream, spectra_per_heap, COMPLEX)

    ig = spead2.send.ItemGroup(flavour=FLAVOUR)
    ig.add_item(
        FREQUENCY_ID,
        "frequency",  # Misleading name, but it's what the ICD specifies
        "Value of the first channel in collections stored here.",
        shape=[],
        format=IMMEDIATE_FORMAT,
    )
    ig.add_item(
        TIMESTAMP_ID,
        "timestamp",
        "Timestamp provided by the MeerKAT digitisers and scaled to the digitiser sampling rate.",
        shape=[],
        format=IMMEDIATE_FORMAT,
    )
    ig.add_item(
        BF_RAW_ID,
        "bf_raw",
        "",  # TODO: What to even say here? ICD says "Channelised complex data"
        shape=heap_data_shape,
        dtype=SEND_DTYPE,
    )

    return ig.get_heap(descriptors="all", data="none")
