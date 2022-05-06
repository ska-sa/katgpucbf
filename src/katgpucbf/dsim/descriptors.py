################################################################################
# Copyright (c) 2021, National Research Foundation (SARAO)
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

"""Digitiser simulator descriptor sender."""
import asyncio
from typing import Iterable, Tuple

import spead2
import spead2.send.asyncio

from .. import DIG_HEAP_SAMPLES, SPEAD_DESCRIPTOR_INTERVAL_S
from ..spead import (
    ADC_SAMPLES_ID,
    DIGITISER_ID_ID,
    DIGITISER_STATUS_ID,
    FLAVOUR,
    IMMEDIATE_FORMAT,
    MAX_PACKET_SIZE,
    TIMESTAMP_ID,
)


def make_descriptor_stream(
    endpoints: Iterable[Tuple[str, int]], config: spead2.send.StreamConfig, ttl: int, interface_address: str
) -> spead2.send.asyncio.UdpStream:
    """Create a stream for sending descriptors alongside dsim data."""
    return spead2.send.asyncio.UdpStream(
        spead2.ThreadPool(),
        list(endpoints),
        config=config,
        ttl=ttl,
        interface_address=interface_address,
    )


def create_descriptors_heap() -> spead2.send.Heap:
    """Create a descriptor heap for output dsim data."""
    item_group = spead2.send.ItemGroup(flavour=FLAVOUR)

    # Add items to item group
    item_group.add_item(
        TIMESTAMP_ID,
        "timestamp",
        "Timestamp provided by the MeerKAT digitisers and scaled to the digitiser sampling rate.",
        shape=(),
        format=IMMEDIATE_FORMAT,
    )
    item_group.add_item(
        DIGITISER_ID_ID,
        "digitiser_id",
        "Digitiser Serial Number",
        shape=(),
        format=IMMEDIATE_FORMAT,
    )
    item_group.add_item(
        DIGITISER_STATUS_ID,
        "digitiser_status",
        "Digitiser Status values",
        shape=(),
        format=IMMEDIATE_FORMAT,
    )

    item_group.add_item(
        ADC_SAMPLES_ID,
        "adc_samples",
        "Digitiser Raw ADC Sample Data",
        shape=(DIG_HEAP_SAMPLES,),
        format=[("i", 10)],
    )
    descriptor_heap = item_group.get_heap(descriptors="all", data="none")
    return descriptor_heap


def create_config() -> spead2.send.StreamConfig:
    """Create simple configuration for descriptor stream."""
    return spead2.send.StreamConfig(
        rate=0,
        max_packet_size=MAX_PACKET_SIZE,
        max_heaps=4,
    )


class DescriptorSender:
    """Digitiser descriptor sender.

    Parameters
    ----------
    stream:
        Stream created for descriptors.
    descriptor_heap:
        Descriptor heap for sending.
    """

    def __init__(
        self,
        stream: "spead2.send.asyncio.AsyncStream",
        descriptor_heap: spead2.send.Heap,
    ) -> None:

        self._halt_event = asyncio.Event()

        self.stream = stream
        self.heap_to_send = descriptor_heap

    def halt(self) -> None:
        """Request :meth:`run` to stop, but do not wait for it."""
        self._halt_event.set()

    async def run(self) -> None:
        """Run digitiser descriptor sender."""
        while True:
            try:
                await self.stream.async_send_heap(self.heap_to_send)
                # Wait for event (if a halt requested). wait_for will time out if not received
                # within descriptor_interval period and descriptor will be resent.
                await asyncio.wait_for(self._halt_event.wait(), timeout=SPEAD_DESCRIPTOR_INTERVAL_S)
                break
            except asyncio.TimeoutError:
                pass
