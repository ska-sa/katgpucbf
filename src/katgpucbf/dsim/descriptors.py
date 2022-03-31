"""Digitiser simulator descriptor sender."""
import asyncio

import numpy as np
import spead2
import spead2.recv
import spead2.send
import spead2.send.asyncio

from katgpucbf import BYTE_BITS, SPEAD_DESCRIPTOR_INTERVAL_S

from ..spead import (
    DIGITISER_ID_ID,
    DIGITISER_STATUS_ID,
    FLAVOUR,
    HEAP_SAMPLES,
    IMMEDIATE_FORMAT,
    MAX_PACKET_SIZE,
    RAW_DATA_ID,
    SAMPLE_BITS,
    TIMESTAMP_ID,
)


def create_descriptor_stream(endpoints, config, ttl, interface_address) -> spead2.send.asyncio.UdpStream:
    """Create stream for descriptors."""
    return spead2.send.asyncio.UdpStream(
        spead2.ThreadPool(),
        list(endpoints),
        config=config,
        ttl=ttl,
        interface_address=interface_address,
    )


def create_descriptors_heap() -> spead2.send.Heap:
    """Add descriptor items to item group."""
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
        RAW_DATA_ID,
        "raw_data",
        "Digitiser Raw ADC Sample Data",
        shape=(HEAP_SAMPLES * SAMPLE_BITS // BYTE_BITS,),
        dtype=np.uint8,
    )
    descriptor_heap = item_group.get_heap(descriptors="all", data="none")
    return descriptor_heap


def create_config() -> spead2.send.StreamConfig:
    """Config for stream creation."""
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
        Descriptor heap to sending.
    """

    def __init__(
        self,
        stream: "spead2.send.asyncio.AsyncStream",
        descriptor_heap: spead2.send.Heap,
    ) -> None:
        self.descriptor_rate = SPEAD_DESCRIPTOR_INTERVAL_S
        self.rate = 0  # rate will be determined by the descriptor_rate

        self.event = asyncio.Event()
        self.config = create_config()

        # Setup Asyncio UDP stream
        self.stream = stream
        self.heap_to_send = descriptor_heap

    def halt(self) -> None:
        """Request :meth:`run` to stop, but do not wait for it."""
        self.event.set()

    async def run(self) -> None:
        """Run digitiser descriptor sender."""
        while True:
            try:
                await self.stream.async_send_heap(self.heap_to_send)
                await asyncio.wait_for(self.event.wait(), timeout=self.descriptor_rate)
                break
            except asyncio.TimeoutError:
                pass
