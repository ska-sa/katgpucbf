"""Digitiser simulator descriptor sender."""
import asyncio

import netifaces
import numpy as np
import spead2
import spead2.recv
import spead2.send
import spead2.send.asyncio

from katgpucbf import SPEAD_DESCRIPTOR_INTERVAL_S

from ..spead import (
    DIGITISER_ID_ID,
    DIGITISER_STATUS_ID,
    FLAVOUR,
    IMMEDIATE_FORMAT,
    MAX_PACKET_SIZE,
    RAW_DATA_ID,
    TIMESTAMP_ID,
)


class DescriptorSender:
    """Digitiser descriptor sender.

    Parameters
    ----------
    args:
        Runtime arguments passed (or as set by default).
    timestamp: float
        Timestamp since start.
    endpoints:
        IP address of multicast address to send descriptors (string). The port is stated as an integer.
    """

    def __init__(self, args, timestamp: float, endpoints: tuple) -> None:
        self._running = True  # Set to false to start shutdown
        self.descriptor_rate = SPEAD_DESCRIPTOR_INTERVAL_S
        self.timestamp = timestamp
        self.endpoints = endpoints
        self.interface_address = netifaces.ifaddresses(args.interface)[netifaces.AF_INET][0]["addr"]

        self.rate = 0  # rate will be determined by the descriptor_rate
  
        self.event = asyncio.Event()

        self.config = spead2.send.StreamConfig(
            rate=self.rate,
            max_packet_size=MAX_PACKET_SIZE,
            max_heaps=4,
        )

        # Setup Asyncio UDP stream
        self.stream = spead2.send.asyncio.UdpStream(
            spead2.ThreadPool(),
            endpoints,
            self.config,
            ttl=args.ttl,
            interface_address=self.interface_address,
        )

        # Create item group
        self.item_group = spead2.send.ItemGroup(flavour=FLAVOUR)
        self.heap_to_send = self.__create_descriptors_heap(args.heap_samples, args.sample_bits)

    def halt(self) -> None:
        """Request :meth:`run` to stop, but do not wait for it."""
        # self._running = False
        self.event.set()

    def __create_descriptors_heap(self, heap_samples, sample_bits) -> spead2.send.Heap:
        """Add descriptor items to item group.

        Parameters
        ----------
        args:
            Runtime arguments passed (or as set by default).
        """
        # Add items to item group
        self.item_group.add_item(
            TIMESTAMP_ID,
            "timestamp",
            "Timestamp provided by the MeerKAT digitisers and scaled to the digitiser sampling rate.",
            shape=(),
            format=IMMEDIATE_FORMAT,
        )

        self.item_group.add_item(
            DIGITISER_ID_ID,
            "Digitiser ID",
            "Digitiser Serial Number",
            shape=(),
            dtype=np.uint8,
        )

        self.item_group.add_item(
            DIGITISER_STATUS_ID,
            "Digitiser Status",
            "Digitiser Status values",
            shape=(),
            dtype=np.uint8,
        )

        self.item_group.add_item(
            RAW_DATA_ID,
            "ADC Samples",
            "Digitiser Raw ADC Sample Data",
            shape=(heap_samples,),
            format=[("i", 10)],
        )

        descriptor_heap = self.item_group.get_heap(descriptors="all", data="none")
        return descriptor_heap

    # async def run(self, heap_to_send: spead2.send.Heap) -> None:
    async def run(self) -> None:
        """Run digitiser descriptor sender.

        Parameters
        ----------
        heap_to_send
            Descriptor heap as formed in create_descriptors method.
        """
        # while self._running:
        #     # self.stream.set_cnt_sequence()
        #     await self.stream.async_send_heap(heap_to_send)
        #     await asyncio.sleep(self.descriptor_rate)

        while True:
            try:
                await self.stream.async_send_heap(self.heap_to_send)
                await asyncio.wait_for(self.event.wait(), timeout=self.descriptor_rate)
                break
            except asyncio.TimeoutError:
                pass
