"""Digitiser simulator descriptor sender."""
import asyncio

import netifaces as ni
import numpy as np
import spead2
import spead2.recv
import spead2.send
import spead2.send.asyncio

from katgpucbf import N_POLS, PREAMBLE_SIZE, SEND_RATE_FACTOR, SPEAD_DESCRIPTOR_INTERVAL_S

from ..spead import DIGITISER_ID_ID, DIGITISER_STATUS_ID, FLAVOUR, RAW_DATA_ID, TIMESTAMP_ID


class Descriptors:
    """Digitiser descriptor sender.

    Parameters
    ----------
    args:
        Runtime arguments passed (or as set by default).
    timestamp: float
        Timestamp since start.
    endpoints: Tuple[str, int]
        IP address of multicast address to send descriptors (string). The port is stated as an integer.
    """

    def __init__(self, args, timestamp, endpoints) -> None:
        self._running = True  # Set to false to start shutdown
        self.descriptor_rate = SPEAD_DESCRIPTOR_INTERVAL_S
        self.timestamp = timestamp
        self.endpoints = endpoints
        self.dest_ip = endpoints[0][0]
        self.dest_port = endpoints[0][1]
        self.interface_ip = ni.ifaddresses(args.interface)[ni.AF_INET][0]["addr"]

        # Create threadpool
        self.thread_pool = spead2.ThreadPool()
        self.rate = N_POLS * args.adc_sample_rate * SEND_RATE_FACTOR

        self.config = spead2.send.StreamConfig(
            rate=self.rate,
            max_packet_size=args.heap_samples + PREAMBLE_SIZE,
            max_heaps=4,
        )

        # Setup Asyncio UDP stream
        stream: "spead2.send.asyncio.AsyncStream"

        self.stream = spead2.send.asyncio.UdpStream(
            self.thread_pool,
            [(self.dest_ip, self.dest_port)],
            self.config,
            ttl=args.ttl,
            interface_address=self.interface_ip,
        )

        # Create item group
        self.item_group = spead2.send.ItemGroup(flavour=FLAVOUR)

        # Delete threadpool
        del self.thread_pool

    def halt(self) -> None:
        """Request :meth:`run` to stop, but do not wait for it."""
        self._running = False

    def create_descriptors(self, args):
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
            shape=[],
            format=[("u", FLAVOUR.heap_address_bits)],
        )

        value = np.uint8(0)
        self.item_group.add_item(
            DIGITISER_ID_ID,
            "Dig ID",
            "DEng ID",
            shape=(value.shape),
            dtype=value.dtype,
            value=value,
        )

        self.item_group.add_item(
            DIGITISER_STATUS_ID,
            "Dig Status ID",
            "DEng Status ID",
            shape=(value.shape),
            dtype=value.dtype,
            value=value,
        )

        n = len([self.timestamp])
        heap_size = args.heap_samples
        payload = np.zeros((N_POLS, n, heap_size), np.uint8)
        heap_payload = payload[0, 0]

        self.item_group.add_item(
            RAW_DATA_ID,
            "Raw Data",
            "DSim Raw Data",
            shape=heap_payload.shape,
            dtype=heap_payload.dtype,
            value=heap_payload,
        )

        descriptor_heap = self.item_group.get_heap(descriptors="all", data="none")
        descriptor_heap.repeat_pointers = True
        return descriptor_heap

    async def run(self, heap_to_send) -> None:
        """Run digitiser descriptor sender.

        Parameters
        ----------
        heap_to_send:
            Descriptor heap as formed in create_descriptors method.
        """
        while self._running:
            futures = []
            futures.append(self.stream.async_send_heap(heap_to_send))
            await asyncio.gather(*futures)
            await asyncio.sleep(self.descriptor_rate)
