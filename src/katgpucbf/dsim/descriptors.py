import spead2
import asyncio
import numpy as np
from typing import Final

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
SEND_RATE_FACTOR = 1.05

# SPEAD flavour used for all send streams
FLAVOUR = spead2.Flavour(4, 64, 48, 0)

class descriptors():
    def __init__(self, args, timestamp, endpoints) -> None:
        self._running = True  # Set to false to start shutdown
        self.descriptor_rate = args.descriptor_rate
        self.timestamp = timestamp
        self.endpoints = endpoints
        self.dest_ip = endpoints[0][0]
        self.dest_port = endpoints[0][1]
        self.interface_ip = "10.100.44.1"

        # Create threadpool
        self.thread_pool = spead2.ThreadPool()

        self.rate = N_POLS * args.adc_sample_rate * SEND_RATE_FACTOR
        args.heap_samples
        self.config = spead2.send.StreamConfig(
        rate=self.rate,
        max_packet_size=args.heap_samples + PREAMBLE_SIZE,
        max_heaps=4,
        )
        
        # Setup Asyncio UDP stream
        stream: "spead2.send.asyncio.AsyncStream"

        self.stream = spead2.send.asyncio.UdpStream(
            self.thread_pool, [(self.dest_ip, self.dest_port)], self.config, ttl=args.ttl, interface_address=self.interface_ip
        )

        # Create item group
        self.item_group = spead2.send.ItemGroup(flavour=FLAVOUR)

        # Delete threadpool
        del self.thread_pool

    def halt(self) -> None:
        """Request :meth:`run` to stop, but do not wait for it."""
        self._running = False

    def create_descriptors(self, args):
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
        heap_payload = payload[0,0]

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
        return [descriptor_heap, self.stream]


    async def run(self, stream: "spead2.send.asyncio.AsyncStream", heap_to_send) -> None:
        while self._running:
            futures = []
            futures.append(stream.async_send_heap(heap_to_send, spead2.send.GroupMode.ROUND_ROBIN))
            await asyncio.gather(*futures)
            await asyncio.sleep(self.descriptor_rate)
