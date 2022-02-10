# import sys
# import re
import asyncio
from asyncio import create_task, gather
from typing import Final

import katsdpsigproc.accel as accel
import numpy as np
import spead2
import spead2.recv
import spead2.recv.asyncio
from scapy.all import *

# from .. import BYTE_BITS
# from ..recv import StatsToCounters
# from ..spead import TIMESTAMP_ID
# from . import METRIC_NAMESPACE
from katgpucbf.fgpu import SAMPLE_BITS, recv
from katgpucbf.monitor import NullMonitor
from katgpucbf.ringbuffer import ChunkRingbuffer

TOTAL_HEAPS: Final[int] = 20
N_POLS: Final = 2
MAX_CHUNKS = 2


def read_pcap_scapy(filename):

    p = rdpcap(filename)
    sessions = p.sessions()

    for session in sessions:
        for packet in sessions[session]:
            try:
                if packet[UDP].dport == 7148:
                    payload = bytes(packet[UDP].payload)
            except Exception as e:
                pass


async def read_pcap(
    recv_stream: spead2.recv.Stream,
) -> None:

    num_received = 0
    ig = spead2.ItemGroup()

    heap = recv_stream.get()
    items = ig.update(heap)

    while num_received < TOTAL_HEAPS:
        heap = recv_stream.get()
        items = ig.update(heap)
        # items = heap.get_items()

        for item in items.values():
            if item.id == 0x1600:
                has_timestamp = True

            # 5.2.2 Check that the received heap has a channel offset item
            # with the correct expected value.
            if item.id == 0x4103:
                has_channel_offset = True

            if item.id == 0x1800:
                has_xeng_raw = True

    a = 1


async def async_main(recv_stream) -> None:

    await gather(
        create_task(read_pcap(recv_stream)),
    )


def main() -> None:
    """Run main program."""

    # filename = "/home/avanderbyl/Git/katgpucbf/src/katgpucbf/dsim/output.pcap"
    filename = "/home/avanderbyl/Git/katgpucbf/src/katgpucbf/dsim/out.pcap"

    # No SPEAD unpack - Raw Data
    # read_pcap_scapy(filename)

    # Option 1:
    # ---------
    thread_pool = spead2.ThreadPool()
    stream_config = spead2.recv.StreamConfig(
        max_heaps=1,  # Digitiser heaps are single-packet, so no need for more
        memory_allocator=spead2.MemoryPool(16384, 26214400, 12, 8),
        stream_id=0,
    )
    recv_stream = spead2.recv.Stream(thread_pool, stream_config)
    del thread_pool

    # Or

    # Option 2:
    # ---------
    # monitor = NullMonitor()
    # channels = 4096
    # chunk_samples = 2**26
    # spectra_per_heap = 256
    # mask_timestamp = False
    # src_packet_samples = 1024 #???
    # pol = 0
    # src_affinity = [-1] * N_POLS

    # ringbuffer_capacity = 2
    # ring = ChunkRingbuffer(ringbuffer_capacity, name="recv_ringbuffer", task_name="run_receive", monitor=monitor)
    # chunk_samples = accel.roundup(chunk_samples, 2 * channels * spectra_per_heap)

    # _src_layout = recv.Layout(SAMPLE_BITS, src_packet_samples, chunk_samples, mask_timestamp)
    # recv_stream = recv.make_stream(
    #             pol,
    #             _src_layout,
    #             ring,
    #             src_affinity[pol],
    #         )

    # Or

    # Option 3:
    # ---------
    # active_frames = 3
    # n_xengs = 4
    # pol = 0
    # # n_chans_per_substream = self.corrVars.baseline_correlation_products_n_chans_per_substream
    # n_chans_per_substream = 16

    # # heap_data_size = (
    # #         np.dtype(np.complex64).itemsize * n_chans_per_substream *
    # #         self.corrVars.baseline_correlation_products_n_bls)

    # baseline_correlation_products_n_bls = 48
    # heap_data_size = (
    #         np.dtype(np.complex64).itemsize * n_chans_per_substream *
    #         baseline_correlation_products_n_bls)

    # # stream_xengs = ring_heaps = (
    # #         self.corrVars.baseline_correlation_products_n_chans // n_chans_per_substream)

    # baseline_correlation_products_n_chans = 48 * 4
    # stream_xengs = ring_heaps = (
    #         baseline_correlation_products_n_chans // n_chans_per_substream)

    # # max_heaps = stream_xengs + 2 * self.corrVars.n_xengs
    # max_heaps = stream_xengs + 2 * n_xengs
    # memory_pool_heaps = ring_heaps + max_heaps + stream_xengs * (active_frames + 5)
    # memory_pool = spead2.MemoryPool(2**14, heap_data_size + 2**9,
    #                                     memory_pool_heaps, memory_pool_heaps)
    # thread_pool = spead2.ThreadPool()
    # stream_config = spead2.recv.StreamConfig(
    #     max_heaps=1,  # Digitiser heaps are single-packet, so no need for more,
    #     memory_allocator=memory_pool,
    #     memcpy=spead2.MEMCPY_NONTEMPORAL,
    #     stream_id=pol,
    # )
    # # recv_stream = spead2.recv.Stream(thread_pool, max_heaps=max_heaps, ring_heaps=ring_heaps)
    # recv_stream = spead2.recv.Stream(thread_pool, stream_config)
    # del thread_pool

    # # recv_stream.set_memory_allocator(memory_pool)
    # # recv_stream.set_memcpy(spead2.MEMCPY_NONTEMPORAL)

    # Then:
    recv_stream.add_udp_pcap_file_reader(filename=filename)
    asyncio.run(async_main(recv_stream))


if __name__ == "__main__":
    main()
