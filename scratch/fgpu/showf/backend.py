import argparse
import asyncio
import copy
import functools
from collections import deque

import bokeh.server.contexts
import numpy as np
import spead2.recv.asyncio
from katsdpservices import get_interface_address
from katsdptelstate.endpoint import endpoint_list_parser

TIMESTAMP_ID = 0x1600
FREQUENCY_ID = 0x4103
FENG_RAW_ID = 0x4300


def parse_args(args: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--channels", "-c", type=int, required=True)
    parser.add_argument("--substreams", "-s", type=int)
    parser.add_argument("--acc-len", "-a", type=int, default=256)
    parser.add_argument("--keep-ratio", "-k", type=int, default=64)
    parser.add_argument("--interface", "-i", required=True)
    parser.add_argument("address")
    return parser.parse_args(args)


class Batch:
    def __init__(self, timestamp, channels: int, acc_len: int) -> None:
        self.timestamp = timestamp
        self.data = np.empty((channels, acc_len, 2, 2), np.int8)
        self.present = np.zeros(channels, np.bool_)


class DisplayBatch:
    def __init__(self, batch: Batch) -> None:
        cplex = batch.data.astype(np.float32).view(np.complex64)[..., 0]
        self.mag = np.abs(cplex)
        self.phase = np.where(cplex != 0, np.angle(cplex), np.nan)


class Backend:
    def __init__(
        self,
        address: str,
        interface: str,
        channels: int,
        substreams: int | None,
        acc_len: int,
        keep_ratio: int,
        server_context: bokeh.server.contexts.BokehServerContext,
    ) -> None:
        endpoints = endpoint_list_parser(7148)(address)
        endpoint_tuples = [(ep.host, ep.port) for ep in endpoints]
        if substreams is None:
            substreams = len(endpoints)
        channels_per_substream = channels // substreams
        self.channels = channels
        self.channels_per_substream = channels_per_substream
        self.acc_len = acc_len
        self.keep_step = 2 * channels * acc_len * keep_ratio
        heap_size = channels_per_substream * acc_len * 4 * np.dtype(np.int8).itemsize
        pool_items = 36 * substreams + 2  # TODO: total thumb suck
        pool = spead2.MemoryPool(heap_size, heap_size, pool_items, pool_items)
        self.stream = spead2.recv.asyncio.Stream(
            spead2.ThreadPool(),
            spead2.recv.StreamConfig(max_heaps=2 * substreams, memory_allocator=pool),
            spead2.recv.RingStreamConfig(heaps=32 * substreams),
        )
        self.stream.add_udp_ibv_reader(endpoint_tuples, get_interface_address(interface), buffer_size=64 * 1024 * 1024)
        self.batches: deque[Batch] = deque()
        self.last_full_timestamp = -1
        self.server_context = server_context

    def _get_batch(self, timestamp: int) -> Batch | None:
        if timestamp % self.keep_step != 0:
            return None
        if not self.batches or timestamp > self.batches[-1].timestamp:
            self.batches.append(Batch(timestamp, self.channels, self.acc_len))
            if len(self.batches) > 2:
                self.batches.popleft()
            return self.batches[-1]
        else:
            for batch in self.batches:
                if batch.timestamp == timestamp:
                    return batch
        return None

    def _update_document(self, doc: bokeh.document.document.Document, display: DisplayBatch) -> None:
        for pol in range(2):
            source = doc.get_model_by_name(f"source{pol}")
            new_data = copy.copy(source.data)
            new_data["mag"] = [display.mag[..., pol].T]
            new_data["phase"] = [display.phase[..., pol].T]
            source.data = new_data

    async def _update_sessions(self, batch: Batch) -> None:
        display = await asyncio.get_event_loop().run_in_executor(None, DisplayBatch, batch)
        for sctx in self.server_context.sessions:
            sctx.with_document_locked(self._update_document, sctx.document, display)

    async def run(self) -> None:
        try:
            heap_shape = (self.channels_per_substream, self.acc_len, 2, 2)
            async for heap in self.stream:
                timestamp = None
                frequency = None
                raw_data = None
                for item in heap.get_items():
                    if item.is_immediate:
                        if item.id == TIMESTAMP_ID and item.is_immediate:
                            timestamp = int(item.immediate_value)
                        elif item.id == FREQUENCY_ID:
                            frequency = int(item.immediate_value)
                    else:
                        if item.id == FENG_RAW_ID:
                            data = np.array(item, np.int8, copy=False).reshape(heap_shape)
                if timestamp is not None and frequency is not None:
                    batch = self._get_batch(timestamp)
                    if batch is not None:
                        end = frequency + self.channels_per_substream
                        batch.data[frequency:end] = data
                        batch.present[frequency:end] = True
                        if batch.timestamp > self.last_full_timestamp and np.all(batch.present):
                            self._last_full_timestamp = batch.timestamp
                            self.server_context.add_next_tick_callback(functools.partial(self._update_sessions, batch))

                # Allow them to be reclaimed before popping next heap
                del heap
                del item
                del raw_data
        except Exception as exc:
            print("Error in run", exc)
            raise
