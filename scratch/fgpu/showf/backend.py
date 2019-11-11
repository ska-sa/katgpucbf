import argparse
import asyncio
import copy
from collections import deque
import functools
from typing import List, Deque, Optional

import numpy as np
import bokeh.server.contexts
import tornado.gen

from katsdptelstate.endpoint import Endpoint, endpoint_list_parser
from katsdpservices import get_interface_address
import spead2.recv.asyncio


TIMESTAMP_ID = 0x1600
FREQUENCY_ID = 0x4103
FENG_RAW_ID = 0x4300


def parse_args(args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--channels', '-c', type=int, required=True)
    parser.add_argument('--substreams', '-s', type=int)
    parser.add_argument('--acc-len', '-a', type=int, default=256)
    parser.add_argument('--keep-ratio', '-k', type=int, default=64)
    parser.add_argument('--interface', '-i', required=True)
    parser.add_argument('address')
    return parser.parse_args(args)


class Frame:
    def __init__(self, timestamp, channels: int, acc_len: int) -> None:
        self.timestamp = timestamp
        self.data = np.empty((channels, acc_len, 2, 2), np.int8)
        self.present = np.zeros(channels, np.bool_)


class DisplayFrame:
    def __init__(self, frame: Frame) -> None:
        cplex = frame.data.astype(np.float32).view(np.complex64)[..., 0]
        self.mag = np.abs(cplex)
        self.phase = np.where(cplex != 0, np.angle(cplex), np.nan)


class Backend:
    def __init__(self, address: str, interface: str,
                 channels: int, substreams: Optional[int], acc_len: int,
                 keep_ratio: int,
                 server_context: bokeh.server.contexts.BokehServerContext) -> None:
        endpoints = endpoint_list_parser(7148)(address)
        endpoint_tuples = [(ep.host, ep.port) for ep in endpoints]
        if substreams is None:
            substreams = len(endpoints)
        channels_per_substream = channels // substreams
        self.channels = channels
        self.channels_per_substream = channels_per_substream
        self.acc_len = acc_len
        self.keep_step = 2 * channels * acc_len * keep_ratio
        self.stream = spead2.recv.asyncio.Stream(spead2.ThreadPool(),
                                                 max_heaps=2 * substreams,
                                                 ring_heaps=16 * substreams)
        heap_size = channels_per_substream * acc_len * 4 * np.dtype(np.int8).itemsize
        pool_items = 19 * substreams + 2  # TODO: total thumb suck
        pool = spead2.MemoryPool(heap_size, heap_size, pool_items, pool_items)
        self.stream.set_memory_allocator(pool)
        self.stream.add_udp_ibv_reader(endpoint_tuples,
                                       get_interface_address(interface),
                                       buffer_size=64 * 1024 * 1024)
        self.frames: Deque[Frame] = deque()
        self.last_full_timestamp = -1
        self.server_context = server_context

    def _get_frame(self, timestamp: int) -> Optional[Frame]:
        if timestamp % self.keep_step != 0:
            return None
        if not self.frames or timestamp > self.frames[-1].timestamp:
            self.frames.append(Frame(timestamp, self.channels, self.acc_len))
            if len(self.frames) > 2:
                self.frames.popleft()
            return self.frames[-1]
        else:
            for frame in self.frames:
                if frame.timestamp == timestamp:
                    return frame
        return None

    def _update_document(self, doc: bokeh.document.document.Document, display: DisplayFrame) -> None:
        source = doc.get_model_by_name('source')
        new_data = copy.copy(source.data)
        new_data['mag'] = [display.mag[..., 0].T]
        new_data['phase'] = [display.phase[..., 0].T]
        source.data = new_data

    async def _update_sessions(self, frame: Frame) -> None:
        display = await asyncio.get_event_loop().run_in_executor(None, DisplayFrame, frame)
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
                    frame = self._get_frame(timestamp)
                    if frame is not None:
                        end = frequency + self.channels_per_substream
                        frame.data[frequency : end] = data
                        frame.present[frequency : end] = True
                        if frame.timestamp > self.last_full_timestamp and np.all(frame.present):
                            self._last_full_timestamp = frame.timestamp
                            self.server_context.add_next_tick_callback(
                                functools.partial(self._update_sessions, frame))

                # Allow them to be reclaimed before popping next heap
                del heap
                del item
                del raw_data
        except Exception as exc:
            print('Error in run', exc)
            raise
