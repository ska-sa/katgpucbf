import asyncio
from collections import deque
import warnings
from typing import Deque, List, Optional, Union

import numpy as np
from katsdpsigproc import accel, cuda, opencl

from .delay import AbstractDelayModel


# TODO: introduce these as real classes in katsdpsigproc
_AbstractContext = Union[cuda.Context, opencl.Context]
_AbstractCommandQueue = Union[cuda.CommandQueue, opencl.CommandQueue]
_AbstractEvent = Union[cuda.Event, opencl.Event]


class BaseItem:
    timestamp: int

    def __init__(self, timestamp: int = 0) -> None:
        self.reset(timestamp)

    def reset(self, timestamp: int = 0) -> None:
        self.timestamp = timestamp


class EventItem(BaseItem):
    event: Optional[_AbstractEvent]

    def enqueue_wait(self, command_queue: _AbstractCommandQueue) -> None:
        if self.event is not None:
            command_queue.enqueue_wait_for_events([self.event])

    def reset(self, timestamp: int = 0) -> None:
        super().reset(timestamp)
        self.event = None


class InItem(EventItem):
    samples: List[accel.DeviceArray]
    n_samples: int
    sample_bits: int

    def __init__(self, context: _AbstractContext, sample_bits: int, pols: int, capacity: int,
                 timestamp: int) -> None:
        if capacity % 8 != 0:
            raise ValueError('capacity must be a multiple of 8')
        n_bytes = capacity * self.sample_bits // 8
        # TODO: construct from an IOSlot?
        self.samples = [
            accel.DeviceArray(context, (n_bytes,), np.uint8)
            for pol in range(pols)
        ]
        super().__init__(timestamp)

    def reset(self, timestamp: int = 0) -> None:
        super().reset(timestamp)
        self.n_samples = 0

    @property
    def capacity(self) -> int:
        return self.samples[0].shape[0] * 8 // self.sample_bits

    @property
    def pols(self) -> int:
        return len(self.samples)

    @property
    def end_timestamp(self) -> int:
        return self.timestamp + self.n_samples


class OutItem(EventItem):
    spectra: accel.DeviceArray
    n_spectra: int

    def __init__(self, context: _AbstractContext, dtype: np.dtype,
                 capacity: int, channels: int, acc_len: int, pols: int,
                 timestamp: int = 0) -> None:
        if capacity % acc_len != 0:
            raise ValueError('capacity must be a multiple of acc_len')
        # TODO: construct from an IOSlot?
        self.spectra = accel.DeviceArray(context,
                                         (capacity // acc_len, channels, acc_len, pols, 2),
                                         dtype)
        super().__init__(timestamp)

    def reset(self, timestamp: int = 0) -> None:
        super().reset(timestamp)
        self.n_spectra = 0

    @property
    def end_timestamp(self) -> int:
        return self.timestamp + self.n_spectra

    @property
    def capacity(self) -> int:
        """Maximum number of spectra"""
        return self.spectra.shape[0] * self.spectra.shape[2]

    @property
    def pols(self) -> int:
        return self.spectra.shape[3]


class _MidItem(BaseItem):
    n_spectra: int
    offsets: accel.HostArray
    fine_delays: accel.HostArray

    def __init__(self, context: _AbstractContext, capacity: int) -> None:
        self.offsets = accel.HostArray((capacity,), np.uint32, context=context)
        self.fine_delays = accel.HostArray((capacity,), np.float32, context=context)
        self.reset()

    def add(self, offset: int, fine_delay: float) -> None:
        self.offsets[self.n_spectra] = offset
        self.fine_delays[self.n_spectra] = fine_delay
        self.n_spectra += 1

    def reset(self, timestamp: int = 0) -> None:
        super().reset(timestamp)
        self.n_spectra = 0

    @property
    def end_timestamp(self) -> int:
        return self.timestamp + self.n_spectra


class Processor:
    def __init__(self, context: _AbstractContext,
                 capacity: int, channels: int, taps: int, acc_len: int, pols: int,
                 delay_model: AbstractDelayModel,
                 in_queue: asyncio.Queue, in_free_queue: asyncio.Queue,
                 out_queue: asyncio.Queue, out_free_queue: asyncio.Queue) -> None:
        self.context = context
        self.channels = channels
        self.taps = taps
        self.acc_len = acc_len
        self.spectra_samples = 2 * channels
        self.delay_model = delay_model
        self.in_queue = in_queue
        self.in_free_queue = in_free_queue
        self.out_queue = out_queue
        self.out_free_queue = out_free_queue

        self._command_queue = context.create_command_queue()
        self._front_fn = TODO
        self._back_fn = TODO

        self._in_items: Deque[InItem] = deque()
        self._mid_item = _MidItem(context, capacity)
        self._out_item: OutItem = out_free_queue.get_nowait()

    async def _next_in(self) -> None:
        self._in_items.append(await self.in_queue.get())
        self._in_items[-1].enqueue_wait(self._command_queue)

    async def _next_out(self, new_timestamp: int) -> OutItem:
        item = await self.out_free_queue.get()
        item.enqueue_wait(self._command_queue)
        item.reset(new_timestamp)
        return item

    async def _flush_front(self, new_timestamp: int) -> None:
        assert self._out_item is not None
        if self._mid_item.n_spectra > 0:
            if self._out_item.n_spectra == 0:
                self._out_item.timestamp = self._mid_item.timestamp
            assert self._out_item.end_timestamp == self._mid_item.timestamp
            # TODO: bind buffers
            self._front_fn.n_spectra = self._mid_item.n_spectra
            self._front_fn.out_offset = self._out_item.n_spectra
            self._front_fn()
            self._out_item.n_spectra += self._mid_item.n_spectra
        self._mid_item.reset(new_timestamp)

    async def _flush_back(self, new_timestamp: int) -> None:
        await self._flush_front(new_timestamp)
        if self._out_item.n_spectra > 0:
            accs = self._out_item.n_spectra // self.acc_len
            self._out_item.n_spectra = accs * self.acc_len
            # TODO: bind buffers
            self._back_fn.n_spectra = self._out_item.n_spectra
            self._back_fn()
            self._out_item.event = self._command_queue.enqueue_marker()
            self.out_queue.put_nowait(self._out_item)
            self._out_item = await self._next_out(new_timestamp)
        else:
            self._out_item.timestamp = new_timestamp

    async def _run(self) -> None:
        # TODO: add a final flush on CancelledError?
        while True:
            if len(self._in_items) == 0:
                await self._next_in()
            if len(self._in_items) == 1:
                await self._next_in()
                # Copy the head of the new chunk to the tail of the older chunk
                # to allow for PFB windows to fit and for some protection against
                # sharp changes in delay.
                if self._in_items[0].end_timestamp == self._in_items[1].timestamp:
                    sample_bits = self._in_items[0].sample_bits
                    copy_samples = self._in_items[0].capacity - self._in_items[0].end_timestamp
                    copy_bytes = copy_samples * sample_bits // 8
                    end_bytes = self._in_items[0].n_samples * sample_bits // 8
                    for pol in range(len(self._in_items[0].samples)):
                        self._in_items[1].samples[pol].copy_region(
                            self._command_queue,
                            self._in_items[0].samples[pol],
                            np.s_[:copy_bytes],
                            np.s_[-copy_bytes:])
                    self._in_items[0].n_samples += copy_samples

            # If the input starts too late for the next expected timestamp,
            # we need to skip ahead to the next heap after the start.
            timestamp = self._mid_item.end_timestamp
            orig_timestamp, fine_delay = self.delay_model.invert(timestamp)
            if orig_timestamp < self._in_items[0].timestamp:
                align = self.acc_len * self.spectra_samples
                timestamp = max(timestamp, self._in_items[0].timestamp)
                timestamp = accel.roundup(timestamp, align)
                # TODO: add a helper to the delay model to accelerate this?
                # Might not be needed, since max delay is not many multiples of
                # align.
                while True:
                    timestamp += align
                    orig_timestamp, fine_delay = self.delay_model.invert(timestamp)
                    if orig_timestamp >= self._in_items[0].timestamp:
                        break
                await self._flush_back(timestamp)

            assert self._mid_item.end_timestamp == timestamp
            if orig_timestamp + self.taps * self.spectra_samples > self._in_items[0].end_timestamp:
                # Insufficient space in current input buffer for the PFB
                await self._flush_front(timestamp)
                item = self._in_items.popleft()
                item.event = self._command_queue.enqueue_marker()
                self.in_free_queue.put_nowait(item)
                continue

            assert self._mid_item.end_timestamp == timestamp
            self._mid_item.add(orig_timestamp - self._in_items[0].timestamp, fine_delay)
            if self._mid_item.n_spectra + self._out_item.n_spectra >= self._out_item.capacity:
                # We have enough data to fill an output buffer
                await self._flush_back(timestamp)
