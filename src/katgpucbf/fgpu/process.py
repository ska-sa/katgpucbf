import asyncio
from collections import deque
import warnings
from typing import Deque, List, Optional, Union

import numpy as np
from katsdpsigproc import accel
from katsdpsigproc.resource import async_wait_for_events

from .delay import AbstractDelayModel
from .compute import Compute
from .types import AbstractContext, AbstractCommandQueue, AbstractEvent
from . import recv, send, Stopped, Empty


def _device_allocate_slot(context: AbstractContext, slot: accel.IOSlot) -> accel.DeviceArray:
    return accel.DeviceArray(context, slot.shape, slot.dtype, slot.required_padded_shape())


def _host_allocate_slot(context: AbstractContext, slot: accel.IOSlot) -> accel.HostArray:
    return accel.HostArray(slot.shape, slot.dtype, slot.required_padded_shape(), context=context)


class BaseItem:
    timestamp: int

    def __init__(self, timestamp: int = 0) -> None:
        self.reset(timestamp)

    def reset(self, timestamp: int = 0) -> None:
        self.timestamp = timestamp


class EventItem(BaseItem):
    events: List[AbstractEvent]

    def enqueue_wait(self, command_queue: AbstractCommandQueue) -> None:
        command_queue.enqueue_wait_for_events(self.events)

    def reset(self, timestamp: int = 0) -> None:
        super().reset(timestamp)
        self.events = []


class InItem(EventItem):
    samples: List[accel.DeviceArray]
    n_samples: int
    sample_bits: int

    def __init__(self, compute: Compute, timestamp: int = 0) -> None:
        self.sample_bits = compute.sample_bits
        self.samples = [
            _device_allocate_slot(compute.template.context, compute.slots[f'in{pol}'])
            for pol in range(compute.pols)
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
    fine_delay: accel.HostArray
    n_spectra: int

    def __init__(self, compute: Compute, timestamp: int = 0) -> None:
        self.spectra = _device_allocate_slot(compute.template.context, compute.slots['out'])
        self.fine_delay = _host_allocate_slot(compute.template.context, compute.slots['fine_delay'])
        super().__init__(timestamp)

    def reset(self, timestamp: int = 0) -> None:
        super().reset(timestamp)
        self.n_spectra = 0

    @property
    def end_timestamp(self) -> int:
        return self.timestamp + self.n_spectra * 2 * self.channels

    @property
    def channels(self) -> int:
        return self.spectra.shape[1]

    @property
    def capacity(self) -> int:
        """Maximum number of spectra"""
        return self.spectra.shape[0] * self.spectra.shape[2]

    @property
    def pols(self) -> int:
        return self.spectra.shape[3]


class Processor:
    def __init__(self, compute: Compute, delay_model: AbstractDelayModel) -> None:
        self.compute = compute
        self.delay_model = delay_model
        self.in_queue = asyncio.Queue()        # type: asyncio.Queue[InItem]
        self.in_free_queue = asyncio.Queue()   # type: asyncio.Queue[InItem]
        self.out_queue = asyncio.Queue()       # type: asyncio.Queue[OutItem]
        self.out_free_queue = asyncio.Queue()  # type: asyncio.Queue[OutItem]
        for i in range(3):
            self.in_free_queue.put_nowait(InItem(compute))
        for i in range(1):
            self.out_free_queue.put_nowait(OutItem(compute))
        self._in_items: Deque[InItem] = deque()
        self._out_item = OutItem(compute)
        self._upload_queue = compute.template.context.create_command_queue()
        self._download_queue = compute.template.context.create_command_queue()

    @property
    def channels(self) -> int:
        return self.compute.channels

    @property
    def taps(self) -> int:
        return self.compute.template.taps

    @property
    def acc_len(self) -> int:
        return self.compute.acc_len

    @property
    def sample_bits(self) -> int:
        return self.compute.sample_bits

    @property
    def spectra_samples(self) -> int:
        return 2 * self.channels

    async def _next_in(self) -> None:
        self._in_items.append(await self.in_queue.get())
        print(f'Received input with timestamp {self._in_items[-1].timestamp}, {self._in_items[-1].n_samples} samples')
        self._in_items[-1].enqueue_wait(self.compute.command_queue)

    async def _next_out(self, new_timestamp: int) -> OutItem:
        item = await self.out_free_queue.get()
        item.enqueue_wait(self.compute.command_queue)
        item.reset(new_timestamp)
        return item

    async def _flush_out(self, new_timestamp: int) -> None:
        # Round down to a multiple of accs (don't send heap with partial
        # data).
        accs = self._out_item.n_spectra // self.acc_len
        self._out_item.n_spectra = accs * self.acc_len
        if self._out_item.n_spectra > 0:
            # TODO: only need to copy the relevant region, and can limit
            # postprocessing to the relevant range (the FFT size is baked into
            # the plan, so is harder to modify on the fly).
            self.compute.buffer('fine_delay').set_async(self.compute.command_queue,
                                                        self._out_item.fine_delay)
            self.compute.run_backend(self._out_item.spectra)
            self._out_item.events.append(self.compute.command_queue.enqueue_marker())
            self.out_queue.put_nowait(self._out_item)
            # TODO: could set it to None, since we only need it when we're
            # ready to flush again?
            self._out_item = await self._next_out(new_timestamp)
        else:
            self._out_item.timestamp = new_timestamp

    async def run_processing(self) -> None:
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
                    copy_samples = self._in_items[0].capacity - self._in_items[0].n_samples
                    copy_bytes = copy_samples * sample_bits // 8
                    end_bytes = self._in_items[0].n_samples * sample_bits // 8
                    for pol in range(len(self._in_items[0].samples)):
                        self._in_items[1].samples[pol].copy_region(
                            self.compute.command_queue,
                            self._in_items[0].samples[pol],
                            np.s_[:copy_bytes],
                            np.s_[-copy_bytes:])
                    self._in_items[0].n_samples += copy_samples

            # If the input starts too late for the next expected timestamp,
            # we need to skip ahead to the next heap after the start, and
            # flush what we already have.
            timestamp = self._out_item.end_timestamp
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
                await self._flush_out(timestamp)
            assert timestamp == self._out_item.end_timestamp

            coarse_delay = timestamp - orig_timestamp
            offset = orig_timestamp - self._in_items[0].timestamp
            end_timestamp = timestamp
            # Identify a block of frontend work. We can grow it until
            # - the coarse delay changes;
            # - we run out of the current input array; or
            # - we fill up the output array
            max_end_in = self._in_items[0].end_timestamp - self.taps * self.spectra_samples + 1
            max_end_out = self._out_item.timestamp + self._out_item.capacity * self.spectra_samples
            max_end = min(max_end_in, max_end_out)
            batch_spectra = 0
            # TODO: add functionality in delay model to speed this up
            while end_timestamp < max_end:
                orig_timestamp, fine_delay = self.delay_model.invert(end_timestamp)
                if end_timestamp - orig_timestamp != coarse_delay:
                    print('coarse delay changed')
                    break
                self._out_item.fine_delay[self._out_item.n_spectra + batch_spectra] = fine_delay
                end_timestamp += self.spectra_samples
                batch_spectra += 1

            if batch_spectra > 0:
                print(f'Processing {batch_spectra} spectra')
                self.compute.run_frontend(self._in_items[0].samples,
                                          offset,
                                          self._out_item.n_spectra,
                                          batch_spectra)
                self._out_item.n_spectra += batch_spectra

            if end_timestamp >= max_end_out:
                # We've filled up the output buffer.
                await self._flush_out(end_timestamp)

            if end_timestamp >= max_end_in:
                # We've exhausted the input buffer.
                # TODO: should also do this if _in_items[1] would work just as well and we've
                # filled the output buffer.
                item = self._in_items.popleft()
                item.events.append(self.compute.command_queue.enqueue_marker())
                self.in_free_queue.put_nowait(item)

    async def run_receive(self, streams: List[recv.Stream]) -> None:
        async for chunks in recv.chunk_sets(streams):
            in_item = await self.in_free_queue.get()
            await async_wait_for_events(in_item.events)
            in_item.reset(chunks[0].timestamp)
            in_item.n_samples = chunks[0].base.nbytes * 8 // self.sample_bits
            transfer_events = []
            for pol, chunk in enumerate(chunks):
                in_item.samples[pol].set_region(
                    self._upload_queue, chunk.base,
                    np.s_[:chunk.base.nbytes], np.s_[:],
                    blocking=False)
                transfer_events.append(self._upload_queue.enqueue_marker())
            in_item.events.extend(transfer_events)
            self.in_queue.put_nowait(in_item)
            for pol in range(len(chunks)):
                await async_wait_for_events([transfer_events[pol]])
                streams[pol].add_chunk(chunks[pol])

    async def run_transmit(self, sender: send.Sender) -> None:
        free_ring = send.AsyncRingbuffer(sender.free_ring)
        while True:
            out_item = await self.out_queue.get()
            self._download_queue.enqueue_wait_for_events(out_item.events)
            chunk = await free_ring.async_pop()
            # TODO: use get_region since it might be partial
            out_item.spectra.get_async(self._download_queue, chunk.base)
            chunk.timestamp = out_item.timestamp
            chunk.acc_len = self.acc_len
            chunk.channels = self.channels
            chunk.frames = out_item.n_spectra // self.acc_len
            transfer_event = self._download_queue.enqueue_marker()
            await async_wait_for_events([transfer_event])
            out_item.reset()
            self.out_free_queue.put_nowait(out_item)
            sender.send_chunk(chunk)
