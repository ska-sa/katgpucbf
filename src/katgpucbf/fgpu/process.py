################################################################################
# Copyright (c) 2020-2022, National Research Foundation (SARAO)
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

"""Classes which handle movement of data around the system.

Asynchronous coordination between getting data from system RAM to the GPU (or
directly from the NIC to the GPU in the GPU-direct case) is handled here.
Processing by the GPU is scheduled, and an event-based synchronisation mechanism
to manage copies at the right time. Getting the processed data back from the GPU
ultimately to the NIC is also handled.
"""

import asyncio
import functools
import logging
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Iterable, List, Optional, Sequence, Tuple, cast

import numpy as np
import spead2.recv
import spead2.send
import spead2.send.asyncio
from katsdpsigproc import accel
from katsdpsigproc.abc import AbstractCommandQueue, AbstractContext, AbstractEvent
from katsdpsigproc.resource import async_wait_for_events

from .. import BYTE_BITS, N_POLS
from ..monitor import Monitor
from . import recv, send
from .compute import Compute, ComputeTemplate
from .delay import AbstractDelayModel

logger = logging.getLogger(__name__)


def _device_allocate_slot(context: AbstractContext, slot: accel.IOSlot) -> accel.DeviceArray:
    return accel.DeviceArray(context, slot.shape, slot.dtype, slot.required_padded_shape())


def _host_allocate_slot(context: AbstractContext, slot: accel.IOSlot) -> accel.HostArray:
    return accel.HostArray(slot.shape, slot.dtype, slot.required_padded_shape(), context=context)


def _invert_models(
    delay_models: Iterable[AbstractDelayModel], start: int, stop: int, step: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Call :meth:`.AbstractDelayModel.invert_range` on multiple delay models and stack results."""
    # Each element of parts is a tuple of results from one delay model
    parts = [model.invert_range(start, stop, step) for model in delay_models]
    # Transpose so that each element of groups is one result from all delay models
    return tuple(np.stack(group) for group in zip(*parts))  # type: ignore


def generate_weights(channels: int, taps: int) -> np.ndarray:
    """Generate Hann-window weights for the F-engine's PFB-FIR.

    The resulting weights are normalised such that the sum of
    squares is 1.

    Parameters
    ----------
    channels
        Number of channels in the PFB.
    taps
        Number of taps in the PFB-FIR.

    Returns
    -------
    :class:`numpy.ndarray`
        Array containing the weights for the PFB-FIR filters, as
        single-precision floats.
    """
    step = 2 * channels
    window_size = step * taps
    idx = np.arange(window_size)
    hann = np.square(np.sin(np.pi * idx / (window_size - 1)))
    sinc = np.sinc((idx + 0.5) / step - taps / 2)
    weights = hann * sinc
    # Work around https://github.com/numpy/numpy/issues/21898
    weights /= np.sqrt(np.sum(np.square(weights)))  # type: ignore[misc]
    return weights.astype(np.float32)


class BaseItem:
    """Base item for use in the input and output queues.

    .. rubric:: TODO

    - Can probably be combined with :class:`EventItem` since nothing else but
      that inherits from this.

    Parameters
    ----------
    timestamp
        Timestamp of the item.
    """

    timestamp: int

    def __init__(self, timestamp: int = 0) -> None:
        self.reset(timestamp)

    def reset(self, timestamp: int = 0) -> None:
        """Reset the item's timestamp to the provided desired timestamp."""
        self.timestamp = timestamp


class EventItem(BaseItem):
    """Queue Item for use in synchronisation between command queues.

    Derived classes will have allocated memory regions associated with them,
    appropriately sized for input or output data. Actions (whether kernel
    executions or copies to or from the device) for these memory regions are
    initiated, and then an event marker is added to the list in some variation
    of this manner:

    .. code-block:: python

        my_item.events.append(command_queue.enqueue_marker())

    The item can then be passed through a queue to the next stage in the
    program, which waits for the operations to be complete using
    :func:`enqueue_wait()`. This indicates that the operation is complete and
    the next thing can be done with whatever data is in that region of memory.

    Attributes
    ----------
    events
        A list of GPU event markers generated by an
        :class:`~katsdpsigproc.abc.AbstractCommandQueue`.
    """

    events: List[AbstractEvent]

    def enqueue_wait(self, command_queue: AbstractCommandQueue) -> None:
        """Block execution until all of this item's events are finished.

        If the events have already passed, then the function will return
        immediately; if not, execution will pause until they are all reached.
        """
        command_queue.enqueue_wait_for_events(self.events)

    def reset(self, timestamp: int = 0) -> None:
        """Reset the item's timestamp, and empty the event list."""
        super().reset(timestamp)
        self.events = []


@dataclass
class PolInItem:
    """Polarisation-specific elements of :class:`InItem`.

    Attributes
    ----------
    samples
        A device memory region for storing the raw samples.
    present
        Bitmask indicating which packets were present in the chunk.
    present_cumsum
        Cumulative sum over :attr:`present`. It is up to the caller
        to compute it at the appropriate time.
    chunk
        Chunk to return to recv after processing (used with vkgdr only).
    """

    samples: Optional[accel.DeviceArray]
    present: np.ndarray
    present_cumsum: np.ndarray
    chunk: Optional[recv.Chunk] = None  # Used with vkgdr only.


class InItem(EventItem):
    """Item for use in input queues.

    This Item references GPU memory regions for input samples from both
    polarisations, with metadata describing their dimensions (number of samples
    and bitwidth of samples) in addition to the features of :class:`EventItem`.

    An example of usage is as follows:

    .. code-block:: python

        # In the receive function
        my_in_item.pol_data[pol].samples.set_region(...)  # start copying sample data to the GPU,
        my_in_item.events.append(command_queue.enqueue_marker())
        in_queue.put_nowait(my_in_item)
        ...
        # in the processing function
        next_in_item = await self.in_queue.get() # get the item from the queue
        next_in_item.enqueue_wait(command_queue) # wait for its data to be completely copied
        ... # carry on executing kernels or whatever needs to be done with the data

    Attributes
    ----------
    pol_data
        Per-polarisation data
    n_samples
        Number of samples in each :class:`~katsdpsigproc.accel.DeviceArray` in
        :attr:`PolInItem.samples`.
    sample_bits
        Bitwidth of the data in :attr:`PolInItem.samples`.

    Parameters
    ----------
    compute
        F-engine Operation Sequence detailing the computation operations which
        will take place on the data in :attr:`PolInItem.samples`.
    timestamp
        Timestamp of the oldest digitiser sample represented in the data.
    packet_samples
        Number of samples per digitiser packet (for sizing :attr:`PolInItem.present`).
    use_vkgdr
        Use vkgdr to write sample data directly to the GPU rather than staging in
        host memory.
    """

    pol_data: List[PolInItem]
    n_samples: int
    sample_bits: int

    def __init__(self, compute: Compute, timestamp: int = 0, *, packet_samples: int, use_vkgdr: bool = False) -> None:
        self.sample_bits = compute.sample_bits
        self.pol_data = []
        present_size = accel.divup(compute.samples, packet_samples)
        for pol in range(N_POLS):
            if use_vkgdr:
                # Memory belongs to the chunks, and we set samples when
                # initialising the item from the chunks.
                samples = None
            else:
                samples = _device_allocate_slot(compute.template.context, cast(accel.IOSlot, compute.slots[f"in{pol}"]))
            self.pol_data.append(
                PolInItem(
                    samples=samples,
                    present=np.zeros(present_size, dtype=bool),
                    present_cumsum=np.zeros(present_size + 1, np.uint32),
                )
            )
        super().__init__(timestamp)

    def reset(self, timestamp: int = 0) -> None:
        """Reset the item.

        Zero the timestamp, empty the event list and set number of samples to
        zero.
        """
        super().reset(timestamp)
        self.n_samples = 0

    @property
    def capacity(self) -> int:  # noqa: D401
        """Memory capacity in samples.

        The amount of space allocated to each polarisation stored in
        :attr:`PolInData.samples`.
        """
        assert self.pol_data[0].samples is not None
        return self.pol_data[0].samples.shape[0] * BYTE_BITS // self.sample_bits

    @property
    def end_timestamp(self) -> int:  # noqa: D401
        """Past-the-end (i.e. latest plus 1) timestamp of the item."""
        return self.timestamp + self.n_samples


class OutItem(EventItem):
    """Item for use in output queues.

    This Item references GPU memory regions for output spectra from both
    polarisations, with something about the fine delay, in addition to the
    features of :class:`EventItem`.

    An example of usage is as follows:

    .. code-block:: python

        # In the processing function
        compute.run_some_dsp(my_out_item.spectra) # Run the DSP, whatever it is.
        my_out_item.events.append(command_queue.enqueue_marker())
        out_queue.put_nowait(my_out_item)
        ...
        # in the transmit function
        next_out_item = await self.out_queue.get() # get the item from the queue
        next_out_item.enqueue_wait(download_queue) # wait for event indicating DSP is finished
        next_out_item.get_async(download_queue) # Start copying data back to the host
        ... # Normally you'd put a marker on the queue again so that you know when the
            # copy is finished, but this needn't be attached to the item unless
            # there's another queue afterwards.

    Attributes
    ----------
    spectra
        This is the actual output data, a collection of spectra, arranged in
        memory by pol and by heap.
    fine_delay
        Provides a scratch space for collecting per-spectrum fine delays while
        the `OutItem` is being prepared. When the `OutItem` is placed onto the
        queue it is copied to the `Compute`.
    phase
        A similar scratch space for collecting per-spectrum phase offsets while
        the :class:`OutItem` is being prepared.
    gains
        Per-channel gains
    present
        Bit-mask indicating which spectra contain valid data and should be
        transmitted.
    n_spectra
        Number of spectra contained in :attr:`spectra`.
    chunk
        Corresponding chunk for transmission (only used in PeerDirect mode).

    Parameters
    ----------
    compute
        F-engine Operation Sequence detailing the DSP happening on the data,
        including details for buffers, context, shapes, slots, etc.
    timestamp
        Timestamp of the first spectrum in the `OutItem`.
    """

    spectra: accel.DeviceArray
    fine_delay: accel.HostArray
    phase: accel.HostArray
    gains: accel.HostArray
    present: np.ndarray
    n_spectra: int
    chunk: Optional[send.Chunk] = None

    def __init__(self, compute: Compute, timestamp: int = 0) -> None:
        self.spectra = _device_allocate_slot(compute.template.context, cast(accel.IOSlot, compute.slots["out"]))
        self.fine_delay = _host_allocate_slot(compute.template.context, cast(accel.IOSlot, compute.slots["fine_delay"]))
        self.phase = _host_allocate_slot(compute.template.context, cast(accel.IOSlot, compute.slots["phase"]))
        self.gains = _host_allocate_slot(compute.template.context, cast(accel.IOSlot, compute.slots["gains"]))
        self.present = np.zeros(self.fine_delay.shape[0], dtype=bool)
        super().__init__(timestamp)

    def reset(self, timestamp: int = 0) -> None:
        """Reset the item.

        Zero the item's timestamp, empty the event list and set number of
        spectra to zero.
        """
        super().reset(timestamp)
        self.n_spectra = 0

    @property
    def end_timestamp(self) -> int:  # noqa: D401
        """Past-the-end timestamp of the item.

        Following Python's normal exclusive-end convention.
        """
        return self.timestamp + self.n_spectra * 2 * self.channels

    @property
    def channels(self) -> int:  # noqa: D401
        """Number of channels."""
        return self.spectra.shape[1]

    @property
    def capacity(self) -> int:  # noqa: D401
        """Number of spectra stored in memory for each polarisation."""
        # PostProc's __init__ method gives this as (spectra // spectra_per_heap)*(spectra_per_heap), so
        # basically, the number of spectra.
        return self.spectra.shape[0] * self.spectra.shape[2]

    @property
    def pols(self) -> int:  # noqa: D401
        """Number of polarisations."""
        return self.spectra.shape[3]


class Processor:
    """Controls the bulk of the moving of data around the computer.

    The Processor creates input and output :class:`~katgpucbf.monitor.Queue`
    objects as well as a few :class:`InItem` and :class:`OutItem` objects to use
    on them. The actual Items (and the memory associated) are then continuously
    re-used because the allocation of memory is expensive. The data buffers are
    simply overwritten whenever they are used.

    :attr:`in_queue` and :attr:`in_free_queue` recycle the InItems, and
    similarly, :attr:`out_queue` and :attr:`out_free_queue` recycle the
    OutItems.

    The command queues for scheduling data copy tasks to/from the GPU are also
    managed by the Processor. The Events contained by the Items are the link
    between these two kinds of queues.

    Attributes
    ----------
    compute
        :class:`OperationSequence` containing all the steps for carrying out the
        F-engine's processing.
    in_queue
        Ready :class:`InItem` objects for processing on the GPU.
    in_free_queue
        Available :class:`InItem` objects for overwriting.
    out_queue
        Ready :class:`OutItem` objects for transmitting on the network.
    out_free_queue
        Available :class:`OutItem` objects for overwriting.
    gains
        Per-channel, per-pol gains. These can be freely updated, and are
        copied into each OutItem as it is produced. It is thus not possible
        to precisely control when the updated gains will be applied. A
        sequence of changes made without any intervening asynchronous work
        will be applied atomically.

    Parameters
    ----------
    context
        The GPU context that we'll operate in.
    taps
        Number of taps in each branch of the PFB-FIR.
    samples
        Number of samples that will be processed each time the operation is run.
    src_packet_samples
        Number of samples per digitiser packet.
    spectra
        Number of spectra that will be produced from a chunk of incoming
        digitiser data.
    spectra_per_heap
        Number of spectra to send in each output heap.
    channels
        Number of output channels to produce.
    delay_models
        The delay models which should be applied to the data.
    use_vkgdr
        Assemble chunks directly in GPU memory (requires supported GPU).
    peerdirect_chunk_factory
        Specify this to use PeerDirect to transmit data directly from GPU
        memory. The value must be a callable that takes
        :attr:`OutItem.spectra` and returns a chunk that wraps it.
    monitor
        Monitor object to use for generating the :class:`~asyncio.Queue` objects
        and reporting their events.
    """

    def __init__(
        self,
        context: AbstractContext,
        taps: int,
        samples: int,
        src_packet_samples: int,
        spectra: int,
        spectra_per_heap: int,
        channels: int,
        delay_models: Sequence[AbstractDelayModel],
        use_vkgdr: bool,
        peerdirect_chunk_factory: Optional[Callable[[accel.DeviceArray], send.Chunk]],
        monitor: Monitor,
    ) -> None:
        compute_queue = context.create_command_queue()
        self._upload_queue = context.create_command_queue()
        self._download_queue = context.create_command_queue()

        template = ComputeTemplate(context, taps, channels)
        self.compute = template.instantiate(compute_queue, samples, spectra, spectra_per_heap)
        device_weights = self.compute.slots["weights"].allocate(accel.DeviceAllocator(context))
        device_weights.set(compute_queue, generate_weights(channels, taps))

        self.delay_models = delay_models

        # TODO: Perhaps declare these as constants at the top? Or Class variables?
        n_in = 3
        n_send = 4
        n_out = n_send if peerdirect_chunk_factory is not None else 2

        # The type annotations have to be in comments because Python 3.8
        # doesn't support the syntax at runtime (Python 3.9 fixes that).
        self.in_queue = monitor.make_queue("in_queue", n_in)  # type: asyncio.Queue[Optional[InItem]]
        self.in_free_queue = monitor.make_queue("in_free_queue", n_in)  # type: asyncio.Queue[InItem]
        self.out_queue = monitor.make_queue("out_queue", n_out)  # type: asyncio.Queue[Optional[OutItem]]
        self.out_free_queue = monitor.make_queue("out_free_queue", n_out)  # type: asyncio.Queue[OutItem]
        self.send_free_queue = monitor.make_queue("send_free_queue", n_send)  # type: asyncio.Queue[send.Chunk]

        self.monitor = monitor
        self._src_packet_samples = src_packet_samples
        for _ in range(n_in):
            self.in_free_queue.put_nowait(InItem(self.compute, packet_samples=src_packet_samples, use_vkgdr=use_vkgdr))
        for _ in range(n_out):
            item = OutItem(self.compute)
            if peerdirect_chunk_factory is not None:
                item.chunk = peerdirect_chunk_factory(item.spectra)
            self.out_free_queue.put_nowait(item)
        self._in_items: Deque[InItem] = deque()
        self._out_item = self.out_free_queue.get_nowait()
        self._use_vkgdr = use_vkgdr

        self.gains = np.zeros((self.channels, self.pols), np.complex64)

    @property
    def channels(self) -> int:  # noqa: D401
        """Number of channels into which the incoming signal is decomposed."""
        return self.compute.template.channels

    @property
    def taps(self) -> int:  # noqa: D401
        """Number of taps in the PFB-FIR filter."""
        return self.compute.template.taps

    @property
    def spectra_per_heap(self) -> int:  # noqa: D401
        """The number of spectra which will be transmitted per output heap."""
        return self.compute.spectra_per_heap

    @property
    def sample_bits(self) -> int:  # noqa: D401
        """Bitwidth of the incoming digitiser samples."""
        return self.compute.sample_bits

    @property
    def spectra_samples(self) -> int:  # noqa: D401
        """Number of incoming digitiser samples needed per spectrum.

        Note that this is the spacing between spectra. Each spectrum uses
        an overlapping window with more samples than this.
        """
        return 2 * self.channels

    @property
    def pols(self) -> int:  # noqa: D401
        """Number of polarisations."""
        return N_POLS

    async def _next_in(self) -> Optional[InItem]:
        """Load next InItem for processing.

        Move the next :class:`InItem` from the `in_queue` to `_in_items`, where
        it will be picked up by the processing.
        """
        with self.monitor.with_state("run_processing", "wait in_queue"):
            item = await self.in_queue.get()

        if item is not None:
            self._in_items.append(item)
            # print(f'Received input with timestamp {self._in_items[-1].timestamp}, '
            #       f'{self._in_items[-1].n_samples} samples')

            # Make sure that all events associated with the item are past.
            self._in_items[-1].enqueue_wait(self.compute.command_queue)
        else:
            # To keep run_processing simple, it may make further calls to
            # _next_in after receiving a None. To keep things simple, put
            # a None back into the queue so that the next call also gets
            # None rather than hanging.
            self.in_queue.put_nowait(None)
        return item

    async def _fill_in(self) -> bool:
        """Load sufficient InItems to continue processing.

        Tries to get at least two items into ``self._in_items``, and if
        loading a second item that is adjacent to the first, copies the overlap
        region.

        Returns true if processing can proceed, false if the stream is
        exhausted.
        """
        if len(self._in_items) == 0:
            if not (await self._next_in()):
                return False
        if len(self._in_items) == 1:
            # Copy the head of the new chunk to the tail of the older chunk
            # to allow for PFB windows to fit and for some protection against
            # sharp changes in delay.
            #
            # This could only fail if we'd lost a whole input chunk of
            # data from the digitiser. In that case the data we'd like
            # to copy is missing so we can't do this step.
            chunk_packets = self._in_items[0].n_samples // self._src_packet_samples
            copy_packets = len(self._in_items[0].pol_data[0].present) - chunk_packets
            if (await self._next_in()) and self._in_items[0].end_timestamp == self._in_items[1].timestamp:
                sample_bits = self._in_items[0].sample_bits
                copy_samples = self._in_items[0].capacity - self._in_items[0].n_samples
                copy_samples = min(copy_samples, self._in_items[1].n_samples)
                copy_bytes = copy_samples * sample_bits // BYTE_BITS
                for pol_data0, pol_data1 in zip(self._in_items[0].pol_data, self._in_items[1].pol_data):
                    assert pol_data0.samples is not None
                    assert pol_data1.samples is not None
                    pol_data1.samples.copy_region(
                        self.compute.command_queue,
                        pol_data0.samples,
                        np.s_[:copy_bytes],
                        np.s_[-copy_bytes:],
                    )
                    pol_data0.present[-copy_packets:] = pol_data1.present[:copy_packets]
                self._in_items[0].n_samples += copy_samples
            else:
                for pol_data in self._in_items[0].pol_data:
                    pol_data.present[-copy_packets:] = 0  # Mark tail as absent, for each pol
            # Update the cumulative sums. Note that during shutdown this may be
            # done more than once, but since it is shutdown the performance
            # implications aren't too important.
            # np.cumsum doesn't provide an initial zero, so we output starting at
            # position 1.
            for pol_data in self._in_items[0].pol_data:
                np.cumsum(pol_data.present, dtype=pol_data.present_cumsum.dtype, out=pol_data.present_cumsum[1:])
        return True

    def _pop_in(self, streams: List[spead2.recv.ChunkRingStream]) -> None:
        """Remove the oldest InItem."""
        item = self._in_items.popleft()
        event = self.compute.command_queue.enqueue_marker()
        if self._use_vkgdr:
            chunks = []
            for pol_data in item.pol_data:
                pol_data.samples = None
                assert pol_data.chunk is not None
                chunks.append(pol_data.chunk)
                pol_data.chunk = None
            asyncio.create_task(self._push_recv_chunks(streams, chunks, event))
        else:
            item.events.append(event)
        self.in_free_queue.put_nowait(item)

    async def _next_out(self, new_timestamp: int) -> OutItem:
        """Grab the next free OutItem in the queue."""
        with self.monitor.with_state("run_processing", "wait out_free_queue"):
            item = await self.out_free_queue.get()

        # Just make double-sure that all events associated with the item are past.
        item.enqueue_wait(self.compute.command_queue)
        item.reset(new_timestamp)
        return item

    async def _flush_out(self, new_timestamp: int) -> None:
        """Start the backend processing and prepare the data for transmission.

        Kick off the `run_backend()` processing, and put an event on the
        relevant command queue. This lets the next coroutine (run_transmit) know
        that the backend processing is finished, and the data can be transmitted
        out.

        Parameters
        ----------
        new_timestamp
            The timestamp that will immediately follow the current OutItem.
        """
        # Round down to a multiple of accs (don't send heap with partial
        # data).
        accs = self._out_item.n_spectra // self.spectra_per_heap
        self._out_item.n_spectra = accs * self.spectra_per_heap
        if self._out_item.n_spectra > 0:
            # Take a copy of the gains synchronously. This avoids race conditions
            # with gains being updated at the same time as they're in the
            # middle of being transferred.
            self._out_item.gains[:] = self.gains
            # TODO: only need to copy the relevant region, and can limit
            # postprocessing to the relevant range (the FFT size is baked into
            # the plan, so is harder to modify on the fly).
            self.compute.buffer("fine_delay").set_async(self.compute.command_queue, self._out_item.fine_delay)
            self.compute.buffer("phase").set_async(self.compute.command_queue, self._out_item.phase)
            self.compute.buffer("gains").set_async(self.compute.command_queue, self._out_item.gains)
            self.compute.run_backend(self._out_item.spectra)
            self._out_item.events.append(self.compute.command_queue.enqueue_marker())
            self.out_queue.put_nowait(self._out_item)
            # TODO: could set it to None, since we only need it when we're
            # ready to flush again?
            self._out_item = await self._next_out(new_timestamp)
        else:
            self._out_item.timestamp = new_timestamp

    @staticmethod
    async def _push_recv_chunks(
        streams: Iterable[spead2.recv.ChunkRingStream], chunks: Iterable[recv.Chunk], event: AbstractEvent
    ) -> None:
        """Return chunks to the streams once `event` has fired.

        This is only used when using vkgdr.
        """
        await async_wait_for_events([event])
        for stream, chunk in zip(streams, chunks):
            stream.add_free_chunk(chunk)

    async def run_processing(self, streams: List[spead2.recv.ChunkRingStream]) -> None:
        """Do the hard work of the F-engine.

        This function takes place entirely on the GPU. First, a little bit of
        the next chunk is copied to the end of the previous one, to allow for
        the overlap required by the PFB. Coarse delay happens. Then a batch FFT
        operation is applied, and finally, fine-delay, phase correction,
        quantisation and corner-turn are performed.

        Parameters
        ----------
        streams
            These only seem to be used in the _use_vkgdr case.
        """
        while await self._fill_in():
            # If the input starts too late for the next expected timestamp,
            # we need to skip ahead to the next heap after the start, and
            # flush what we already have.
            start_timestamp = self._out_item.end_timestamp
            orig_start_timestamps = [model.invert(start_timestamp)[0] for model in self.delay_models]
            if min(orig_start_timestamps) < self._in_items[0].timestamp:
                align = self.spectra_per_heap * self.spectra_samples
                start_timestamp = max(start_timestamp, self._in_items[0].timestamp)
                start_timestamp = accel.roundup(start_timestamp, align)
                # TODO: add a helper to the delay model to accelerate this?
                # Might not be needed, since max delay is not many multiples of
                # align.
                while True:
                    orig_start_timestamps = [model.invert(start_timestamp)[0] for model in self.delay_models]
                    if min(orig_start_timestamps) >= self._in_items[0].timestamp:
                        break
                    start_timestamp += align
                await self._flush_out(start_timestamp)
            # When we add new spectra they must follow contiguously for any
            # that we've already buffered.
            assert start_timestamp == self._out_item.end_timestamp

            # Compute the coarse delay for the first sample.
            # `orig_timestamp` is the timestamp of first sample from the input
            # to process in the PFB to produce the output spectrum with
            # `timestamp`. `offset` is the sample index corresponding to
            # `orig_timestamp` within the InItem.
            start_coarse_delays = [start_timestamp - orig_timestamp for orig_timestamp in orig_start_timestamps]
            offsets = [orig_timestamp - self._in_items[0].timestamp for orig_timestamp in orig_start_timestamps]

            # Identify a block of frontend work. We can grow it until
            # - we run out of the current input array;
            # - we fill up the output array; or
            # - the coarse delay changes.
            # We speculatively calculate delays until one of the first two is
            # met, then truncate if we observe a coarse delay change. Note:
            # max_end_in is computed assuming the coarse delay does not change.
            max_end_in = (
                self._in_items[0].end_timestamp + min(start_coarse_delays) - self.taps * self.spectra_samples + 1
            )
            max_end_out = self._out_item.timestamp + self._out_item.capacity * self.spectra_samples
            max_end = min(max_end_in, max_end_out)
            # Speculatively evaluate until one of the first two conditions is met
            timestamps = np.arange(start_timestamp, max_end, self.spectra_samples)
            orig_timestamps, fine_delays, phase = _invert_models(
                self.delay_models, start_timestamp, max_end, self.spectra_samples
            )
            for pol in range(len(orig_timestamps)):
                coarse_delays = timestamps - orig_timestamps[pol]
                # Uses fact that argmax returns first maximum i.e. first true value
                delay_change = int(np.argmax(coarse_delays != start_coarse_delays[pol]))
                if coarse_delays[delay_change] != start_coarse_delays[pol]:
                    logger.debug(
                        "Coarse delay on pol %d changed from %d to %d at %d",
                        pol,
                        start_coarse_delays[pol],
                        coarse_delays[delay_change],
                        orig_timestamps[pol, delay_change],
                    )
                    timestamps = timestamps[:delay_change]
                    orig_timestamps = orig_timestamps[:, :delay_change]
                    fine_delays = fine_delays[:, :delay_change]
                    phase = phase[:, :delay_change]
            batch_spectra = orig_timestamps.shape[1]

            # Here we run the "frontend" which handles:
            # - 10-bit to float conversion
            # - Coarse delay
            # - The PFB-FIR.
            if batch_spectra > 0:
                logging.debug("Processing %d spectra", batch_spectra)
                out_slice = np.s_[self._out_item.n_spectra : self._out_item.n_spectra + batch_spectra]
                self._out_item.fine_delay[out_slice] = fine_delays.T
                # Divide by pi because the arguments of sincospif() used in the
                # kernel are in radians/PI.
                self._out_item.phase[out_slice] = phase.T / np.pi
                samples = []
                for pol_data in self._in_items[0].pol_data:
                    assert pol_data.samples is not None
                    samples.append(pol_data.samples)
                self.compute.run_frontend(samples, offsets, self._out_item.n_spectra, batch_spectra)
                self._out_item.n_spectra += batch_spectra
                # Work out which output spectra contain missing data.
                self._out_item.present[out_slice] = True
                for pol, pol_data in enumerate(self._in_items[0].pol_data):
                    # Offset in the chunk of the first sample for each spectrum
                    first_offset = np.arange(
                        offsets[pol],
                        offsets[pol] + batch_spectra * self.spectra_samples,
                        self.spectra_samples,
                    )
                    # Offset of the last sample (inclusive, rather than past-the-end)
                    last_offset = first_offset + self.taps * self.spectra_samples - 1
                    first_packet = first_offset // self._src_packet_samples
                    # last_packet is exclusive
                    last_packet = last_offset // self._src_packet_samples + 1
                    present_packets = pol_data.present_cumsum[last_packet] - pol_data.present_cumsum[first_packet]
                    self._out_item.present[out_slice] &= present_packets == last_packet - first_packet

            # The _flush_out method calls the "backend" which triggers the FFT
            # and postproc operations.
            end_timestamp = self._out_item.end_timestamp
            if end_timestamp >= max_end_out:
                # We've filled up the output buffer.
                await self._flush_out(end_timestamp)

            if end_timestamp >= max_end_in:
                # We've exhausted the input buffer.
                # TODO: should maybe also do this if _in_items[1] would work
                # just as well and we've filled the output buffer.
                self._pop_in(streams)
        # Timestamp mostly doesn't matter because we're finished, but if a
        # katcp request arrives at this point we want to ensure the
        # steady-state-timestamp sensor is updated to a later timestamp than
        # anything we'll actually send.
        await self._flush_out(self._out_item.end_timestamp)
        logger.debug("run_processing completed")
        self.out_queue.put_nowait(None)

    async def run_receive(self, streams: List[spead2.recv.ChunkRingStream], layout: recv.Layout) -> None:
        """Receive data from the network, queue it up for processing.

        This function receives chunk sets, which are chunks in groups of two -
        one per polarisation, from the spead2 receiver streams given. For each
        chunk set received, copies of the data to the GPU are initiated,
        awaited, and then the chunk containers are returned to the receiver
        stream so that the memory need not be expensively re-allocated every
        time.

        In the GPU-direct case, <TODO clarify once I understand better>.

        Parameters
        ----------
        streams
            There should be only two of these because they each represent one of
            the digitiser's two polarisations.
        layout
            The structure of the streams.
        """
        async for chunks in recv.chunk_sets(streams, layout):
            with self.monitor.with_state("run_receive", "wait in_free_queue"):
                in_item = await self.in_free_queue.get()
            with self.monitor.with_state("run_receive", "wait events"):
                # Make sure all the item's events are past.
                await async_wait_for_events(in_item.events)
            in_item.reset(chunks[0].timestamp)

            # In steady-state, chunks should be the same size, but during
            # shutdown, the last chunk may be short.
            in_item.n_samples = chunks[0].data.nbytes * BYTE_BITS // self.sample_bits

            transfer_events = []
            for pol_data, chunk in zip(in_item.pol_data, chunks):
                # Copy the present flags (synchronously).
                pol_data.present[: len(chunk.present)] = chunk.present
            if self._use_vkgdr:
                for pol_data, chunk in zip(in_item.pol_data, chunks):
                    assert pol_data.samples is None
                    pol_data.samples = chunk.device  # type: ignore
                    pol_data.chunk = chunk
                self.in_queue.put_nowait(in_item)
            else:
                # Copy each pol chunk to the right place on the GPU.
                for pol_data, chunk in zip(in_item.pol_data, chunks):
                    assert pol_data.samples is not None
                    pol_data.samples.set_region(
                        self._upload_queue, chunk.data, np.s_[: chunk.data.nbytes], np.s_[:], blocking=False
                    )
                    transfer_events.append(self._upload_queue.enqueue_marker())

                # Put events on the queue so that run_processing() knows when to
                # start.
                in_item.events.extend(transfer_events)
                self.in_queue.put_nowait(in_item)

                # Wait until the copy is done, and then give the chunks of memory
                # back to the receiver streams for reuse.
                for pol in range(len(chunks)):
                    with self.monitor.with_state("run_receive", "wait transfer"):
                        await async_wait_for_events([transfer_events[pol]])
                    streams[pol].add_free_chunk(chunks[pol])
        logger.debug("run_receive completed")
        self.in_queue.put_nowait(None)

    def _chunk_finished(self, chunk: send.Chunk, future: asyncio.Future) -> None:
        """Return a chunk to the free queue after it has completed transmission.

        This is intended to be used as a callback on an :class:`asyncio.Future`.
        """
        if chunk.cleanup is not None:
            chunk.cleanup()
            chunk.cleanup = None  # Potentially helps break reference cycles
        try:
            future.result()  # No result, but want the exception
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Error sending chunk")

    async def run_transmit(self, stream: "spead2.send.asyncio.AsyncStream") -> None:
        """Get the processed data from the GPU to the Network.

        This could be done either with or without PeerDirect. In the
        non-PeerDirect case, :class:`OutItem` objects are pulled from the
        `out_queue`. We wait for the events that mark the end of the processing,
        then copy the data to host memory before turning it over to the
        :obj:`sender` for transmission on the network. The "empty" item is then
        returned to :meth:`run_processing` via the `out_free_queue`, and once
        the chunk has been transmitted it is returned to `send_free_queue`.

        In the PeerDirect case, the item and the chunk are bound together and
        share memory. In this case `send_free_queue` is unused. The item is
        only returned to `out_free_queue` once it has been fully transmitted.

        Parameters
        ----------
        stream
            The stream transmitting data.
        """
        task: Optional[asyncio.Future] = None
        last_end_timestamp: Optional[int] = None
        while True:
            with self.monitor.with_state("run_transmit", "wait out_queue"):
                out_item = await self.out_queue.get()
            if not out_item:
                break
            if out_item.chunk is not None:
                # We're using PeerDirect
                chunk = out_item.chunk
                chunk.cleanup = functools.partial(self.out_free_queue.put_nowait, out_item)
                events = out_item.events
            else:
                with self.monitor.with_state("run_transmit", "wait send_free_queue"):
                    chunk = await self.send_free_queue.get()
                chunk.cleanup = functools.partial(self.send_free_queue.put_nowait, chunk)
                self._download_queue.enqueue_wait_for_events(out_item.events)
                assert isinstance(chunk.data, accel.HostArray)
                # TODO: use get_region since it might be partial
                out_item.spectra.get_async(self._download_queue, chunk.data)
                events = [self._download_queue.enqueue_marker()]

            chunk.timestamp = out_item.timestamp
            # Each frame is valid if all spectra in it are valid
            out_item.present.reshape(-1, self.spectra_per_heap).all(axis=-1, out=chunk.present)
            with self.monitor.with_state("run_transmit", "wait transfer"):
                await async_wait_for_events(events)
            n_frames = out_item.n_spectra // self.spectra_per_heap
            if last_end_timestamp is not None and out_item.timestamp > last_end_timestamp:
                # Account for heaps skipped between the end of the previous out_item and the
                # start of the current one.
                skipped_samples = out_item.timestamp - last_end_timestamp
                skipped_frames = skipped_samples // (self.spectra_per_heap * self.spectra_samples)
                send.skipped_heaps_counter.inc(skipped_frames * stream.num_substreams)
            last_end_timestamp = out_item.end_timestamp
            out_item.reset()  # Safe to call in PeerDirect mode since it doesn't touch the raw data
            if out_item.chunk is None:
                # We're not in PeerDirect mode
                # (when we are the cleanup callback returns the item)
                self.out_free_queue.put_nowait(out_item)
            task = asyncio.create_task(chunk.send(stream, n_frames))
            task.add_done_callback(functools.partial(self._chunk_finished, chunk))

        if task:
            try:
                await task
            except Exception:
                pass  # It's already logged by the chunk_finished callback
        stop_heap = spead2.send.Heap(send.FLAVOUR)
        stop_heap.add_end()
        for substream_index in range(stream.num_substreams):
            await stream.async_send_heap(stop_heap, substream_index=substream_index)
        logger.debug("run_transmit completed")

    async def run_descriptors_loop(
        self,
        stream: "spead2.send.asyncio.AsyncStream",
        descriptor_heap_reflist: List[spead2.send.HeapReference],
        n_ants: int,
        feng_id: int,
        base_interval_s: float,
    ) -> None:
        """Send the Antenna Channelised Voltage descriptors.

        The descriptors are initially sent once straight away. This loop then
        sleeps for `feng_id x base_interval_s` seconds, then continually sends
        descriptors every `n_ants x base_interval_s` seconds.

        Parameters
        ----------
        stream
            This object takes large chunks of data and packages it
            appropriately in SPEAD heaps for transmission on the network.
        descriptor_heap_reflist
            The descriptors describing the format of the heaps in the data
            stream. Formatted as a list of HeapReference's to be as efficient
            as possible during the send procedure.
        base_interval_s
            The base interval used as a multiplier on feng_id and n_ants to
            dictate the initial 'engine sleep interval' and 'send interval'
            respectively.
        """
        await asyncio.gather(
            self.async_send_descriptors(stream, descriptor_heap_reflist), asyncio.sleep(feng_id * base_interval_s)
        )
        send_interval_s = n_ants * base_interval_s
        while True:
            await asyncio.gather(
                self.async_send_descriptors(stream, descriptor_heap_reflist), asyncio.sleep(send_interval_s)
            )

    def async_send_descriptors(
        self,
        stream: "spead2.send.asyncio.AsyncStream",
        descriptor_heap_reflist: List[spead2.send.HeapReference],
    ) -> asyncio.Future:
        """Send one descriptor to every substream.

        Parameters
        ----------
        stream
            This object takes large chunks of data and packages it
            appropriately in SPEAD heaps for transmission on the network.
        descriptor_heap_reflist
            See :meth:`~fgpu.send.make_descriptor_heaps` for more information.
        """
        return stream.async_send_heaps(heaps=descriptor_heap_reflist, mode=spead2.send.GroupMode.ROUND_ROBIN)

    def set_gains(self, input: int, gains: np.ndarray) -> int:
        """Update the gains.

        Returns
        -------
        steady_state_timestamp
            A timestamp by which the gains are guaranteed to be applied
        """
        self.gains[:, input] = gains
        # This timestamp is conservative: self._out_item.timestamp is almost
        # always valid, except while _flush_out is waiting to update
        # self._out_item. If a less conservative answer is needed, one would
        # need to track a separate timestamp in the class that is updated
        # as gains are copied to the OutItem.
        return self._out_item.end_timestamp

    def delay_update_timestamp(self) -> int:
        """Return a timestamp by which an update to the delay model will take effect."""
        # end_timestamp is updated whenever delays are written into the out_item
        return self._out_item.end_timestamp
