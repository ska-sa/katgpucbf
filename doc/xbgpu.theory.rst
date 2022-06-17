Theory Of Operation
===================

.. todo::

   Most of this needs to be folded into the higher-level GPU "Design" document.
   Whatever remains will probably need re-naming under and "F-engine" sub-
   heading or some such.

Signal Flow
-----------

The general flow of data through the system is shown in the the image below:

.. figure:: images/concept.png
  :width: 887px

  XBGPU Concept

The X-Engine processing pipeline can be broken into three different stages:

  1. Receive data from the network and assemble it into a chunk. This chunk is
     then transferred to the GPU.
  2. The data is then correlated using a modified version of the ASTRON Tensor
     Core Kernel. This is done by the
     :class:`~katgpucbf.xbgpu.correlation.Correlation` class. This correlated
     data is then transferred back to system RAM.
  3. Send the correlated data (known as baseline correlation products) back into
     the network. This is implemented by :mod:`.xsend`.

The image below shows where the data is located at the various stages mentioned above:

.. figure:: images/hardware_path.png
  :width: 1096px

  Hardware Path


The numbers in the above image correspond to the following actions:

  0. Receive heaps from F-Engines.
  1. Assemble heaps into a chunk in system RAM.
  2. Transfer chunk to GPU memory.
  3. and
  4. Correlate data and transfer baselines to GPU memory.
  5. Transfer baselines from GPU memory to host memory.
  6. Transfer baselines from host memory to the NIC and onto the network.

Synchronization and Coordination
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


The :mod:`~katgpucbf.xbgpu.engine` module does the work of assembling all
the different modules into a pipeline. This module has three different async
processing pipelines know as the ``_receiver_loop``, ``_gpu_proc_loop`` and the
``_sender_loop``. Data is passed between these three processing loops using
:class:`asyncio.Queue`\ s. Buffers in queues are reused to prevent unnecessary memory
allocations. Additionally, buffers are passed between the Python program to the
network threads and back in order to reuse these buffers too.

The image below demonstrates how data moves through the pipeline and how it is
reused:

.. figure:: images/async_loops.png
  :width: 1112px

The :class:`asyncio.Queue` objects help to coordinate the flow of data through
the different asyncio functions. However the GPU requires a separate type of
coordination. The GPU has three different command queues that manage the
coordination.

One command queue is for processing and the other two are for transferring data
from host memory to the GPU and back. Events are put onto the command queue and
the async processing loops can :keyword:`await` for these events to be complete.
Often one async function will enqueue some commands followed by an event onto
the GPU command queue and the next async function will :keyword:`await` for this
event to complete as it is the function that needs to work with this data.
Tracking the different events across functions requires a bit of care to prevent
race conditions and deadlock.

The image below shows the interaction between the processing loops and the
command queues:

.. figure:: images/gpu_command_queues.png
  :width: 1094px

The numbers in the image above correspond to the following actions:

  1. Copy chunk to GPU memory from host
  2. Correlate chunk
  3. Transfer heap to host memory from GPU

Accumulations, Dumps and Auto-resync
------------------------------------

The input data is accumulated before being output. For every output heap,
multiple input heaps are received.

A heap from a single F-Engine consists of a set number of spectra indicated by
the :option:`!--spectra-per-heap` flag, where the spectra are time samples. Each of
these time samples is part of a different spectrum, meaning that the timestamp
difference per sample is equal to the value of :option:`!--samples-between-spectra`.
The timestamp difference between two consecutive heaps from the same F-Engine is equal to:

  `heap_timestamp_step = --spectra-per-heap * --samples-between-spectra`.

A :dfn:`batch` of heaps is a collection of heaps from different F-Engines with the same
timestamp. A :dfn:`chunk` consists of multiple consecutive batches (the number is given
by the option :option:`!--heaps-per-fengine-per-chunk`). Correlation generally occurs on
a chunk at a time, accumulating results, with the batches of the chunk being
processed in parallel.  To avoid race conditions in accumulation, there are
multiple accumulators, and batch *i* of a chunk uses accumulator *i*.
An accumulation period is called an :dfn:`accumulation` and the data output
from that accumulation is normally called a :dfn:`dump` — the terms are used
interchangeably. Once all the data for a dump has been correlated, the separate
accumulators are added together ("reduced") to produce a final result.  This
reduction process also converts from 64-bit to 32-bit integers, saturating if
necessary.

The number of batches to accumulate in an accumulation
is equal to the :option:`!--heap-accumulation-threshold` flag. The timestamp difference
between succesive dumps is therefore equal to:

  `timestamp_difference = --spectra-per-heap * --samples-between-spectra * --heap-accumulation-threshold`

The output heap timestamp is aligned to an integer multiple of
`timestamp_difference` (equivalent to the current SKARAB "auto-resync" logic).
The total accumulation time is equal to:

  `accumulation_time_s = timestamp_difference * --adc-sample-rate(Hz)` seconds.

The output heap contains multiple packets and these packets are distributed over
the entire `accumulation_time_s` interval to reduce network burstiness. The
default configuration in :mod:`katgpucbf.xbgpu.main` is for 0.5 second dumps
when using the MeerKAT 1712 MSps L-band digitisers.

The dump boundaries are aligned to whole batches, but may fall in the middle of
a chunk. In this case, each invocation of the correlation kernel will only
process a subset of the batches in the chunk.
