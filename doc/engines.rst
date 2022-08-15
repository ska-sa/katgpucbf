DSP Engine Design
=================


.. _gpu-terminology:

Terminology
-----------

We will use OpenCL terminology, as it is more generic. If you're more familiar
with CUDA terminology, katsdpsigproc's
:external+katsdpsigproc:doc:`introduction <user/intro>` has a table mapping the
most important concepts. For definitions of the concepts, refer to chapter 2 of
the `OpenCL specification`_. A summary of the most relevant concepts can also
be found `here`_.

.. _OpenCL specification: https://www.khronos.org/registry/OpenCL/specs/3.0-unified/pdf/OpenCL_API.pdf
.. _here: http://downloads.ti.com/mctools/esd/docs/opencl/execution/terminology.html

Glossary
--------
This section serves (hopefully) to clarify some potentially confusing terms used
within the source code.

Chunk
    An array of data and associated metadata, including a timestamp. Chunks
    are the granularity at which data is managed within an engine (e.g., for
    transfer between CPU and GPU). To amortise per-chunk costs, chunks
    typically contain many SPEAD heaps.

Command Queue
    Channel for submitting work to a GPU. See
    :class:`katsdpsigproc.abc.AbstractCommandQueue`.

Device
    GPU or other OpenCL accelerator device (which in general could even be the
    CPU). See :class:`katsdpsigproc.abc.AbstractDevice`.

Engine
    A single process which consumes and/or produces SPEAD data, and is managed
    by katcp. An F-engine processes data for one antenna; an XB-Engine
    processes data for a configurable subset of the correlator's bandwidth. It
    is expected that a correlator will run more than one engine per server.

.. _dfn-event:

Event
    Used for synchronisation between command queues or between a command queue
    and the host. See :class:`katsdpsigproc.abc.AbstractEvent`.

Heap
    Basic message unit of SPEAD. Heaps may comprise one or more packets.

Queue
    See :class:`asyncio.Queue`. Not to be confused with Command Queues.

Queue Item
    See :class:`.QueueItem`. These are passed around on Queues.

Stream
    A stream of SPEAD data. The scope is somewhat flexible, depending on the
    viewpoint, and might span one or many multicast groups. For example, one
    F-engine sends to many XB-engines (using many multicast groups), and this
    is referred to as a single stream in the fgpu code. Conversely, an
    XB-engine receives data from many F-engines (but using only one multicast
    group), and that is also called "a stream" within the xbgpu code.

    This should not be confused with a CUDA stream, which corresponds to a
    Command Queue in OpenCL terminology.

Timestamp
    Timestamps are expressed in units of ADC (analogue-to-digital converter)
    samples, measured from a configurable "sync epoch" (also known as the "sync
    time"). When a timestamp is associated with a collection of data, it
    generally reflects the timestamp of the *first* ADC sample that forms part
    of that data.


Operation
---------

The general operation of the DSP engines is illustrated in the diagram below:

.. tikz:: Data Flow. Double-headed arrows represent data passed through a
   queue and returned via a free queue.
   :libs: chains

   \tikzset{proc/.style={draw, rounded corners, minimum width=4.5cm, minimum height=1cm},
            pproc/.style={proc, minimum width=2cm},
            flow/.style={->, >=latex, thick},
            queue/.style={flow, <->},
            fqueue/.style={queue, color=blue}}
   \node[proc, start chain=going below, on chain] (align) {Align, copy to GPU};
   \node[pproc, draw=none, anchor=west,
         start chain=rx0 going above, on chain=rx0] (align0) at (align.west) {};
   \node[pproc, draw=none, anchor=east,
         start chain=rx1 going above, on chain=rx1] (align1) at (align.east) {};
   \node[proc, on chain] (process) {GPU processing};
   \node[proc, on chain] (download) {Copy from GPU};
   \node[proc, on chain] (transmit) {Transmit};
   \node[pproc, draw=none, anchor=west,
         start chain=tx0 going below, on chain=tx0] (transmit0) at (transmit.west) {};
   \node[pproc, draw=none, anchor=east,
         start chain=tx1 going below, on chain=tx1] (transmit1) at (transmit.east) {};
   \foreach \i in {0, 1} {
     \node[pproc, on chain=rx\i] (receive\i) {Receive};
     \node[pproc, on chain=rx\i] (stream\i) {Stream};
     \node[pproc, on chain=tx\i] (outstream\i) {Stream};
   }
   \foreach \i in {0, 1} {
     \draw[flow] (stream\i) -- (receive\i);
     \draw[queue] (receive\i) -- (align\i);
     \draw[flow] (transmit\i) -- (outstream\i);
   }
   \draw[queue] (align) -- (process);
   \draw[queue] (process) -- (download);
   \draw[queue] (download) -- (transmit);

The F-engine uses two input streams and aligns two incoming polarisations, but
in the X-engine, this collapses to only a single stream.

Chunking
^^^^^^^^
GPUs have massive parallelism, and to exploit them fully requires large batch
sizes (millions of elements). To accommodate this, the input packets are grouped
into "chunks" of fixed sizes. There is a tradeoff in the chunk size: large
chunks use more memory, add more latency to the system, and reduce LLC
(last-level cache) hit rates. Smaller chunks limit parallelism, and in the case
of the F-engine, increase the overheads associated with overlapping PFB
(polyphase filter bank) windows.

Chunking also helps reduce the impact of slow Python code. Digitiser and
F-engine output heaps consist of only a single packet, and involving Python on a
per-heap basis would be far too slow. We use
:class:`spead2.recv.ChunkRingStream` to group heaps into chunks, which means
Python code is only run per-chunk.


Queues
^^^^^^
Both engines consist of several components which run independently of each
other - either via threads (spead2's C++ code) or Python's asyncio framework. The
general pattern is that adjacent components are connected by a pair of queues:
one carrying full buffers of data forward, and one returning free buffers. This
approach allows all memory to be allocated up front. Slow components thus
cause back-pressure on up-stream components by not returning buffers through
the free queue fast enough. The number of buffers needs to be large enough to
smooth out jitter in processing times.

Transfers and events
^^^^^^^^^^^^^^^^^^^^

To achieve the desired throughput it is necessary to overlap transfers to and
from the GPU with its computations. Transfers are done using separate command
queues, and an CUDA/OpenCL event (see :ref:`the glossary<dfn-event>`) is
associated with the completion of each transfer. Where possible, these events
are passed to the device to be waited for, so that the CPU does not need to
block. The CPU does need to wait for host-to-device transfers before putting the
buffer onto the free queue, and for device-to-host transfers before transmitting
results, but this is deferred as long as possible.




Common features
---------------

.. _engines-shutdown-procedure:

Shutdown procedures
^^^^^^^^^^^^^^^^^^^
The dsim, fgpu and xbgpu all make use of the
:external+aiokatcp:py:class:`aiokatcp server <aiokatcp.server.DeviceServer>`'s
:external+aiokatcp:py:meth:`on_stop <aiokatcp.server.DeviceServer.on_stop>`
feature which allows for any engine-specific clean-up to take place before
coming to a final halt.

The ``on_stop`` procedure is broadly similar between the dsim, fgpu and xbgpu.

* The ``dsim`` simply stops its internal calculation and sending processes of
  data and descriptors respectively.
* ``fgpu`` and ``xbgpu`` both stop their respective
  :external+spead2:doc:`spead2 receivers <recv-chunk>`, which allows for a more
  natural ending of internal processing operations.

  *  Each stage of processing passes a `None`-type on to the next stage,
  *  Eventually resulting in the engine sending a
     :external+spead2:doc:`SPEAD stop heap <py-protocol>` across its output
     streams.
