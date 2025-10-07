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

Batch
    In the context of the networking code, a collection of heaps that share a
    timestamp.

Chunk
    An array of data and associated metadata, including a timestamp. Chunks
    are the granularity at which data is managed within an engine (e.g., for
    transfer between CPU and GPU). To amortise per-chunk costs, chunks
    typically contain many SPEAD heaps. A chunk consists of one or more
    batches (see above) that are sequential in time.

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

Stream Group
    A group of incoming streams whose data are combined in chunks (see
    :class:`spead2.recv.ChunkStreamRingGroup`). Stream groups can be logically
    treated like a single stream, but allow receiving to be scaled across
    multiple CPU cores (with one member :class:`stream
    <spead2.recv.ChunkStreamGroupMember>` per thread).

Timestamp
    Timestamps are expressed in units of ADC (analogue-to-digital converter)
    samples, measured from a configurable "sync time". When a timestamp is
    associated with a collection of data, it generally reflects the timestamp
    of the *first* ADC sample that forms part of that data.

Operation
---------

The general operation of the DSP engines is illustrated in the diagram below:

.. tikz:: Data Flow. Double-headed arrows represent data passed through a
   queue and returned via a free queue.
   :libs: chains, fit

   \tikzset{proc/.style={draw, rounded corners, minimum width=4.5cm, minimum height=1cm},
            pproc-base/.style={minimum width=2cm, minimum height=1cm},
            pproc/.style={proc, pproc-base},
            flow/.style={->, >=latex, thick},
            queue/.style={flow, <->},
            fqueue/.style={queue, color=blue}}
   \begin{scope}[start chain=chain going below]
   \node[proc, on chain] (group) {Stream group};
   \node[proc, on chain] (upload) {Copy to GPU};
   \node[pproc, draw=none, anchor=west,
         start chain=rx0 going above, on chain=rx0] (group0) at (group.west) {};
   \node[pproc, draw=none, anchor=east,
         start chain=rx1 going above, on chain=rx1] (group1) at (group.east) {};
   \begin{scope}[start branch=stream0 going below]
     \node[proc, on chain=going below left] (process0) {GPU processing};
   \end{scope}
   \begin{scope}[start branch=stream1 going below]
     \node[proc, on chain=going below right] (process1) {GPU processing};
   \end{scope}
   \foreach \s in {0, 1} {
     \begin{scope}[continue chain=chain/stream\s]
     \node[proc, on chain] (download\s) {Copy from GPU};
     \node[proc, on chain] (transmit\s) {Transmit};
     \node[proc, on chain] (outstream\s) {Stream};
     \draw[queue] (upload) -- (process\s);
     \draw[queue] (process\s) -- (download\s);
     \draw[queue] (download\s) -- (transmit\s);
     \draw[flow] (transmit\s) -- (outstream\s);
     \end{scope}
   }
   \foreach \i in {0, 1} {
     \node[pproc, on chain=rx\i] (stream\i) {Stream};
     \draw[flow] (stream\i) -- (group\i);
   }
   \draw[queue] (group) -- (upload);
   \end{scope}

The F-engine uses two input streams and aligns two incoming polarisations, but
in the XB-engine there is only one.

There might not always be multiple processing pipelines. When they exist, they
are to support multiple outputs generated from the same input, such as wide-
and narrow-band F-engines, or correlation products and beams. Separate outputs
use separate output streams so that they can interleave their outputs while
transmitting at different rates. They share a thread to reduce the number of
cores required.

Chunking
^^^^^^^^
GPUs have massive parallelism, and to exploit them fully requires large batch
sizes (millions of elements). To accommodate this, the input packets are grouped
into "chunks" of fixed sizes. There is a tradeoff in the chunk size: large
chunks use more memory, add more latency to the system, and reduce LLC
(last-level cache) hit rates. Smaller chunks limit parallelism, and in the case
of the F-engine, increase the overheads associated with overlapping PFB
(polyphase filter bank) windows.

Chunking also helps reduce the impact of slow Python code. Digitiser output
heaps consist of only a single packet, and while F-engine output heaps can span
multiple packets, they are still rather small and involving Python on a per-heap
basis would be far too slow. We use :class:`spead2.recv.ChunkRingStream` or
:class:`spead2.recv.ChunkStreamRingGroup` to group heaps into chunks, which
means Python code is only run per-chunk.

Queues
^^^^^^
Both engines consist of several components which run independently of each
other — either via threads (spead2's C++ code) or Python's asyncio framework. The
general pattern is that adjacent components are connected by a pair of queues:
one carrying full buffers of data forward, and one returning free buffers. This
approach allows all memory to be allocated up front. Slow components thus
cause back-pressure on up-stream components by not returning buffers through
the free queue fast enough. The number of buffers needs to be large enough to
smooth out jitter in processing times.

A special case is the split from the receiver into multiple processing
pipelines. In this case each processing pipeline has an incoming queue with new
data (and each buffer is placed in each of these queues), but a single queue
for returning free buffers. Since a buffer can only be placed on the free queue
once it has been processed by all the pipelines, a reference count is held with
the buffer to track how many usages it has. This should not be confused with
the Python interpreter's reference count, although the purpose is similar.

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

The above concepts are illustrated in the following figure:

.. tikz:: GPU command queues, showing the upload, processing and download
    command queues, and the events (shown in green) used for synchronisation.
    :libs: chains

		[
		>=latex,
		block_clear/.style={rectangle,draw=black,minimum height=1cm,text width=2.0cm,align=center},
		block_green/.style={rectangle,draw=black,fill=green,minimum height=1cm,text width=0.25cm,align=center},
		block_text/.style={rectangle,minimum height=1cm,text width=2.0cm,align=center},
		]
        \node[block_text, anchor=center] (node0) at (0.0,0.0) {upload command queue};
		\node[block_clear, right = of node0, anchor=west] (node1) {Copy CPU $\rightarrow$ GPU};
		\draw[-] (node0.east) -- (node1.west);

		\node[block_green, right=0cm of node1] (node2){};

		\node[block_text, right=9.75cm of node2] (node10){};
		\draw[-] (node2.east) -- (node10.west);

		\node[block_clear,  above=3.0cm of node2.north east, anchor=west] (node3) {Recycle CPU Memory};
		\draw [->] (node2.east) -- (node3.south west) node [pos=0.5,left] {\texttt{async\_wait\_for\_events()}};

		\node[block_clear, below=2.0cm of node2.east, anchor=west] (node4) {Process};
		\draw [->] (node2.south east) -- (node4.north west) node [pos=0.5,right] {\texttt{enqueue\_wait\_for\_events()}};
		\node[block_green, right=0cm of node4] (node5){};

		\node[block_text, right=7.0cm of node5] (node11){};
		\draw[-] (node5.east) -- (node11.west);

		\node[block_text, left=4.75cm of node4, anchor=center] (node6) {processing command queue};
		\draw[-] (node6.east) -- (node4.west);

		\node[block_clear,  below=2.0cm of node5.south east, anchor=west] (node7) {Copy GPU $\rightarrow$ CPU};
		\node[block_green, right=0cm of node7] (node8){};

		\node[block_text, right=4.25cm of node8] (node12){};
		\draw[-] (node8.east) -- (node12.west);

		\draw [->] (node5.south east) -- (node7.north west) node [pos=0.5,left] {\texttt{enqueue\_wait\_for\_events()}};

		\node[block_text, left=6.25 of node7] (node9) {download command queue};
		\draw[-] (node9.east) -- (node7.west);

		\node[block_clear,  above=8.0cm of node8.east, anchor=west] (node10) {Transmit};
		\draw [->] (node8.north east) -- (node10.south west) node [pos=0.7,right] {\texttt{async\_wait\_for\_events()}};



Common features
---------------

.. _dithering:

Dithering
^^^^^^^^^
To improve linearity, a random value in the interval (-0.5, 0.5) is added to
each component (real and imaginary) before quantisation, in both the F-engine
and in beamforming (it is not needed for correlation because that takes place
entirely in integer arithmetic with no loss of precision). These values are
generated using `curand`_, with its underlying XORWOW generator. It is
designed for parallel use, with each work-item having the same seed but a
different `sequence` parameter to :cpp:func:`!curand_init`. This minimises
correlation between sequences generated by different threads. The sequence
numbers are also chosen to be distinct between the different engines, to avoid
correlation between channels.

Floating-point rounding issues make it tricky to get a perfectly zero-mean
distribution. While it is probably inconsequential, simply using
``curand_uniform(state) - 0.5f`` will not give zero mean. We solve this by
mapping the :math:`2^{32}` possible return values of :cpp:func:`!curand` to
the range :math:`(-2^{31}, 2^{31})` with zero represented twice, before
scaling to convert to a real value in :math:`(-0.5, 0.5)`. While this is
still a deviation from uniformity, it does give a symmetric distribution.

The :c:struct:`curandStateXORWOW_t` struct defined by curand is unnecessarily large
for our purposes, because it retains state needed to generate Gaussian
distributions (Box-Muller transform). To reduce global memory traffic, we use
a different type we define (:c:struct:`randState_t`) to hold random states in
global memory, together with helpers that save and restore this smaller state
from a private :c:struct:`curandStateXORWOW_t` used within a kernel.

.. _curand: https://docs.nvidia.com/cuda/curand/index.html

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
