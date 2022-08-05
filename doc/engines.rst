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

.. todo:: ``NGC-675``

    Explanation of network receive, GPU processing and network transmit "loops".
    There'll be a few merges from the existing F- and XBgpu sections, and the
    Glossary as well.

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
