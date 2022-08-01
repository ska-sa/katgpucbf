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

.. todo::  ``NGC-674``
    Update Glossary section.

This section serves (hopefully) to clarify some potentially confusing terms used
within the source code.

Stream
^^^^^^
 - In the katgpucbf sense, this is derived from a spead2 recv or send stream.
 - Not to be confused with a CUDA stream, which is here encapsulated as a
   command queue. CUDA streams aren't referred to directly in katgpucbf.

Queue
^^^^^

- asyncio.Queue

  - This is a python Queue object that is designed for use in async programs.
    In katgpucbf.fgpu, we use EventItems on Queues.

- CommandQueue

  - This is relevant to the GPU. Things placed in this queue are guaranteed to
    be executed in order.
  - There can be an arbitrary number of command queues. In katgpucbf.fgpu, we use one
    for uploading, one for the actual DSP, and one for downloading from the GPU.
  - You can put markers called Events in these command queues to synchronise
    between different ones, or even just to make sure you don't download data
    before it's finished processing.
  - The term is borrowed from OpenCL. In CUDA it's called a Stream but it's the
    same thing.
  - In general if you don't specify one, CUDA will have a "default" one, but
    katsdpsigproc requires you to specify one. This matches with the OpenCL
    model and prevents accidentally submitting to the default CUDA queue.

Event
^^^^^

- The thing used to synchronise GPU command_queues.
- EventItems though, are containers that assist in transferring stuff between
  async coroutines in the main body of the F-engine.

  - They are objects that have GPU memory arrays and event lists associated with
    them.
  - They get populated then put on Queues, so that the next stage in the
    pipeline can wait for the events so that it knows the memory is ready for
    copying.


Common features
---------------

.. todo:: ``NGC-675``
    Explanation of network receive, GPU processing and network transmit "loops".
    There'll be a few merges from the existing F- and XBgpu sections, and the
    Glossary as well.

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
