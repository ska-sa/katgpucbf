GPU DSP engines
===============


.. _gpu-terminology:

Terminology
-----------

.. todo::

    Update katsdpsigproc documentation with proper explanations of terminology
    (if what's there already isn't sufficient) and then remove this paragraph,
    which was lifted from fgpu's earlier documentation.

We will use OpenCL terminology, as it is more generic. An OpenCL *workitem*
corresponds to a CUDA *thread*. Each workitem logically executes the same
program but with different parameters, and can share data through *local
memory* (shared memory in CUDA) with other workitems in the same
*workgroup* (thread block in CUDA).

Glossary
--------

.. todo::

    Update Glossary section.

This section serves (hopefully) to clarify some potentially confusing terms used
within the source code.

Stream
^^^^^^
 - In the katgpucbf.fgpu sense, this is derived from a spead2 recv or send stream.
 - Not to be confused with a CUDA stream, which is here encapsulated as a
   command queue. CUDA streams aren't referred to directly in katgpucbf.fgpu.

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



.. todo::

    Explanation of network receive, GPU processing and network transmit "loops".
    There'll be a few merges from the existing F- and XBgpu sections, and the
    Glossary as well.
