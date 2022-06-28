Installation and Operation
==========================

.. todo::

    Make an "installation" section. It'll need to focus on the following points:

    - Installing module using setuptools, if you must
    - But we'd rather you didn't, it's better to use the Dockerfile and deploy
      it that way
    - If you want to poke around at it, then refer to the dev-guide section.
      (There'll need to be a proper cross-reference.)

.. todo::

    Move the following sections here from the fgpu sections:

    - Packet Size
    - Simultaneous receive and transmit
    - Hardware / BIOS configuration
        The NUMA section should go under here, and be accompanied by other
        relevant stuff.

    Update the contents in case something has become outdated. Some things may
    rightly belong in spead2's documentation, and we can make references to it
    here if necessary.


Controlling the Correlator
--------------------------

.. todo::

    If this section gets too much, it can possibly make its way into its own
    ``controlling.rst`` file or some such.

katsdpcontroller
^^^^^^^^^^^^^^^^

.. todo::

    Describe katsdpcontroller, its role, note that the module can be used
    without it and whatever is used in its place will need to implement the
    functionality described in this "chapter".

    Important to note is that we try to make interacting with katsdpcontroller
    as similar as possible compared to interacting with the individual engines,
    for ease of understanding.


Starting the correlator
^^^^^^^^^^^^^^^^^^^^^^^

.. todo::

    Describe how a correlator should be started. Master controller figures out
    based on a set of input parameters, how to invoke a few instances of
    katgpucbf as dsim, fgpu or xbgpu.


Controlling the correlator
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. todo::

    Describe how the correlator is controlled. This will mostly be delays and
    gains. Product controller passes almost identical requests on to relevant
    instances of katgpucbf.


Shutting down the correlator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. todo::

    Describe how to shut the correlator down. Product or master controller
    passes requests on to individual running instances.

Monitoring
^^^^^^^^^^

.. todo::

    - Describe KATCP sensors.
    - Describe Prometheus monitoring capabilities.
    - Probably also a good idea to mention the general logic distinguishing
      between what goes to katcp and what to prometheus.


Data Interfaces
---------------

.. todo::

    If this section gets to be too large, it can probably also make its way into
    its own file.

.. _spead-protocol:

SPEAD protocol
^^^^^^^^^^^^^^

The Streaming Protocol for Exchanging Astronomical Data (`SPEAD`_) is a
lightweight streaming protocol, primarily UDP-based, designed for components
of a radio astronomy signal-chain to transmit data to each other over Ethernet
links.

.. _SPEAD: https://spead2.readthedocs.io/en/latest/_downloads/6160ba1748b1812337d9c7766bdf747a/SPEAD_Protocol_Rev1_2012.pdf

The SPEAD implementation used in :mod:`katgpucbf` is :mod:`spead2`. It is highly
recommended that consumers of :mod:`katgpucbf` output data also make use of
:mod:`spead2`. For those who cannot, this document serves as a brief summary
of the SPEAD protocol in order to understand the output of each application
within :mod:`katgpucbf`, which are further detailed elsewhere.

SPEAD transmits logical collections of data known as :dfn:`heaps`. A heap
consists of one or more UDP packets. A SPEAD transmitter will decompose a heap
into packets and the receiver will collect all the packets and reassemble the
heap.


Packet Format
^^^^^^^^^^^^^

A number of metadata fields are included within each packet, to facilitate heap
reassembly. The SPEAD flavour used in :mod:`katgpucbf` is 64-48, which means that
each metadata field is 64 bits wide, with the first bit indicating the address
mode, the next 15 carrying the item ID and the remaining 48 carrying the value
(in the case of immediate items).

Each packet contains the following metadata fields:

``header``
  Contains information about the flavour of SPEAD being used.

``heap counter/id``
  A unique identifier for each new heap.

``heap size``
  Size of the heap in bytes.

``heap offset``
  Address in bytes indicating the current packet's location within the heap.

``payload size``
  Number of bytes within the current packet payload.


Each SPEAD stream will have additional 64-bit fields specific to itself,
referred to in SPEAD nomenclature as :dfn:`immediate items`. Each packet
transmitted will contain all the immediate items to assist third-party consumers
that prefer to work at the packet level (see
:attr:`spead2.send.Heap.repeat_pointers` — note that this is not default spead2
behaviour, but it is always enabled in katgpucbf).

Most of the metadata remains constant for all packets in a heap. The heap offset
changes across packets, in multiples of the packet size (which is configurable
at runtime). This is used by the receiver to reassemble packets into a full heap.

The values contained in the immediate items may change from heap to heap, or
they may be static, with the data payload being the only changing thing,
depending on the nature of the stream.

.. todo::

    Consolidate ``fgpu.networking`` and ``xbgpu.networking`` (i.e. input and
    output packet format sections) here.
