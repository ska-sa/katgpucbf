Data Interfaces
===============

.. todo::

    If this section gets to be too large, it can probably also make its way into
    its own file.

.. _spead-protocol:

SPEAD Protocol
--------------

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

.. _spead-packet-format:

Packet Format
-------------

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

F-Engine Data Format
--------------------

Input
^^^^^
The F-engine receives dual-polarisation input from a digitiser (raw antenna)
stream. In MeerKAT and MeerKAT Extension, each polarisation's raw digitiser data
is distributed over eight contiguous multicast addresses, to facilitate load-
balancing on the network, but the receiver is flexible enough to accept input
from more or fewer multicast addresses.

The only immediate item in the digitiser's output heap used by the F-engine is
the ``timestamp``.

Output Packet Format
^^^^^^^^^^^^^^^^^^^^

In addition to the fields described in SPEAD's :ref:`spead-packet-format`
above, the F-Engine's have an output data format as follows - formally
labelled elsewhere as **Channelised Voltage Data SPEAD packets**.
These immediate items are specific to the F-Engine's output stream.

``timestamp``
  A number to be scaled by an appropriate scale factor,
  provided as a KATCP sensor, to get the number of Unix
  seconds since epoch of the first time sample used to
  generate data in the current SPEAD heap.

``feng_id``
  Uniquely identifies the F-engine source for the data.
  A sensor can be consulted to determine the mapping of
  F-engine to antenna antenna input. The X-engine uses
  this field to distinguish data received from multiple
  F-engines.

``frequency``
  Identifies the first channel in the band of frequencies in
  the SPEAD heap. Can be used to reconstruct the full spectrum.
  Although each packet may represent a different frequency,
  this value remains constant across a heap and represents
  only the first frequency channel in the range of channels
  within the heap. The X-engine does not strictly need this
  information.

``feng_raw item pointer``
  Channelised complex data from both polarisations of
  digitiser associated with F-engine. Real comes before
  imaginary and input 0 before input 1. A number of
  consecutive samples from each channel are in the same
  packet.

The F-engines in an array each transmit a subset of frequency channels to each
X-engine, with each X-engine receiving from a single multicast group. F-engines
therefore need to ensure that their heap IDs do not collide.

X-Engine Data Format
---------------------

Input
^^^^^
The X-Engine receives antenna channelised data from the output of the F-engines,
as discussed above. Each X-Engine receives data from each F-engine, but only
from a subset of the channels.

Output Packet Format
^^^^^^^^^^^^^^^^^^^^

In addition to the fields described in SPEAD's :ref:`spead-packet-format` above,
the X-Engine's have an output data format as follows - formally labelled
elsewhere as **Baseline Correlation Products**. These immediate items are
specific to the X-Engine's output stream.

``frequency``
  Identifies the first channel in the band of frequencies
  in the SPEAD heap. Although each packet represents a
  different frequency, this value remains constant across
  a heap and represents only the first frequency channel
  in the range of channels within the heap.

``timestamp``
  A number to be scaled by an appropriate scale factor,
  provided as a KATCP sensor, to get the number of Unix
  seconds since epoch of the first time sample used to
  generate data in the current SPEAD heap.

``xeng_raw item pointer``
  Integrated Baseline Correlation Products; packed in an order
  described by the KATCP sensor :samp:`{xeng-stream-name}-bls-ordering`.
  Real values are before imaginary. The bandwidth and centre
  frequencies of each sub-band are subject to the granularity
  offered by the X-engines.

In MeerKAT Extension, four correlation products are computed for each baseline,
namely vv, hv, vh, and hh. Thus, for an 80-antenna correlator, there are
:math:`\frac{n(n+1)}{2} = 3240` baselines, and 12960 correlation products. The
parameter ``n-bls`` mentioned under ``xeng_raw`` refers to the latter figure.

Each X-engine sends data to its own multicast group. A receiver can combine data
from several multicast groups to consume a wider spectrum, using the
``frequency`` item to place each heap. To facilitate this, X-engine output heap
IDs are kept unique across all X-engines in an array.
