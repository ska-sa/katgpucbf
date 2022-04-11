.. _spead-protocol:

SPEAD protocol
==============

Introduction
------------

The Streaming Protocol for Exchanging Astronomical Data (SPEAD) is a
lightweight streaming protocol, primarily UDP-based, designed for components
of a radio astronomy signal-chain to transmit data to each other over Ethernet
links.

The SPEAD implementation used in :mod:`katgpucbf` is :mod:`spead2`. It is highly
recommended that users of this module (i.e. :mod:`katgpucbf`) also make use of
:mod:`spead2`. For those who cannot, this document serves as a brief summary
of the SPEAD protocol in order to understand the output of each application
within :mod:`katgpucbf`, which are further detailed elsewhere.

SPEAD transmits logical collections of data known as :dfn:`heaps`. A heap
consists of one or more UDP packets. A SPEAD transmitter will decompose a heap
into packets and the receiver will collect all the packets and reassemble the
heap.


Packet Format
-------------

A number of metadata fields are included within each packet, to facilitate heap
reassembly. The SPEAD flavour used in :mod:`katgpucbf` is 64-48, which means that
each metadata field is 64 bits wide, with the first 16 bits carrying the item ID
and the remaining 48 carrying the value.

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

The value contained in the immediate items may change from heap to heap, or it
may be static, with the data payload being the only changing thing, this depends
on the nature of the stream.
