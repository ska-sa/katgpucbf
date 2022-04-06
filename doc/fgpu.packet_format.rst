.. _channelised-voltage-data-packet-format:

Packet Format
=============

According to the **MeerKAT M1000-0001 CBF-Data Subscribers ICD (M1200-0001-020)**,
the Channelised Voltage Data SPEAD heaps have the following data format:

.. figure:: images/feng_spead_heap_format_table.png

  Table indicating SPEAD heap format output by F-Engine

In the case of a 80A 8k array with 64 X-engines, each heap contains 8192/(64) =
128 channels. By default, there are 256 time samples per channel. Each sample is
dual-pol complex 8-bit data for a combined sample width of 32 bits or 4 bytes.

The heap payload size in this example is equal to

    channels_per_heap * samples_per_channel * complex_sample_size = 128 * 256 * 4 = 131,072 = 128 KiB.

The SPEAD protocol assigns a number of metadata fields to each packet. Each metadata
field is 64 bits/8 bytes wide. More information on these fields is listed in the
`SPEAD specification`_. The metadata fields are as follows:

.. _SPEAD specification: https://casper.ssl.berkeley.edu/astrobaki/images/9/93/SPEADsignedRelease.pdf

``header``
  Contains information about the flavour of SPEAD being used.

``heap counter/id``
  A unique identifier for each heap.

``heap size``
  Size of the heap in bytes.

``heap offset``
  Address in bytes indicating the current packet's location within the heap.

``payload size``
  Number of bytes within the current packet payload.

In addition, each packet has a number of additional 64-bit fields specific
to this heap. The fields are as follows:

``timestamp`` (See above table)
  .. comment just to get this formatted as definition list

``feng_id`` (See above table)
  .. comment just to get this formatted as definition list

``frequency`` (See above table)
  Although each packet may represent a different frequency,
  this value remains constant across a heap and represents
  only the first frequency channel in the range of
  channels within the heap.

``feng_raw item pointer`` (See above table)
  .. comment just to get this formatted as definition list

Most of the above fields remain constant for all packets in a heap.
The heap offset changes across packets. Heap offset is expected to change in
multiples of the packet size (which is configurable at runtime). This is used by
the receiver to reassemble packets into a full heap.
