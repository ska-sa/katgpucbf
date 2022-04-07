.. _baseline-correlation-products-data-packet-format:

Packet Format
=============

According to the **MeerKAT M1000-0001 CBF-Data Subscribers ICD (M1200-0001-020)**,
the Baseline Correlation Products SPEAD packets have the following data format:

.. figure:: images/xeng_spead_heap_format_table.png

  SPEAD packet format output by an X-Engine

In MeerKAT Extension, four correlation products are computed for each canonical
baseline, namely vv, hv, vh, and hh. Thus, for an 80-antenna correlator, there are
:math:`\frac{n(n+1)}{2} = 3240` baselines, and 12960 correlation products. The parameter
``n-bls`` in the above table refers to the latter figure.

Each correlation product contains a real and imaginary sample (both 32-bit
integer) for a combined size of 8 bytes per baseline. The ordering of the
correlation products is given in the :samp:`{xeng-stream-name}-bls-ordering` sensor in
the product controller, but can be calculated deterministically:
:func:`~katgpucbf.xbgpu.correlation.get_baseline_index` indicates the ordering
of the baselines, and the four individual correlation products are always
ordered ``vv, hv, vh, hh``.

All the baselines for a single channel are grouped together contiguously in the
heap, and each X-engine correlates a contiguous subset of the entire spectrum.
For example, in an 80-antenna, 8192-channel array with 64 X-engines, each X-engine output
heap contains 8192/64 = 128 channels.

The heap payload size in this example is equal to

  channels_per_heap * correlation_products * complex_sample_size = 128 * 12960 * 8 = 13,271,040 bytes or 12.656 MiB.

The SPEAD format assigns a number of metadata fields to each packet. Each metadata
field is 64 bits/8 bytes wide. More information on these fields is listed in the
`SPEAD specification`_. The metadata fields are as follows:

.. _SPEAD specification: https://casper.ssl.berkeley.edu/astrobaki/images/9/93/SPEADsignedRelease.pdf

``header``
  Contains information about the flavour of SPEAD being used.

``heap counter/id``
  A unique identifier that increments for each new heap.

``heap size``
  Size of the heap in bytes.

``heap offset``
  Address in bytes indicating the current packets location within the heap.

``payload size``
  Number of bytes within the current packet payload.

In addition, each packet has a number of additional 64-bit fields specific
to this heap that are used for reassembling the packets.

``frequency`` (See above table)
  Although each packet represents a different frequency,
  this value remains constant across a heap and represents
  only the first frequency channel in the range of
  channels within the heap.

``timestamp`` (See above table)
  .. comment just to get this formatted as definition list

``xeng_raw item pointer`` (See above table)
  .. comment just to get this formatted as definition list

Most of the above fields remain constant for all packets in a heap. The heap
offset changes across packets. Heap offset is expected to change in multiples
of the packet size (which is configurable at runtime). This is used by the
receiver to reassemble packets into a full heap.
