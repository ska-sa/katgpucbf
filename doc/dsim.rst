Digitiser Packet Simulator
==========================

The digitiser simulator (:dfn:`dsim` for short) is a tool that provides the
same heap format as the MeerKAT digitisers, with a configurable payload.

Usage
-----
The dsim process generates an arbitrary number of single-pol data streams.
However, it only uses a single sending thread, so in practice it does not
scale well beyond two streams (a dual-pol antenna) for typical MeerKAT
bandwidths. Instead, one can use multiple instances.
It also relies heavily on the ibverbs support in spead2 for performance at
typical MeerKAT bandwidths. It can nevertheless be used without it, but the
bandwidth will most likely need to be reduced. Pass :option:`!--ibv` to
use the ibverbs acceleration.

When using multiple processes, it is usually necessary to synchronise them.
The :option:`!--sync-time` specifies a time (in the past) that will correspond
to a zero timestamp. The synchronisation is accurate to about a millisecond,
provided that all threads are pinned to specific CPU cores and real-time
scheduling is used to prevent other tasks from sharing those cores. Streams
sent by the same process are interleaved in a single transmit queue, so will
be perfectly synchronised as they leave the NIC (but could be desynchronised
by a multi-path network).

By default the content of the signal is a sine wave with a fixed frequency.
However, the signal is highly configurable with the :option:`!--signals`
option. A domain-specific language (DSL) allows continuous waves and Gaussian
noise to be combined with basic operators. See :func:`.parse_signal` for a
specification of this DSL. As an example, the following specification [#nl]_
defines two signals which share a sine wave and some noise, and adds further
noise that is uncorrelated between the polarisations:

.. code::

   base = cw(1.0, 1e9) + wgn(0.1);
   base + wgn(0.05);
   base + wgn(0.05);

The signals to send can also be changed on the fly by issuing the ``?signals``
command over katcp.

.. [#nl] While shown split over multiple lines, whitespace is not significant
   and it may be easier to place it all on one line.

Design
------

Signal generation
~~~~~~~~~~~~~~~~~
It would be extremely challenging for a CPU to simulate a signal in real-time,
particularly given the need to pack the results into 10-bit samples. Instead,
a window of signal is generated on startup, or on request to change the
signal, and then replayed over and over. The length of this window is
determined by the :option:`!--signal-heaps` command-line option.
This has a few implications:

1. The frequency resolution is limited by the inverse of the window length.
   For example, a sinusoidal signal must have an integer number of cycles per
   window, which means that the frequency is rounded to a multiple of
   :math:`\frac{\text{adc-sample-rate}}{\text{signal-heaps}\times \text{heap-samples}}`.

2. Noise is correlated in time, and when averaging over long periods of time
   (longer than the window), the standard deviation does not decrease with the
   square root of the integration time. Similarly, the sample mean converges
   to the mean of the generated window rather than the population mean.

To speed up the signal generation, `dask`_ is used to parallelise the process
across multiple CPU cores. Dask presents a numpy-like interface, but
internally splits arrays into chunks and performs computations for each chunk
in parallel. The chunk size is determined by the constant
:data:`katgpucbf.dsim.signal.CHUNK_SIZE`.

Generating reproducible random signals needs to be done carefully when
parallelising. The given random seed is first used to produce a
:class:`~numpy.random.SeedSequence` for each chunk, and each chunk then uses
an independent generator seeded with its corresponding sequence. This ensures
that different instances of the simulator will produce the same sequence given
the same entropy (hence giving correlated noise). Note that the result is
dependent on the chunk size.

.. _dask: https://dask.org/

Transmission
~~~~~~~~~~~~
Most of the heavy work of transmission is handled by spead2. To minimise
overheads, the heaps are pre-defined, and put into a
:class:`spead2.send.HeapReferenceList` for bulk transmission with
:meth:`spead2.send.asyncio.AsyncStream.async_send_heaps`. Additionally,
spead2's rate limiting is used to control the simulated digitiser clock
speed. Since spead2 sends data in small bursts (64 KiB) between sleeps, the
delivery of packets will not be as smooth as from a real MeerKAT digitiser.

To avoid stalling transmission, it is important that spead2's C++ worker
thread always has more data to send, as the latency of signalling
end-of-transmission to Python and then waiting for Python to respond with new
heaps would be significant. To accommodate this, the window is split in half,
and each call to spead2 sends only half the window. As soon as one half
finishes transmission, the Python code prepares it to be sent again, in
parallel with spead2 starting transmission of the other half.

Although the signal is recycled, some work is still needed to prepare a
half-window for retransmission, because the timestamps need to be updated. To
make this as efficient as possible, all the timestamps are allocated in a
single numpy array, and each heap references the appropriate entry of the
array. This allows a range of timestamps to be updated with a single numpy
operation, rather than a Python loop.

Allowing the signal to be changed mid-flow is done with double-buffering. The
new signal is computed asynchronously into a spare second window. Once that's
completed, the spare and active windows are swapped. The new spare window may
still be referenced by in-flight heaps, so it is necessary to await
transmission of those heaps before allowing the signal to be changed again.
