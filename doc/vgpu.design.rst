V-Engine Design
===============

.. note:: This has not yet been implemented yet, so is still speculative.

The processing is largely handled by
:external+katcbf-vlbi-resample:doc:`katcbf-vlbi-resample <introduction>`. This
means that GPU memory is managed by cupy rather than all allocated up front.
The design is also much simpler than the other engines because we do not try
to overlap CPU-GPU transfers with GPU computations [#]_. The processing steps are
the same as those documented for the :program:`mk_vlbi_resample` script,
except that there is no mixer because the incoming data stream is expected to
already have the correct centre frequency.

To avoid unbounded memory usage if transmission cannot keep up with reception,
a pull model is used. The main loop repeatedly obtains a completed chunk from
an asynchronous iterator then transmits it. That iterator obtains input from
an upstream iterator, which uses another iterator and so on, until eventually
an iterator blocks on being able to get a chunk from the receive queue.

While there is no overlap between GPU work and CPU-GPU transfers, CPU work
(specifically networking) must still be overlapped rather than paused during
transfers: on the receive side we do not want to miss packets arriving, and on
the send side we want to send packets continuously at a steady rate rather
than bursts with gaps between. On the receive side that happens naturally due
to spead2's design, with a separate thread handling reception of packets and
assembly into chunks. On the send side the overlap is created by the
:class:`~katcbf_vlbi_resample.cupy_bridge.AsNumpy` class, which proactively starts transferring chunks from
upstream before they are requested.

.. [#] There is no need to do so, because the data rates are quite low and so
   blocking GPU work during transfers does not significantly impact
   performance.
