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

Transmission
------------
While the other engines use spead2 to send SPEAD data, the output of the
V-Engine is `VDIF`_ and so we are not able to use the high-speed kernel bypass
and packet pacing capabilities of spead2. Instead, packet pacing is
re-implemented in Python, following essentially the same
:external+spead2:doc:`design <dev-send-rate-limit>` as used by spead2. There
are a few changes to specialise things to the use case:

1. When the time to sleep is less than a threshold (1ms at the time of
   writing), we omit the sleep, as the wakeup overheads can be quite high in
   Python and cause significant overhead.
2. Instead of buffering up packets to a given burst size, we treat each
   frameset (group of frames with the same timestamp but different thread IDs)
   as a burst that is transmitted without intervening sleeps. This will
   typically create smaller such bursts than the default in spead2, but
   combined with the point above the actual number of bytes between bursts can
   be quite large.
3. The burst (catch-up) rate is set significantly higher than the default in
   spead2, to compensate for potentially long pauses due to both Python's
   stop-the-world garbage collector and to asyncio multiplexing work onto a
   single kernel thread rather than having a dedicated thread for
   transmission.

Initially we tried to perform transmission serially with the iterator over the
processed frames, on the assumption that the asynchronous buffering in
:class:`~katcbf_vlbi_resample.cupy_bridge.AsNumpy` would allow GPU work to
proceed in parallel with data transmission. However, we found that this did
not work well, as some requests for the next frame would block for hundreds of
milliseconds, during which no packets were being transmitted. Instead,
:class:`.VDIFSender` uses a queue of packets and a background task to service
them concurrently with data processing.

.. _VDIF: https://vlbi.org/vlbi-standards/vdif/
