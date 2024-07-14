Beamforming
===========

GPU kernel
----------
For the low number of beams required for MeerKAT (8 single-pol beams), the
beamforming operation is entirely limited by the GPU DRAM bandwidth. There is
thus no benefit to using tensor cores, and potentially significant downsides
(implementation complexity and reduced accuracy for steering coefficients). The
kernel thus uses standard single-precision floating-point operations, and is
written to maximise bandwidth usage.

The beamforming operation consists of two phases: coefficient generation and
coefficient application. To minimise use of system memory, the coefficients
are generated on the fly in the same kernel, rather than computed in a
separate kernel and communicated through global memory. This design allows for
more dynamic coefficients in the future, such as delays that are updated much
more frequently according to a formula.

The external interface to the beamformer has four parameters per beam-antenna
pair: a weight, a quantisation gain (common to all antennas), a delay and a
fringe-rate. All except the delay are combined by the CPU into a single
(complex) weight per beam-antenna pair, and the delay is scaled into a
convenient unit for computing the phase slope. The final coefficient applied
to channel :math:`c`, beam :math:`b`, antenna :math:`a` is

.. math:: W_{abc} = w_{ab} e^{j\pi cd_{ab}}

where :math:`w_{ab}` and :math:`d_{ab}` are the weight and delay values passed
to the kernel.

Each work-group of the kernel handles multiple spectra and all beams and
antennas, but only a single channel. Conceptually, the kernel first computes
:math:`W_{abc}` for all antennas and beams and stores it to local memory, then
applies it to all antennas and beams. Each input sample is loaded once before
it is used for all beams. An accumulator is maintained for each beam. Since
each coefficient is used many times (the number depends on the work-group
size, which is a tuning parameter, but 64-256 is reasonable) after it is
computed, the cost for computing coefficients is amortised.

In practice, this would cause local memory usage to scale without bound as the
number of antennas increases. To keep it bounded (for a fixed number of
beams), the antennas are processed in batches, computing then applying
:math:`W_{abc}` for each batch before starting the next batch. Larger batch
sizes have two advantages:

1. The two phases in each batch need to be separated by a barrier to
   coordinate access to the shared memory. Larger batches reduce the number of
   barriers.

2. If the batch size is small, the number of coefficients to compute is also
   small, and there is not enough work to keep all the work-items busy, making
   the coefficient computation less efficient.

Higher beam counts
^^^^^^^^^^^^^^^^^^
The design above works well for small numbers of beams (up to about 64
single-pol beams), but the register usage scales with the number of beams and
eventually the registers spill to memory, causing very poor performance.

To handle more beams, the kernel batches over beams, just as it does over
antennas. The beam batch loop becomes an outer loop, with the rest of the
kernel operating as before but only on a single batch.

This does mean that the inputs are loaded multiple times, but caches help
significantly here, and the kernel tends to be more compute-bound in this
domain.

Data flow
---------
The host side of the beamforming is simpler than for correlation because
there is no accumulation. For simplicity, the output chunk size (in time) is
set to the same as the input chunk size.

Because the kernel operates on all beams together, there is only one instance
of the :class:`.BPipeline` class, and it handles all the beams. This is in
contrast to :class:`.XPipeline`, which only handles a single :class:`Output`.

Unlike the correlation pipeline, the beamformer pipeline can be controlled
dynamically by setting weights and gains. Transferring these to the GPU is
somewhat expensive and should not be done for every chunk. Instead, there is
an associated version number. When the weights/gains are updated, the master
version number is incremented. Before processing a chunk on the GPU, this
version number is compared to a version number associated with the GPU copy.
If they differ, an update is performed.

Missing data handling
^^^^^^^^^^^^^^^^^^^^^
When some input data is not received, we wish to exclude it from the beam
sums. At present this is handled by zeroing out missing data when it is
received in :meth:`.XBEngine._receiver_loop`. This is potentially expensive if
a lot of data goes missing. Should it prove too expensive, we could zero the
data on the GPU, or pass metadata to the kernel to indicate which values
should not be used.

The presence of heaps is also tracked to allow the ``beam_ants`` SPEAD item
to be populated on transmission. This item is a compromise: ideally we'd like
to indicate exactly which antennas were present, but this would require more
than the 48 bits available in a SPEAD immediate item.
