F-Engine Design
===============

.. todo::  ``NGC-675``
   Most of this needs to be folded into the higher-level GPU "Design" document.
   Whatever remains will probably need re-naming under and "F-engine" sub-
   heading or some such.

The actual GPU kernels are reasonably straight-forward, because they're
generally memory-bound rather than compute-bound. The main challenges are in
data movement through the system.

.. tikz:: Data Flow. Double-headed arrows represent data passed through a
   queue and returned via a free queue.
   :libs: chains

   \tikzset{proc/.style={draw, rounded corners, minimum width=4.5cm, minimum height=1cm},
            pproc/.style={proc, minimum width=2cm},
            flow/.style={->, >=latex, thick},
            queue/.style={flow, <->},
            fqueue/.style={queue, color=blue}}
   \node[proc, start chain=going below, on chain] (align) {Align, copy to GPU};
   \node[pproc, draw=none, anchor=west,
         start chain=rx0 going above, on chain=rx0] (align0) at (align.west) {};
   \node[pproc, draw=none, anchor=east,
         start chain=rx1 going above, on chain=rx1] (align1) at (align.east) {};
   \node[proc, on chain] (process) {GPU processing};
   \node[proc, on chain] (download) {Copy from GPU};
   \node[proc, on chain] (transmit) {Transmit};
   \node[pproc, draw=none, anchor=west,
         start chain=tx0 going below, on chain=tx0] (transmit0) at (transmit.west) {};
   \node[pproc, draw=none, anchor=east,
         start chain=tx1 going below, on chain=tx1] (transmit1) at (transmit.east) {};
   \foreach \i in {0, 1} {
     \node[pproc, on chain=rx\i] (receive\i) {Receive};
     \node[pproc, on chain=rx\i] (stream\i) {Stream};
     \node[pproc, on chain=tx\i] (outstream\i) {Stream};
   }
   \foreach \i in {0, 1} {
     \draw[flow] (stream\i) -- (receive\i);
     \draw[queue] (receive\i) -- (align\i);
     \draw[flow] (transmit\i) -- (outstream\i);
   }
   \draw[queue] (align) -- (process);
   \draw[queue] (process) -- (download);
   \draw[queue] (download) -- (transmit);

Chunking
--------
GPUs have massive parallelism, and to exploit them fully requires large batch
sizes (millions of elements). To accommodate this, the input packets are
grouped into "chunks" of fixed numbers of samples. There is a tradeoff in the
chunk size: large chunks use more memory, add more latency to the system, and
reduce LLC (last-level cache) hit rates. Smaller chunks limit parallelism, and
as will be seen later, increase the overheads associated with overlapping PFB
(polyphase filter bank) windows.

Chunking also helps reduce the impact of slow Python code. Digitiser heaps
consist of only a single packet, and involving Python on a per-heap basis
would be far too slow. We use :class:`spead2.recv.ChunkRingStream` to group
heaps into chunks, which means Python code is only run per-chunk.

Queues
------
The system consists of several components which run independently of each
other - either via threads (spead2's C++ code) or Python's asyncio framework. The
general pattern is that adjacent components are connected by a pair of queues:
one carrying full buckets of data forward, and one returning free data. This
approach allows all memory to be allocated up front. Slow components thus
cause back-pressure on up-stream components by not returning buckets through
the free queue fast enough. The number of buckets needs to be large enough to
smooth out jitter in processing times.

Network receive
---------------
Each polarisation is handled as a separate SPEAD stream, with a separate
thread. Separate threads are necessary
because a single core is not fast enough to load the data. This introduces
some challenges in aligning the polarisations, because locking a shared
structure on every packet would be prohibitively expensive. Instead, the
polarisations are kept separate during chunking, and aligned afterwards (in
Python). A chunk is buffered until the matching chunk is received on the
other polarisation. Alternatively, if a later chunk is seen for the other
polarisation, then the chunk can never match and is discarded.

To minimise the number of copies, chunks are initialised with CUDA pinned
memory (host memory that can be efficiently copied to the GPU).
Alternatively, it is possible to use `vkgdr`_ to have the CPU write directly
to GPU memory while assembling the chunk. This is not enabled by default
because it is not always possible to use more than 256 MiB of the GPU memory
for this, which can severely limit the chunk size.

.. _vkgdr: https://github.com/ska-sa/vkgdr

GPU Processing
--------------


Narrowband
^^^^^^^^^^
In narrow-band modes, the first step is a down-conversion filter that produces
a new sample stream with a lower bandwidth and sampling rate. The kernel
implementing this is particularly complex, and is discussed separately in
`fgpu.ddc`_.

.. note::

   At the time of writing, the kernel has been written but the full narrowband
   implementation is not yet implemented.

Decode
^^^^^^
Digitiser samples are 10-bit and stored compactly. While it is possible to
write a dedicated kernel for decoding that makes efficient accesses to memory
(using contiguous word-size loads), it is faster overall to do the decoding as
part of the PFB filter because it avoids a round trip to memory. For the
PFB, the decode is done in a very simple manner:

 1. Determine the two bytes that hold the sample.
 2. Load them and combine them into a 16-bit value.
 3. Shift left to place the desired 10 bits in the high bits.
 4. Shift right to sign extend.
 5. Convert to float.

While many bytes get loaded twice (because they hold bits from two samples),
the cache is able to prevent this affecting DRAM bandwidth.

The narrowband digital down conversion also decodes the 10-bit samples, but this
is discussed :ref:`separately <ddc-load>`.

Polyphase Filter Bank
^^^^^^^^^^^^^^^^^^^^^
The polyphase filter bank starts with a finite impulse response (FIR) filter,
with some number of *taps* (e.g., 16), and a *step* size which is twice the
number of output channels. This can be thought of as organising the samples as
a 2D array, with *step* columns, and then applying a FIR down each column.
Since the columns are independent, we map each column to a separate workitem,
which keeps a sliding window of samples in its registers. GPUs generally don't
allow indirect indexing of registers, so loop unrolling (by the number of
taps) is used to ensure that the indices are known at compile time.

This might not give enough parallelism, particularly for small channel counts,
so in fact each column in split into sections and a separate workitem is used
for each section. There is a trade-off here as samples at the boundaries
between sections need to be loaded by both workitems, leading to overheads.

Registers are used to hold both the sliding window and the weights, which
leads to significant register pressure. This reduces occupancy and leads to
reduced performance, but it is still good for up to 16 taps. For higher tap
counts it would be necessary to redesign the kernel.

The weights are passed into the kernel as a table, rather than computed on the
fly. While it may be possible to compute weights on the fly, using single
precision in the computation would reduce the accuracy. Instead, we compute
weights once on the host in double precision and then convert them to
single precision.

A single FIR may also need to cross the boundary between chunks. To handle
this, we allocate sufficient space at the end of each chunk for the PFB
footprint, and copy the start of the next chunk to the end of the current one.
Note that this adds an extra chunk worth of latency to the process.

.. _fgpu-fft:

FFT
^^^
After the FIR above, we can perform the FFT, which is done with cuFFT. The
built-in support for doing multiple FFTs at once means that it can saturate
the GPU even with small channel counts.

Naïvely using cuFFT for the full real-to-complex transformation can be quite
slow and require multiple passes over the memory, because

1. There is a maximum number of channels that cuFFT can handle in one pass (it
   depends on the GPU, but seems to be 16384 for a GeForce RTX 3080 Ti).
   Larger transforms require at least one more pass.

2. It appears to handle real-to-complex transforms by first doing a
   complex-to-complex transform and then using an additional pass to fix up
   the result (i.e. form final FFT output).

For performance reasons, we move part of the Fourier Transform into the
post-processing kernel (using the Cooley-Tukey algorithm), and also handle
fixing up the real-to-complex transformation. This is achieved by decomposing
transformation into separately computed smaller parts (this is where Cooley-Tukey
comes in). Part of the Fourier Transform is computed using cuFFT in the compute
kernel and the rest of the process moves into the post-processing
kernel. The computation to form the the real-to-complex transformation takes place here.

Real-to-complex transform
~~~~~~~~~~~~~~~~~~~~~~~~~
To start, let's consider the tradition equation for the Fourier Transform. Let :math:`n`
be the number of channels we wish to decompose the input sequence into and let
:math:`x_i` be the (real) time-domain samples and :math:`X_k` be the discrete Fourier transform (DFT)
(the reasons for dividing by :math:`2n` rather than :math:`n` will be discussed shortly
but simply put it is due to the number of input samples used).

.. math:: X_k = \sum_{i=0}^{2n} e^{\frac{-2\pi j}{2n}\cdot ik} x_i.

We know that a direct implementation of the DFT is inefficient and alternative more efficient
means exist to perform this computation. One such method is the FFT introduce by Cooley-Tukey
and in the GPU space the cuFFT is one such implemetation. As highlighted earlier, for transform sizes
of greater than 16384 (for a GeForce RTX 3080 Ti at least) requires more than one memory pass making it
less efficient than it needs to be. The technique detailed below uses the decomposition as provided by
Cooley-Tukey to breakdown a larger transform into smaller 'sub-transforms' where each 'sub-transform'
is intentionally kept smaller for effiecincy reasons and later combined (same process as the FFT) to form
the larger transform size. This is a multi-step process and requires some extra notation and math tricks.

Now for some notation to see how this works. let :math:`z` be the array of samples of length :math:`2n`.
The extra length is due to the way in which the samples are used in array :math:`z`. In this case the
original input sequence is a real-valued array but for the purposes of this computational sequence lets
pretend it is a complex-valued array (i.e. every even-indexed sample is real-valued and every
odd-indexed sample is the imaginary sample). The initial array was said to be made up of :math:`x_i` samples
but lets break that into the even and odd indexed samples.

Let :math:`u_i = x_{2i}` and :math:`v_i = x_{2i+1}` be the even and odd input samples, with Fourier
transforms :math:`U` and :math:`V` respectively. By pretending the data is complex, we're computing
the Fourier transform of :math:`z = u + jv`, namely :math:`U + jV`. Let's call it :math:`Z`.

It is important to remember that both :math:`u` and :math:`v` are real-valued, so :math:`U`
and :math:`V` are Hermitian symmetric. By re-arranging things we can reconstruct :math:`U` and
:math:`V` from :math:`Z` by using Hermetian symmetry properties. Now for a nifty trick: Let :math:`Z'`
be :math:`Z` with reversed indices i.e., :math:`Z'_k = Z_{-k}` where indices are taken
modulo :math:`n`. Taking this once step further then we can say :math:`Z'_k = Z_{-k} = \overline{Z}` where
the 'overline' in :math:`\overline{Z}` denotes conjugation. This is effectively saying that by taking the
reverse indices in :math:`Z_k` we get a conjugated result.

Why is this so? Going back to the original definition for the DFT we saw the complex exponential
:math:`e^{\frac{-2\pi j}{2n}\cdot ik}` has a variable :math:`k` where :math:`k` represents the
frequency component under computation for the input sequence :math:`x_i`. If :math:`k` is reveresed
(i.e. negative) the complex exponential changes to:

.. math::
   e^{\frac{-2\pi j}{2n}\cdot i(-k)}
   = e^{\frac{2\pi j}{2n}\cdot i(k)}

So by reversing the index has the same effect as taking the conjugate. Applying this to the :math:`U`
and :math:`V` components then :math:`U' = \overline{U}, V' = \overline{V}`. Going full circle (for completeness)
if one was to conjugate the reversal (i.e. :math:`\overline{U'}`) then since :math:`U' = \overline{U}`,
then :math:`\overline{U'} = \overline{\overline{U}} = U`.

Why is this important?


.. math:: Z + \overline{Z'} = (U + \overline{U'}) + j(V - \overline{V'}) = 2U.


Thus, :math:`U = \frac{Z + \overline{Z'}}{2}` and similarly
:math:`V = \frac{Z - \overline{Z'}}{2j}`. Next, we use the Cooley-Tukey
transform to construct :math:`X` from :math:`U` and :math:`V`. We can also
re-use some common expressions by computing :math:`X_{n-k}` at the same time:

.. math::

   X_k &= \sum_{i=0}^{2n-1} e^{\frac{-2\pi j}{2n}\cdot ik} x_i\\
       &= \sum_{i=0}^{n-1} e^{\frac{-2\pi j}{2n}\cdot 2ik} u_i +
          \sum_{i=0}^{n-1} e^{\frac{-2\pi j}{2n}\cdot (2i+1)k} v_i\\
       &= \sum_{i=0}^{n-1} e^{\frac{-2\pi j}{n}\cdot ik} u_i +
          e^{\frac{-\pi j}{n}\cdot k}\sum_{i=0}^{n-1} e^{\frac{-2\pi j}{n}\cdot ik} v_i\\
       &= U_k + e^{\frac{-\pi j}{n}\cdot k} V_k.\\
   X_{n-k} &= U_{n-k} + e^{\frac{-pi j}{n}\cdot (n-k)} V_{n-k}\\
           &= \overline{U_k} - \overline{e^{\frac{-\pi j}{n}\cdot k} V_k}.

Note that when :math:`k = 0` or :math:`k = \frac{n}{2}` the second calculation
is unnecessary, but it is simpler (and probably cheaper) to just do it anyway
rather than branch to avoid it.

Why is doing all this work more efficient that letting cuFFT handle the
real-to-complex transformation? After all, cuFFT most likely does this (or
something equivalent) internally. The answer is that instead of using a
separate kernel for it (which would consume memory bandwidth), we built it
into the postprocessing kernel (see the next section).

Unzipping the FFT
~~~~~~~~~~~~~~~~~
From here we'll assume all transforms are complex-to-complex unless specified
otherwise. The Cooley-Tukey algorithm allows a transform of size :math:`c =
mn` to be decomposed into :math:`n` transforms of size :math:`m` followed by
:math:`m` transforms of size :math:`n`. We'll refer to :math:`n` as the
"unzipping factor". We will keep it small (typically not more than 4), as the
implementation requires registers proportional to this factor.

To recap the Cooley-Tukey algorithm: let a time-domain index :math:`i` be
written as :math:`qn + r` and a frequency-domain index :math:`k` be
written as :math:`pm + s`. Let :math:`z^r` denote the array :math:`z_r, z_{n+r},
\dots, z_{(m-1)n+r}`, and denote its Fourier transform by :math:`Z^r`. Then

.. math::

   Z_k = Z_{pm+s}
   &= \sum_{i=0}^{mn - 1} e^{\frac{-2\pi j}{mn}\cdot ik} z_i\\
   &= \sum_{q=0}^{m - 1}\sum_{r=0}^{n-1}
      e^{\frac{-2\pi j}{mn}(qn + r)(pm + s)} z_{qn + r}\\
   &= \sum_{r=0}^{n-1} e^{\frac{-2\pi j}{n}\cdot rp} \left[e^{\frac{-2\pi j}{mn}\cdot rs}
      \sum_{q=0}^{m-1} e^{\frac{-2\pi j}{m}\cdot qs} z^r_q\right]\\
   &= \sum_{r=0}^{n-1} e^{\frac{-2\pi j}{n}\cdot rp}
      \left[e^{\frac{-2\pi j}{mn}\cdot rs} Z^r_s\right].

The whole expression is a Fourier transform of the expression in brackets
(the exponential inside the bracket is the so-called "twiddle factor").

An inconvenience of this structure is that :math:`z^r` is not a contiguous
set of input samples, but a strided array. While cuFFT does support both
strided inputs and batched transformations, we cannot batch over :math:`r`
and over multiple spectra at the same time as it only supports a single batch
dimension with corresponding stride. We solve this by modifying the PFB kernel
to reorder its output such that each :math:`z^r` is output contiguously. This
can be done by shuffling some bits in the output index (because we assume
powers of two everywhere).

In the post-processing kernel, each work-item computes the results for a
single :math:`s` and for all :math:`p`. To compute the real-to-complex
transformation, it also needs to compute

.. math::

   \overline{Z_{-k}} = \overline{Z_{-pm - s}}
   = \sum_{r=0}^{n-1} e^{\frac{-2\pi j}{n}\cdot rp}
     \left[e^{\frac{-2\pi j}{mn}\cdot rs} \overline{Z^r_{-s}}\right].

Postprocessing
^^^^^^^^^^^^^^
The remaining steps are to

 1. Compute the real Fourier transform from several complex-to-complex
    transforms (see the previous section).
 2. Apply gains and fine delays.
 3. Do a partial transpose, so that *spectra-per-heap* (256 by default) spectra
    are stored contiguously for each channel (the Nyquist frequencies are also
    discarded at this point).
 4. Convert to int8.
 5. Interleave the polarisations.

These are all combined into a single kernel to minimise memory traffic. The
katsdpsigproc package provides a template for transpositions, and the other
operations are all straightforward. While C++ doesn't have a convert with
saturation function, we can access the CUDA functionality through inline PTX
assembly (OpenCL C has an equivalent function).

Fine delays and the twiddle factor for the Cooley-Tukey transformation are
computed using the ``sincospi`` function, which saves both a multiplication by
:math:`\pi` and a range reduction.

Coarse delays
^^^^^^^^^^^^^
One of the more challenging aspects of the processing design was the handling
of delays. In the end we chose to exploit the fact that the expected delay
rates are very small, typically leading to at most one coarse delay change per
chunk. We thus break up each chunk into sections where the coarse delay is
constant for both polarisations.

Our approach is based on inverting the delay model: output timestamps are
regularly spaced, and for each output spectrum, determine the sample in the
input that will be delayed until that time (to the nearest sample). We then
take a contiguous range of input samples starting from that point to use in
the PFB. Unlike the MeerKAT FPGA F-engine, this means that every output
spectrum has a common delay for all samples. There will also likely be
differences from the MeerKAT F-engine when there are large discontinuities in
the delay model, as the inversion becomes ambiguous.

The polarisations are allowed to have independent delay models. To accommodate
different coarse delays, the space at the end of each chunk (to which the start
of the following chunk is copied to accommodate the PFB footprint) is expanded,
to ensure that as long as one polarisation's input starts within the chunk
proper, both can be serviced from the extended chunk. This involves a tradeoff
where support for larger differential delays requires more memory and more
bandwidth. The dominant terms of the delay are shared between polarisations,
and the differential delay is expected to be extremely small (tens of
nanoseconds), so this has minimal impact.

The GPU processing is split into a front-end and a back-end: the front-end
consists of just the PFB FIR, while the backend consists of FFT and
post-processing. Because changes in delay affect the ratio of input samples to
output spectra, the front-end and back-end may run at different cadences. We
run the front-end until we've generated enough spectra to fill a back-end
buffer, then run the back-end and push the resulting spectra into a queue for
transmission. It's important to (as far as possible) always run the back-end
on the same amount of data, because cuFFT bakes the number of FFTs into its
plan.

Transfers and events
^^^^^^^^^^^^^^^^^^^^
To achieve the desired throughput it is necessary to overlap data transfers
with computations. Transfers are done using separate command queues, and an
CUDA/OpenCL event is associated with the completion of each transfer. Where
possible, these events are passed to the device to be waited for, so that the
CPU does not need to block. The CPU does need to wait for host-to-device
transfers before putting the buffer onto the free queue, and for
device-to-host transfers before transmitting results, but this is deferred as
long as possible.

Network transmit
----------------
The current transmit system is quite simple. A single spead2 stream is created,
with one substream per multicast destination. For each output chunk, memory
together with a set of heaps is created in advance. The heaps are carefully
constructed so that they reference numpy arrays (including for the timestamps),
rather than copying data into spead2. This allows heaps to be recycled for new
data without having to create new heap objects.

PeerDirect
^^^^^^^^^^
When GPUDirect RDMA / PeerDirect is used, the mechanism is altered slightly to
eliminate the copy from the GPU to the host:

1. Chunks no longer own their memory. Instead, they use CUDA device pointers
   referencing the memory stored in an OutItem. As a result, Chunks and
   OutItems are tied tightly together (each OutItem holds a reference to the
   corresponding Chunk), instead of existing on separate queues.

2. Instead of OutItems being returned to the free queue once the data has been
   copied to the host, they are only returned after the data they hold has
   been fully transmitted.

3. More OutItems are allocated to compensate for the increased time required
   before an OutItem can be reused. This has not yet been tuned.

There may be opportunities for further optimisation, in the sense of reducing
the amount of memory that is not actively in use, because some parts of an
OutItem can be recycled sooner than others. Since GPUs that support this
feature tend to have large amounts of memory, this is not seen as a priority.

Output Heap Payload Composition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the case of an 8192-channel array with 64 X-engines, each heap contains 8192/64 =
128 channels. By default, there are 256 time samples per channel. Each sample is
dual-pol complex 8-bit data for a combined sample width of 32 bits or 4 bytes.

The heap payload size in this example is equal to

    channels_per_heap * samples_per_channel * complex_sample_size = 128 * 256 * 4 = 131,072 = 128 KiB.

The payload size defaults to a power of 2, so that packet boundaries in a heap
align with channel boundaries. This isn't important for the :mod:`spead2`
receiver used in the X-engine, but it may be useful for potential third party
consumers of F-engine data.

Missing data handling
---------------------
Inevitably some input data will be lost and this needs to be handled. The
approach taken is that any output heap which is affected by data loss is
instead not transmitted. All the processing prior to transmission happens as
normal, just using bogus data (typically whatever was in the chunk from the
previous time it was used), as this is simpler than trying to make vectorised
code skip over the missing data.

To track the missing data, a series of "present" boolean arrays passes down
the pipeline alongside the data. The first such array is populated by spead2.
From there a number of transformations occur:

1. When copying the head of one chain to append it to the tail of the previous
   one, the same is done with the presence flags.
2. A prefix sum (see :func:`numpy.cumsum`) is computed over the flags of the
   chunk. This allows the number of good packets in any interval to be
   computed quickly.
3. For each output spectrum, the corresponding interval of input heaps is
   computed (per polarisation) to determine whether any are missing, to
   produce per-spectrum presence flags.
4. When an output chunk is ready to be sent, the per-spectrum flags are
   reduced to per-frame flags.

.. _fgpu.ddc:

Narrowband down-conversion kernel
---------------------------------

To provide efficient operation on a narrowband region, several logical steps are
performed:

1. The signal is multiplied (:dfn:`mixed`) by a complex tone of the form
   :math:`e^{2\pi jft}`, to effect a shift in the frequency of the
   signal. The centre of the desired band is placed at the DC frequency.

2. The signal is convolved with a low-pass filter. This eliminates the
   unwanted parts of the band, to the extent possible with a FIR filter.

3. The signal is decimated (every Nth sample is retained), reducing the data
   rate. The low-pass filter above limits aliasing.

For efficiency, all three operations are implemented in the same kernel. In
particular, the filtered samples that would be removed by decimation are never
actually computed.

The kernel is one of the more complex in katgpucbf. Simpler implementations
tend to have low performance because the target GPUs (NVIDIA Ampere
architecture, particularly those based on GA-102) have far more throughput for
flops than for the load-store pipeline or local memory (recall that we're
using OpenCL :ref:`gpu-terminology`), and attempts to allievate this can also
easily consume a lot of local memory and thus reduce occupancy.

Work groups
^^^^^^^^^^^
Each work group is responsible for producing a contiguous set of output
samples (given by the constant :c:macro:`GROUP_OUT_SIZE`). To do so, it needs
to load data from :c:macro:`LOAD_SIZE` input samples, which includes the extra
samples needed to cater for the footprint of the low-pass filter.

To maximise the arithmetic intensity and minimise the number of load/store
operations, it's necessary for the kernel to hold a lot of data in registers.
To avoid needing all the data at the same time, it has an outer loop that
alternates between firstly, loading, decoding and mixing some data, and
secondly, applying the low-pass filter. These two stages use different
mappings of work items to work, and communicate through local memory.

.. _ddc-load:

Loading and unpacking
^^^^^^^^^^^^^^^^^^^^^
Initially (prior to the outer loop mentioned above), each work item loads the
packed 10-bit samples for some number of input samples into registers (between
them they load all :c:macro:`LOAD_SIZE` samples). To save space, these are
unpacked only as needed.

To simplify alignment, the input samples are divided
into :dfn:`segments` of 16 consecutive samples, which consumes 20 bytes or
five 32-bit words. The segments are distributed amongst the work items in
round-robin fashion, so that work item :math:`i` holds segments :math:`i + jW`
where :math:`W` is the work group size (:c:macro:`WGS` in the code). There
won't be an equal number of segments for each work item, so some work items
will be holding useless data.

When a sample is required, it is unpacked, given the segment and position
within the segment. The kernel is designed so that the position in the segment
is always a compile-time constant (after loop unrolling), which means the
necessary registers and shift amounts are also known at compile-time.

To cheaply achieve sign extension, the value is first shifted to the top 10
bits of a 32-bit (signed integer), then shifted right. In standard C/C++ this
is undefined behaviour, but CUDA implements the common behaviour of performing
sign extension.

In some cases the desired sample is split across a word boundary. CUDA
provides a (hardware-accelerated) :dfn:`funnel-shift` intrinsic, which allows two
words to be combined into a 64-bit word and shifted, retaining just the high
32 bits of the result; this is ideal for our use case.

Mixer signal
^^^^^^^^^^^^
Care needs to be taken with the precision of the argument to the mixer signal.
Simply evaluating the sine and cosine of :math:`2\pi f t` when
:math:`t` is large can lead to a catastrophic loss of precision, as the
product :math:`f t` will have a large integer part and leave few bits for
the fractional part. Even passing :math:`f` in single precision can lead
to large errors.

To overcome this, a hybrid approach is used. Let the first sample handled by a
work item be :math:`t_0`, and the kth sample of the ith segment be :math:`t_0
+ t_{i,k}`. Note that :math:`t_{i,k}` is the same for all work items.
We can write the mixer value as
:math:`e^{2\pi j f t_0}e^{2\pi j f t_{i,k}}`. The second factor can be
pre-computed for all :math:`i` and :math:`k` and stored in a small lookup
table. The former still needs expensive handling, but needs to be performed
far fewer times. We compute :math:`f t_0` in double precision, subtract
the nearest integer (to increase the number of fractional mantissa bits
available) and then proceed in single precision.

FIR filter
^^^^^^^^^^
For the FIR filter, a different mapping of work items to samples is used.
The work items are partitioned into :dfn:`subgroups` each containing
:c:macro:`SG_SIZE` work items. Each subgroup collaborates to produce
:c:macro:`COARSEN` consecutive output samples.

The position of each work item within its subgroup is stored in
:c:var:`sg_rank`). Each work item is responsible only for samples whose index
modulo :c:macro:`SG_SIZE` equals :c:var:`sg_rank`. It's not entirely clear why
having this division of labour improves performance, although it does reduce
the ratio of (input and output) samples to threads and hence allows for
greater occupancy.

Samples are loaded in an order that processes all input samples with the
same index modulo :c:macro:`DECIMATION` together, keeping a sliding window of
:c:macro:`COARSEN` such samples. This allows each subgroup to load each input
sample from local memory just once, even though each contributes to multiple
output samples. Note that other subgroups will still retrieve some of the
same samples (from local memory), but the coarsening mitigates the cost of
this.

At the end of the kernel, the work items in a subgroup need to sum their
individual results. This is done using a facility of :mod:`katsdpsigproc`,
which in practice utilises warp shuffle instructions. While reasonably
efficient for small values of :c:var:`SG_SIZE`, this rapidly becomes costly as
it increases: the overhead relative to the per-work item accumulation scales
as :math:`O(n\log n)`.

Tiles
^^^^^
Each segment is further subdivided into :dfn:`tiles`. For each tile,
:c:macro:`SG_SIZE` decoded and mixed samples are kept in local memory at a
time; this limitation helps reduce local memory usage. These are written in
the first phase (decoding and mixing), and read in the second phase (FIR
filter), and then the next set of :c:macro:`SG_SIZE` samples are written for
every tile, etc.

The tile size should generally be as large as possible (so that the fraction
of data held in memory is as small as possible), and in the simplest
case, tiles correspond exactly to segments. However, the tile
size must divide into the decimation factor, so when the decimation factor is
smaller than (or not a multiple of) the segment size, tiles must be smaller
than segments.

Uncoalesced access
^^^^^^^^^^^^^^^^^^
Both the global reads and writes use uncoalesced accesses, meaning that
adjacent work items do not read from/write to adjacent addresses. This can
harm performance, and usually it is beneficial to stage copies through local
memory using coalesced accesses. However, attempts to do so have only reduced
performance. It's not clear why, but it may be that there is sufficient
instruction-level parallelism to hide the latency, and the extra work on the
load-store pipeline when using local memory just slows things down.

Performance tuning
^^^^^^^^^^^^^^^^^^
The work group size, subgroup size and coarsening factor can all affect
performance significantly, and not always in obvious ways. It will likely be
necessary to implement autotuning to get optimal results across a range of
problem parameters and hardware devices, but this has not yet been done.
