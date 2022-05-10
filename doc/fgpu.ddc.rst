Narrowband down-conversion kernel
=================================

To provide efficient operation on a narrowband region, several logical steps are
performed:

1. The signal is multiplied (:dfn:`mixed`) by a complex tone of the form
   :math:`e^{2\pi i\omega t}`, to effect a shift in the frequency of the
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
-----------
Each work group is responsible for producing a contiguous set of output
samples (given by the constant :c:macro:`GROUP_OUT_SIZE`). To do so, it needs
to load data from :c:macro:`LOAD_SIZE` input samples, which includes the extra
samples needed to cater for the footprint of the low-pass filter.

Modes
-----
The kernel uses two different mappings of work items to work, and switches
back and forth between them (using local memory to pass results between them).
In the first (let's call it Mode A), each work item holds the packed 10-bit
data for some number of input samples, in registers. To save space, these are
unpacked only on demand. To simplify alignment, the input samples are divided
into :dfn:`segments` of 16 consecutive samples, which consumes 20 bytes or
five 32-bit words. The segments are distributed amongst the work items in
round-robin fashion, so that work item :math:`i` holds segments :math:`i + jW`
where :math:`W` is the work group size (:c:macro:`WGS` in the code). There
won't be an equal number of segments for each work item, so some work items
will be holding dummy data.

In the other mode (Mode B), :c:macro:`SG_SIZE` work items (a :dfn:`subgroup`)
collaborate to compute :c:macro:`COARSEN` consecutive output samples.

Unpacking
---------
When a sample is required, it is unpacked, given the segment and position
within the segment. The kernel is designed so that the position in the segment
is always a compile-time constant, which means the necessary registers and
shift amounts are also known at compile-time.

To cheaply achieve sign extension, the value is first shifted to the top 10
bits of a 32-bit (signed integer), then shifted right. In standard C/C++ this
is undefined behaviour, but CUDA implements the common behaviour of performing
sign extension.

In some cases the desired sample is split across a word boundary. CUDA
provides a (hardware-accelerated) :dfn:`funnel-shift` intrinsic, which allows two
words to be combined into a 64-bit word and shifted, retaining just the high
32 bits of the result; this is ideal for our use case.

Tiles
-----
Each segment is further subdivided into :dfn:`tiles`. For each tile,
:c:macro:`SG_SIZE` decoded and mixed samples are kept in local memory at a
time; this limitation helps reduce local memory usage. These are prepared in
mode A (decoding and mixing), processed (to apply the FIR filter) in mode B,
and then the next set of :c:macro:`SG_SIZE` samples are prepared in mode A
again etc.

Mixer signal
------------
Care needs to be taken with the precision of the argument to the mixer signal.
Simply evaluating the sine and cosine of :math:`2\pi \omega t` when
:math:`t` is large can lead to a catastrophic loss of precision, as the
product :math:`\omega t` will have a large integer part and leave few bits for
the fractional part. Even passing :math:`\omega` in single precision can lead
to large errors.

To overcome this, a hybrid approach is used. Let the first sample handled by a
work item be :math:`t_0`, and the jth sample of the ith segment be :math:`t_0
+ t_{i,j}`. Note that :math:`t_{i,j}` is the same for all work items.
We can write the mixer value as
:math:`e^{2\pi \omega t_0}e^{2\pi \omega t_{i,j}}`. The second factor can be
pre-computed for all :math:`i` and :math:`j` and stored in a small lookup
table. The former still needs expensive handling, but needs to be performed
far fewer times. We compute :math:`\omega t_0` in double precision, subtract
the nearest integer (to increase the number of fractional mantissa bits
available) and then proceed in single precision.

FIR filter
----------
The position of each work item with its subgroup is stored in
:c:var:`sg_rank`). Each work item is responsible only for samples whose index
modulo :c:macro:`SG_SIZE` equals :c:var:`sg_rank`. It's not entirely clear why
having this division of labour improves performance, although it does reduce
the ratio of (input and output) samples to threads and hence allows for
greater occupancy.

Samples are a loaded in an order that processes all input samples with the
same index modulo :c:macro:`DECIMATION` together, keeping a sliding window of
:c:macro:`COARSEN` such samples. This allows each subgroup to load each input
sample from local memory just once, even though each contributes to multiple
output samples. Note that other subgroups will still retrieve some of the
same samples, but the coarsening mitigates the cost of this.

At the end of the kernel, the work items in a subgroup need to sum their
individual results. This is done using a facility of :mod:`katsdpsigproc`,
which in practice utilises warp shuffle instructions. While reasonably
efficient for small values of :c:var:`SG_SIZE`, this rapidly becomes costly as
it increases: the overhead relative to the per-work item accumulation scales
as :math:`O(n\log n)`.

Uncoalesced access
------------------
Both the global reads and writes use uncoalesced accesses, meaning that
adjacent work items do not read from/write to adjacent addresses. This can
harm performance, and usually it is beneficial to stage copies through local
memory using coalesced accesses. However, attempts to do so have only reduced
performance. It's not clear why, but it may be that there is sufficient
instruction-level parallelism to hide the latency, and the extra work on the
load-store pipeline when using local memory just slows things down.

Performance tuning
------------------
The work group size, subgroup size and coarsening factor can all affect
performance significantly, and not always in obvious ways. It will likely be
necessary to implement autotuning to get optimal results across a range of
problem parameters and hardware devices, but this has not yet been done.
