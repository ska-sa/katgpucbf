Design
======

The actual GPU kernels are reasonably straight-forward, because they're
generally memory-bound rather than compute-bound. The main challenges are in
data movement through the system.

Chunking
--------
GPUs have massive parallelism, and to exploit them fully requires large batch
sizes (millions of elements). To accommodate this, the input packets are
grouped into "chunks" of fixed numbers of samples. There is a tradeoff in the
chunk size: large chunks use more memory, add more latency to the system, and
reduce LLC (last-level cache) hit rates. Smaller chunks limit parallelism, and
as will be seen later, increase the overheads associated with overlapping PFB
windows.

Chunking also helps reduce the impact of slow Python code. Digitiser heaps
consist of only a single packet, and involving Python on a per-heap basis
would be far to slow. The code for assembling chunks is all implemented in
C++, with Python working mostly at chunk granularity.

Queues
------
The system consists of several components which run independently of each
other - either via threads (C++ code) or Python's asyncio framework. The
general pattern is that adjacent components are connected by a pair of queues:
one carrying full buckets of data forward, and one returning free data. This
approach allows all memory to be allocated up front. Slow components thus
cause back-pressure on up-stream components by not returning buckets through
the free queue fast enough. The number of buckets needs to be large enough to
smooth out jitter in processing times.

Network receive
---------------
Each polarisation is handled as a separate SPEAD stream, with a separate
thread. Separate streams are necessary because the packets themselves don't
distinguish between the polarisations, and separate threads are necessary
because a single core is not fast enough to load the data. This introduces
some challenges in aligning the polarisations, because locking a shared
structure on every packet would be prohibitively expensive. Instead, the
polarisations are kept separate during chunking, and aligned afterwards (in
Python). Large chunks also make it unlikely that the polarisations will be out
of sync by more than one chunk, which makes the alignment code quite simple.

To minimise the number of copies, a spead2 custom allocator is used to have
spead2 copy the data directly into contiguous memory for the chunk. We use
CUDA pinned memory so that it can be efficiently copied to the GPU. To avoid a
compile-time dependency on CUDA, we have Python code allocate the memory (from
PyCUDA) and pass pre-allocated chunks into the C++ code.

GPU Processing
--------------

Terminology
^^^^^^^^^^^
We will use OpenCL terminology, as it is more generic. An OpenCL *workitem*
corresponds to a CUDA *thread*. Each workitem logically executes the same
program but with different parameters, and can share data through *local
memory* (shared memory in CUDA) with other workitems in the same
*workgroup* (thread block in CUDA).

Decode
^^^^^^
Digitiser samples are 10-bit and stored compactly. While it is possible to
write a dedicated kernel for decoding that makes efficient accesses to memory
(using contiguous word-size loads), it is faster overall to do the decoding as
part of the PFB filter because it avoids a round trip to memory. The decode is
done in a very simple manner:

 1. Determine the two bytes that hold the sample.
 2. Load them and combine them into a 16-bit value.
 3. Shift left to place the desired 10 bits in the high bits.
 4. Shift right to sign extend.
 5. Convert to float.

While many bytes get loaded twice (because they hold bits from two samples),
the cache is able to prevent this affecting DRAM bandwidth.

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

FFT
^^^
After the FIR above, we can perform the FFT, which is done with a cuFFT
real-to-complex transformation. This is straightforward, and the built-in
support for doing multiple FFTs at once means that it can saturate the GPU
even with small channel counts. cuFFT does write an output for the Nyquist
frequency (which is discarded in the MeerKAT design), but we take care of that
in the following step.

Postprocessing
^^^^^^^^^^^^^^
The remaining steps are to

 1. Apply gains and fine delays.
 2. Do a partial transpose, so that *acc_len* (256) spectra are stored
    contiguously for each channel (the Nyquist frequencies are also discarded
    at this point).
 3. Convert to int8.
 4. Interleave the polarisations.

These are all combined into a single kernel to minimise memory traffic. The
katsdpsigproc package provides a template for transpositions, and the other
operations are all straightforward. While C++ doesn't have a convert with
saturation function, we can access the CUDA functionality through inline PTX
assembly (OpenCL C has an equivalent function).

Fine delays are computed using the `sincospi` function, which saves both a
multiplication by π and a range reduction.

Coarse delays
^^^^^^^^^^^^^
One of the more challenging aspects of the processing design was the handling
of delays. In the end we chose to exploit the fact that the expected delay
rates are very small, typically leading to at most one coarse delay change per
chunk. We thus break up each chunk into sections where the coarse delay is
constant.

Our approach is based on inverting the delay model: output timestamps are
regularly spaced, and for each output spectrum, determine the sample in the
input that will be delayed until that time (to the nearest sample). We then
take a contiguous range of input samples starting from that point to use in
the PFB. Unlike the MeerKAT FPGA F-engine, this means that every output
spectrum has a common delay for all samples. There will also likely be
differences from the MeerKAT F-engine when there are large discontinuities in
the delay model, as the inversion becomes ambiguous.

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
The current transmit system is quite simple and could use optimisations. A
separate SPEAD stream is created for each X-engine, and C++ code splits each
output chunk into heaps.

Proposed F-X ICD changes
------------------------
If a software F-engine is combined with a software X-engine, there would be
opportunity to more easily change the interface.

While the transmit code could still be optimised, the 1 KiB output packet size
adds significant CPU load. If the X-engines could support it, a larger (e.g.,
8 KiB) packet size would reduce load. Similarly, with many X-engines and few
channels, the heap size becomes extremely small, and this would ideally be
compensated for by an increase in the number of spectra per heap.
