Design
======

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
windows.

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
2. A prefix sum (see :func:`np.cumsum`) is computed over the flags of the
   chunk. This allows the number of good packets in any interval to be
   computed quickly.
3. For each output spectrum, the corresponding interval of input heaps is
   computed (per polarisation) to determine whether any are missing, to
   produce per-spectrum presence flags.
4. When an output chunk is ready to be sent, the per-spectrum flags are
   reduced to per-frame flags.

Challenges and lessons learnt
-----------------------------

Packet size
^^^^^^^^^^^
The FPGA F-engine outputs packets with 1 KiB of payload. Matching this in
software is challenging as the packet rate is high (over 3 million per
second). The transmit code can still be optimised, but we were not able to
make transmission reliable even with multiple threads (see more details
below). The small packets (together with the padding needed by the X-engines)
also increases the bandwidth significantly: 27.4 Gb/s of payload requires 31.2
Gb/s total bandwidth.

Simultaneous receive and transmit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The Mellanox ConnectX-5 exhibits some performance anomalies when
simultaneously receiving and transmitting at high speed. When running two
antennas (four polarisations) with a 100 Gb/s, packets were occasionally
dropped by the NIC. This seems to be caused by PCIe bottlenecks, possibly
exacerbated by heavy memory traffic on the host. It seems to be triggered by
micro-second scale jitter rather than a lack of throughput: upgrading to a
faster CPU and RAM did not mitigate the problem.

This problem seems to be exacerbated by memory thrashing. There are a few ways
the memory traffic can be reduced:

1. Don't do SPEAD decoding on the CPU. Receive packets directly into CUDA
   pinned memory and transfer it to the GPU, and sort it out on the GPU. If
   the packet structure is hard-coded it would also be possible to use memory
   scatter to split off the timestamps from the samples.
2. Do transfers to the GPU in smaller increments. PCI devices do DMA directly
   into the last-level cache, and if the data can be moved out again before
   it is flushed the GPU can read it from cache without touching memory.
   Ideally it would also be overwritten again by the NIC before it is
   flushed, but that would require the buffer to fit entirely in the LLC.
3. Similarly to the above, transfer data from the GPU in small pieces, and
   transmit them directly from where they're placed rather than copying the
   data into packets.

A second anomaly is that if the receiver does not make buffers available to
the NIC in time, then not only are packets dropped, but the multicast transmit
stalls every few seconds. This in turn prevents the transmit from keeping up
with the processed data, putting back-pressure on the receiver and causing it
to run out of buffers.

Cases 00690992 and 00699262 were opened with Mellanox for these problems, and
it has since been fixed in the latest firmware.

NUMA
^^^^
One machine used for testing had the GPU on a different NUMA node to the NIC.
The transfers to/from the GPU went across the QPI bus, which limited the
bandwidth and exacerbated the packet drops. This was an older Haswell Xeon;
the newer Skylake Xeon used for these tests uses UPI which provides the full
12-13 GB/s I/O for the GPU, but still exacerbates lost packets. It is
highly recommended that any system using this design has the GPU and NIC on
the same NUMA node.

We also found that single-threaded memcpy bandwidth on the Skylake Xeon
improved from about 4 GB/s to about 7 GB/s when removing the second CPU from
the system. With better memcpy performance it may be possible to use fewer
cores (and conversely, fewer cores on a die may reduce the latency to
memory and hence the memcpy performance).
