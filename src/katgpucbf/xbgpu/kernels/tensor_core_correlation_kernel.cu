/*    Original license:
 *
 *    Copyright 2021 ASTRON
 *
 *   Licensed under the Apache License, Version 2.0 (the "License");
 *   you may not use this file except in compliance with the License.
 *   You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *   Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *   See the License for the specific language governing permissions and
 *   limitations under the License.
 */

/* This code is based on
 * https://git.astron.nl/RD/tensor-core-correlator/-/blob/83abdcc/libtcc/TCCorrelator.cu
 *
 * See https://developer.nvidia.com/gtc/2019/video/s9306 for a high-level overview.
 * Lower-level details are in the doc/xbgpu.tcc.rst (and built by Sphinx with
 * the rest of the documentation).
 *
 * It has been modified by SARAO:
 * - Wrap the file in extern "C++" to make it work with PyCUDA (see below)
 * - Add results to the output instead of overwriting, to allow accumulation
 *   across multiple calls (with saturation rather than wrapping).
 * - Conjugate the output, to provide the other triangle of the visibility
 *   matrix.
 * - Take the input axes in a different order.
 * - Remove trailing whitespace.
 *
 * SARAO's modification is licenced as follows:
 *******************************************************************************
 * Copyright (c) 2020-2021, National Research Foundation (SARAO)
 *
 * Licensed under the BSD 3-Clause License (the "License"); you may not use
 * this file except in compliance with the License. You may obtain a copy
 * of the License at
 *
 *   https://opensource.org/licenses/BSD-3-Clause
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ******************************************************************************/

/* PyCUDA wraps the whole file in 'extern "C"', but most of the code expects
 * C++ linkage. So we wrap the whole original file in 'extern "C++"' to cancel
 * that out.
 *
 * When this code gets closer to production, the suggested fix is to modify the
 * accel.build() and context.compile() functions in katsdpsigproc to take a
 * no_extern_c flag as these ones are the methods that will call the
 * pycuda.compiler.SourceModule(...) constructor.
 */
extern "C++" {

#if 1000 * __CUDACC_VER_MAJOR__ + __CUDACC_VER_MINOR__ >= 11001 && __CUDA_ARCH__ >= 800
#define ASYNC_COPIES
#endif

#if defined ASYNC_COPIES
#include <cooperative_groups/memcpy_async.h>
#endif

#include <mma.h>

#define NR_BASELINES		(NR_RECEIVERS * (NR_RECEIVERS + 1) / 2)
#define ALIGN(A,N)		(((A)+(N)-1)/(N)*(N))

#define NR_TIMES_PER_BLOCK	(128 / (NR_BITS))
#define NR_RECEIVERS_PER_TCM_X	((NR_BITS) == 4 ? 2 : 4)
#define NR_RECEIVERS_PER_TCM_Y	((NR_BITS) == 4 ? 4 : 8)

#define COMPLEX			2

#if __CUDA_ARCH__ < (NR_BITS == 4 ? 730 : NR_BITS == 8 ? 720 : NR_BITS == 16 ? 700 : 0)
#error this architecture has no suitable tensor cores
#endif

#if __CUDA_ARCH__ != 700 && __CUDA_ARCH__ != 720 && __CUDA_ARCH__ != 750 && __CUDA_ARCH__ != 800 && __CUDA_ARCH__ != 860
#define PORTABLE // unknown architecture -> write visibilities in portable way (via shared memory)
#endif

#if NR_RECEIVERS_PER_BLOCK != 32 && NR_RECEIVERS_PER_BLOCK != 48 && NR_RECEIVERS_PER_BLOCK != 64
#error unsupported NR_RECEIVERS_PER_BLOCK
#endif

#if NR_SAMPLES_PER_CHANNEL % NR_TIMES_PER_BLOCK != 0
#error NR_SAMPLES_PER_CHANNEL should be a multiple of NR_TIMES_PER_BLOCK
#endif

#define MIN(A,B) ((A)<(B)?(A):(B))


using namespace nvcuda::wmma;

#if NR_BITS == 4
typedef char    Sample;
typedef int2    Visibilities[NR_CHANNELS][NR_BASELINES][NR_POLARIZATIONS][NR_POLARIZATIONS];
#elif NR_BITS == 8
typedef char2   Sample;
typedef int2    Visibilities[NR_CHANNELS][NR_BASELINES][NR_POLARIZATIONS][NR_POLARIZATIONS];
#elif NR_BITS == 16
typedef __half2 Sample;
typedef float2  Visibilities[NR_CHANNELS][NR_BASELINES][NR_POLARIZATIONS][NR_POLARIZATIONS];
#endif
typedef Sample Samples[NR_RECEIVERS][NR_CHANNELS][NR_SAMPLES_PER_CHANNEL / NR_TIMES_PER_BLOCK][NR_TIMES_PER_BLOCK][NR_POLARIZATIONS];

#if NR_BITS == 4
typedef fragment<matrix_a, 8, 8, 32, experimental::precision::s4, row_major> Afrag;
typedef fragment<matrix_b, 8, 8, 32, experimental::precision::s4, col_major> Bfrag;
typedef fragment<accumulator, 8, 8, 32, int>                                 Sum;
#elif NR_BITS == 8
typedef fragment<matrix_a, 16, 16, 16, signed char, row_major>               Afrag;
typedef fragment<matrix_b, 16, 16, 16, signed char, col_major>               Bfrag;
typedef fragment<accumulator, 16, 16, 16, int>                               Sum;
#elif NR_BITS == 16
typedef fragment<matrix_a, 16, 16, 16, __half, row_major>                    Afrag;
typedef fragment<matrix_b, 16, 16, 16, __half, col_major>                    Bfrag;
typedef fragment<accumulator, 16, 16, 16, float>                             Sum;
#endif


#if NR_BITS == 4
typedef int2   ScratchSpace[4][NR_POLARIZATIONS][2][NR_POLARIZATIONS];
#elif NR_BITS == 8
typedef int2   ScratchSpace[8][NR_POLARIZATIONS][4][NR_POLARIZATIONS];
#elif NR_BITS == 16
typedef float2 ScratchSpace[8][NR_POLARIZATIONS][4][NR_POLARIZATIONS];
#endif


__device__ inline int conj_perm(int v)
{
#if NR_BITS == 4
  //return ((v & 0x0F0F0F0F) << 4) | (__vnegss4(v >> 4) & 0x0F0F0F0F);
  return ((v & 0x0F0F0F0F) << 4) | ((0xF0F0F0F0 - ((v >> 4) & 0x0F0F0F0F)) & 0x0F0F0F0F);
#elif NR_BITS == 8
  //return __byte_perm(v, __vnegss4(v), 0x2705);
  return __byte_perm(v, 0x00FF00FF - (v & 0xFF00FF00), 0x2705);
#elif NR_BITS == 16
  return __byte_perm(v ^ 0x80000000, v, 0x1032);
#endif
}


__device__ inline int2 conj_perm(int2 v)
{
  return make_int2(conj_perm(v.x), conj_perm(v.y));
}


__device__ inline int4 conj_perm(int4 v)
{
  return make_int4(conj_perm(v.x), conj_perm(v.y), conj_perm(v.z), conj_perm(v.w));
}


#if defined ASYNC_COPIES
#define READ_AHEAD        MIN(2, NR_SAMPLES_PER_CHANNEL / NR_TIMES_PER_BLOCK)
#define NR_SHARED_BUFFERS MIN(4, NR_SAMPLES_PER_CHANNEL / NR_TIMES_PER_BLOCK)
#else
#define READ_AHEAD        1
#define NR_SHARED_BUFFERS 2
#endif


template <unsigned nrReceiversPerBlock = NR_RECEIVERS_PER_BLOCK> struct SharedData
{
#if NR_BITS == 4
  typedef char        Asamples[NR_SHARED_BUFFERS][nrReceiversPerBlock][NR_POLARIZATIONS][NR_TIMES_PER_BLOCK][1];
  typedef char        Bsamples[NR_SHARED_BUFFERS][nrReceiversPerBlock][NR_POLARIZATIONS][COMPLEX][NR_TIMES_PER_BLOCK + 16][1];
#elif NR_BITS == 8
  typedef signed char Asamples[NR_SHARED_BUFFERS][nrReceiversPerBlock][NR_POLARIZATIONS][NR_TIMES_PER_BLOCK][COMPLEX];
  typedef signed char Bsamples[NR_SHARED_BUFFERS][nrReceiversPerBlock][NR_POLARIZATIONS][COMPLEX][NR_TIMES_PER_BLOCK + 8][COMPLEX];
#elif NR_BITS == 16
  typedef __half      Asamples[NR_SHARED_BUFFERS][nrReceiversPerBlock][NR_POLARIZATIONS][NR_TIMES_PER_BLOCK][COMPLEX];
  typedef __half      Bsamples[NR_SHARED_BUFFERS][nrReceiversPerBlock][NR_POLARIZATIONS][COMPLEX][NR_TIMES_PER_BLOCK + 4][COMPLEX];
#endif
};


template <typename T> struct FetchData
{
  __device__ FetchData(unsigned loadRecv, unsigned loadPol, unsigned loadTime)
  :
    loadRecv(loadRecv), loadPol(loadPol), loadTime(loadTime), data({0})
  {
  }

  __device__ void load(const Samples samples, unsigned channel, unsigned time, unsigned firstReceiver, bool skipLoadCheck = NR_RECEIVERS % NR_RECEIVERS_PER_BLOCK == 0)
  {
    if (skipLoadCheck || firstReceiver + loadRecv < NR_RECEIVERS)
    {
#pragma unroll
      for (unsigned i = 0; i < sizeof(T) / sizeof(Sample); i++)
        memcpy((char *) &data + i * sizeof(Sample), &samples[firstReceiver + loadRecv][channel][time][loadTime + i][loadPol], sizeof(Sample));
    }
  }

  template <typename SharedData> __device__ void storeA(SharedData samples) const
  {
    //* ((T *) &samples[loadRecv][loadPol][loadTime][0]) = data;
    memcpy(&samples[loadRecv][loadPol][loadTime][0], &data, sizeof(T));
  }

  template <typename SharedData> __device__ void storeB(SharedData samples) const
  {
    //* ((T *) &samples[loadRecv][loadPol][0][loadTime][0]) = data;
    //* ((T *) &samples[loadRecv][loadPol][1][loadTime][0]) = conj_perm(data);
    T tmp = conj_perm(data);
    memcpy(&samples[loadRecv][loadPol][0][loadTime][0], &data, sizeof(T));
    memcpy(&samples[loadRecv][loadPol][1][loadTime][0], &tmp, sizeof(T));
  }

#if defined ASYNC_COPIES
  template <typename Asamples> __device__ void copyAsyncA(nvcuda::experimental::pipeline &pipe, Asamples dest, const Samples samples, unsigned channel, unsigned time, unsigned firstReceiver, bool skipLoadCheck = NR_RECEIVERS % NR_RECEIVERS_PER_BLOCK == 0)
  {
    if (skipLoadCheck || firstReceiver + loadRecv < NR_RECEIVERS)
    {
#pragma unroll
      for (unsigned i = 0; i < sizeof(T) / sizeof(Sample); i++)
        nvcuda::experimental::memcpy_async(* (Sample *) &dest[loadRecv][loadPol][loadTime + i][0], * (const Sample *) &samples[firstReceiver + loadRecv][channel][time][loadTime + i][loadPol], pipe);
    }
  }

  template<typename Bsamples> __device__ void copyAsyncB(nvcuda::experimental::pipeline &pipe, Bsamples dest, const Samples samples, unsigned channel, unsigned time, unsigned firstReceiver, bool skipLoadCheck = NR_RECEIVERS % NR_RECEIVERS_PER_BLOCK == 0)
  {
    if (skipLoadCheck || firstReceiver + loadRecv < NR_RECEIVERS)
    {
#pragma unroll
      for (unsigned i = 0; i < sizeof(T) / sizeof(Sample); i++)
        nvcuda::experimental::memcpy_async(* (Sample *) &dest[loadRecv][loadPol][0][loadTime + i][0], * (const Sample *) &samples[firstReceiver + loadRecv][channel][time][loadTime + i][loadPol], pipe);
    }
  }

  template<typename Bsamples> __device__ void fixB(Bsamples bSamples)
  {
    //* ((T *) &bSamples[loadRecv][loadPol][1][loadTime][0]) = conj_perm(* ((T *) &bSamples[loadRecv][loadPol][0][loadTime][0]));
    T tmp;
    memcpy(&tmp, &bSamples[loadRecv][loadPol][0][loadTime][0], sizeof(T));
    tmp = conj_perm(tmp);
    memcpy(&bSamples[loadRecv][loadPol][1][loadTime][0], &tmp, sizeof(T));
  }
#endif

  unsigned loadRecv, loadPol, loadTime;
  T        data;
};


__device__ inline int2 make_complex(int real, int imag)
{
  return make_int2(real, imag);
}


__device__ inline float2 make_complex(float real, float imag)
{
  return make_float2(real, imag);
}


// Compute x + y clamped to -INT_MAX .. INT_MAX
__device__ inline int add_sat(int x, int y)
{
  int out;
  asm("add.sat.s32 %0, %1, %2;" : "=r" (out) : "r" (x), "r" (y));
  // add.sat.s32 clamps to INT_MIN..INT_MAX, but we want -INT_MAX..INT_MAX
  // for symmetry.
  return max(out, INT_MIN + 1);
}


template <typename T> __device__ inline void accumVisibility(T &out, T value)
{
  /* Store an output value. Unlike the original ASTRON code, for xbgpu this
   * - conjugates the value because we want to store the other half
   *   (triangle) of the visibility matrix; and
   * - adds to the existing value (with saturation, if integer), to allow
   *   accumulation across multiple calls to the kernel.
   */
#if NR_BITS == 16
  out = make_complex(out.x + value.x, out.y - value.y);
#else
  out = make_complex(add_sat(out.x, value.x), add_sat(out.y, -value.y));
#endif
}


template <typename T> __device__ inline void storeVisibility(Visibilities visibilities, unsigned channel, unsigned baseline, unsigned recvY, unsigned recvX, unsigned tcY, unsigned tcX, unsigned polY, unsigned polX, bool skipCheckY, bool skipCheckX, T sumR, T sumI)
{
  if ((skipCheckX || recvX + tcX <= recvY + tcY) && (skipCheckY || recvY + tcY < NR_RECEIVERS))
  {
    accumVisibility(visibilities[channel][baseline + tcY * recvY + tcY * (tcY + 1) / 2 + tcX][polY][polX],
                    make_complex(sumR, sumI));
  }
}


__device__ inline void storeVisibilities(Visibilities visibilities, unsigned channel, unsigned firstReceiverY, unsigned firstReceiverX, unsigned recvYoffset, unsigned recvXoffset, unsigned y, unsigned x, bool skipCheckY, bool skipCheckX, const Sum &sum, ScratchSpace scratchSpace[], unsigned warp)
{
#if defined PORTABLE
 store_matrix_sync(&scratchSpace[warp][0][0][0][0].x, sum, NR_BITS == 4 ? 8 : 16, mem_row_major);
  __syncwarp();

#if 0
  if (threadIdx.x == 0)
    for (unsigned _y = 0; _y < 8; _y ++)
      for (unsigned pol_y = 0; pol_y < NR_POLARIZATIONS; pol_y ++)
        for (unsigned _x = 0; _x < 4; _x ++)
          for (unsigned pol_x = 0; pol_x < NR_POLARIZATIONS; pol_x ++)
            if (scratchSpace[warp][_y][pol_y][_x][pol_x],x != 0 || scratchSpace[warp][_y][pol_y][_x][pol_x].y != 0)
              printf("firstY=%u firstX=%u warp=%u y=%u x=%u _y=%u pol_y=%u _x=%u pol_x=%u val=(%f,%f)\n", firstReceiverY, firstReceiverX, warp, y, x, _y, pol_y, _x, pol_x, scratchSpace[warp][_y][pol_y][_x][pol_x].x, scratchSpace[warp][_y][pol_y][_x][pol_x].y);
#endif

#if NR_BITS == 4
  unsigned _y       = threadIdx.x >> 3;
  unsigned _x       = (threadIdx.x >> 2) & 1;
  unsigned polY     = (threadIdx.x >> 1) & 1;
  unsigned polX     = threadIdx.x & 1;
#elif NR_BITS == 8 || NR_BITS == 16
  unsigned _y       = threadIdx.x >> 2;
  unsigned _x       = threadIdx.x & 3;
#endif

  unsigned recvY    = firstReceiverY + recvYoffset + NR_RECEIVERS_PER_TCM_Y * y + _y;
  unsigned recvX    = firstReceiverX + recvXoffset + NR_RECEIVERS_PER_TCM_X * x + _x;
  unsigned baseline = (recvY * (recvY + 1) / 2) + recvX;

  if ((skipCheckX || recvX <= recvY) && (skipCheckY || recvY < NR_RECEIVERS))
#if NR_BITS == 4
    accumVisibility(visibilities[channel][baseline][polY][polX], scratchSpace[warp][_y][polY][_x][polX]);
#elif NR_BITS == 8 || NR_BITS == 16
    for (unsigned polY = 0; polY < NR_POLARIZATIONS; polY ++)
      for (unsigned polX = 0; polX < NR_POLARIZATIONS; polX ++)
        accumVisibility(visibilities[channel][baseline][polY][polX], scratchSpace[warp][_y][polY][_x][polX]);
#endif
#else
#if __CUDA_ARCH__ == 700 || (__CUDA_ARCH__ == 720 && NR_BITS == 16)
  unsigned recvY    = firstReceiverY + recvYoffset + NR_RECEIVERS_PER_TCM_Y * y + ((threadIdx.x >> 3) & 2) + (threadIdx.x & 4);
  unsigned recvX    = firstReceiverX + recvXoffset + NR_RECEIVERS_PER_TCM_X * x + ((threadIdx.x >> 2) & 2);
  unsigned polY     = threadIdx.x & 1;
  unsigned polX     = (threadIdx.x >> 1) & 1;
#elif (__CUDA_ARCH__ == 720 && NR_BITS == 8) || __CUDA_ARCH__ == 750 || __CUDA_ARCH__ == 800 || __CUDA_ARCH__ == 860
  unsigned recvY    = firstReceiverY + recvYoffset + NR_RECEIVERS_PER_TCM_Y * y + ((threadIdx.x >> 3) & 3);
  unsigned recvX    = firstReceiverX + recvXoffset + NR_RECEIVERS_PER_TCM_X * x + ((threadIdx.x >> 1) & 1);
  unsigned polY     = (threadIdx.x >> 2) & 1;
  unsigned polX     = threadIdx.x & 1;
#endif

  unsigned baseline = (recvY * (recvY + 1) / 2) + recvX;

#if __CUDA_ARCH__ == 700 || (__CUDA_ARCH__ == 720 && NR_BITS == 16)
  storeVisibility(visibilities, channel, baseline, recvY, recvX, 0, 0, polY, polX, skipCheckY, skipCheckX, sum.x[0], sum.x[1]);
  storeVisibility(visibilities, channel, baseline, recvY, recvX, 0, 1, polY, polX, skipCheckY, skipCheckX, sum.x[4], sum.x[5]);
  storeVisibility(visibilities, channel, baseline, recvY, recvX, 1, 0, polY, polX, skipCheckY, skipCheckX, sum.x[2], sum.x[3]);
  storeVisibility(visibilities, channel, baseline, recvY, recvX, 1, 1, polY, polX, skipCheckY, skipCheckX, sum.x[6], sum.x[7]);
#elif (__CUDA_ARCH__ == 720 && NR_BITS == 8) || __CUDA_ARCH__ == 750 || __CUDA_ARCH__ == 800 || __CUDA_ARCH__ == 860
  storeVisibility(visibilities, channel, baseline, recvY, recvX, 0, 0, polY, polX, skipCheckY, skipCheckX, sum.x[0], sum.x[1]);
#if NR_BITS == 8 || NR_BITS == 16
  storeVisibility(visibilities, channel, baseline, recvY, recvX, 0, 2, polY, polX, skipCheckY, skipCheckX, sum.x[4], sum.x[5]);
  storeVisibility(visibilities, channel, baseline, recvY, recvX, 4, 0, polY, polX, skipCheckY, skipCheckX, sum.x[2], sum.x[3]);
  storeVisibility(visibilities, channel, baseline, recvY, recvX, 4, 2, polY, polX, skipCheckY, skipCheckX, sum.x[6], sum.x[7]);
#endif
#endif
#endif
}


#define NR_WARPS 4

#if NR_RECEIVERS_PER_BLOCK == 64

template <bool fullTriangle> __device__ void doCorrelateTriangle(Visibilities visibilities, const Samples samples, unsigned firstReceiver, unsigned warp, unsigned tid, SharedData<>::Bsamples &bSamples, ScratchSpace scratchSpace[NR_WARPS])
{
  const unsigned nrFragmentsX = NR_BITS == 4 ? 12 : 6;
  const unsigned nrFragmentsY = nrFragmentsX / 2;
  Sum            sum[nrFragmentsX * nrFragmentsY];

  for (auto &s : sum)
    fill_fragment(s, 0);

  unsigned channel = blockIdx.y;

  const uchar2 offsets[] = {
    make_uchar2( 0,  0),
    make_uchar2( 0, 16),
    make_uchar2( 0, 40),
    make_uchar2(24, 40),
  };

  unsigned recvXoffset = offsets[warp].x;
  unsigned recvYoffset = offsets[warp].y;

  FetchData<int4> tmp0((tid >> 2)                             , (tid >> 1) & 1, 64 / NR_BITS * (tid & 1));
  FetchData<int4> tmp1((tid >> 2) + NR_RECEIVERS_PER_BLOCK / 2, (tid >> 1) & 1, 64 / NR_BITS * (tid & 1));

#if defined ASYNC_COPIES
  using namespace nvcuda::experimental;
  pipeline pipe;

  for (unsigned majorTime = 0; majorTime < READ_AHEAD; majorTime ++) {
    unsigned fetchBuffer = majorTime;

    tmp0.copyAsyncB(pipe, bSamples[fetchBuffer], samples, channel, majorTime, firstReceiver, fullTriangle);
    tmp1.copyAsyncB(pipe, bSamples[fetchBuffer], samples, channel, majorTime, firstReceiver, fullTriangle);

    pipe.commit();
  }
#else
  tmp0.load(samples, channel, 0, firstReceiver, fullTriangle);
  tmp1.load(samples, channel, 0, firstReceiver, fullTriangle);
#endif

  for (unsigned majorTime = 0; majorTime < NR_SAMPLES_PER_CHANNEL / NR_TIMES_PER_BLOCK; majorTime ++) {
    unsigned buffer = majorTime % NR_SHARED_BUFFERS;

#if !defined ASYNC_COPIES
    tmp0.storeB(bSamples[buffer]);
    tmp1.storeB(bSamples[buffer]);
#endif

    unsigned majorReadTime = majorTime + READ_AHEAD;

    if (majorReadTime < NR_SAMPLES_PER_CHANNEL / NR_TIMES_PER_BLOCK) {
#if defined ASYNC_COPIES
      unsigned fetchBuffer = (buffer + READ_AHEAD) % NR_SHARED_BUFFERS;

      tmp0.copyAsyncB(pipe, bSamples[fetchBuffer], samples, channel, majorReadTime, firstReceiver, fullTriangle);
      tmp1.copyAsyncB(pipe, bSamples[fetchBuffer], samples, channel, majorReadTime, firstReceiver, fullTriangle);
#else
      tmp0.load(samples, channel, majorReadTime, firstReceiver, fullTriangle);
      tmp1.load(samples, channel, majorReadTime, firstReceiver, fullTriangle);
#endif
    }

#if defined ASYNC_COPIES
    pipe.commit();
    pipe.wait_prior<READ_AHEAD>();

    tmp0.fixB(bSamples[buffer]);
    tmp1.fixB(bSamples[buffer]);
#endif

    __syncthreads();

#pragma unroll
    for (unsigned minorTime = 0; minorTime < NR_TIMES_PER_BLOCK; minorTime += ((NR_BITS) == 4 ? 16 : 8)) {
      Afrag aFrag[nrFragmentsY];
      Bfrag bFrag[nrFragmentsX];

      if (warp != 0) {
	for (unsigned y = 0; y < nrFragmentsY; y ++)
	  load_matrix_sync(aFrag[y], &bSamples[buffer][recvYoffset + NR_RECEIVERS_PER_TCM_Y * y][0][0][minorTime][0], sizeof(bSamples[0][0][0]) * 8 / NR_BITS);

	for (unsigned x = 0; x < nrFragmentsX; x ++)
	  load_matrix_sync(bFrag[x], &bSamples[buffer][recvXoffset + NR_RECEIVERS_PER_TCM_X * x][0][0][minorTime][0], sizeof(bSamples[0][0][0][0]) * 8 / NR_BITS);

	for (unsigned y = 0, i = 0; y < nrFragmentsY; y ++)
	  for (unsigned x = 0; x < nrFragmentsX; x ++, i ++)
	    mma_sync(sum[i], aFrag[y], bFrag[x], sum[i]);
      } else {
	for (unsigned z = 0, i = 0; z < 3; z ++) {
	  for (unsigned y = 0; y < (NR_BITS == 4 ? 4 : 2); y ++)
	    load_matrix_sync(aFrag[y], &bSamples[buffer][/*recvYoffset*/ 24 * z + NR_RECEIVERS_PER_TCM_Y * y][0][0][minorTime][0], sizeof(bSamples[0][0][0]) * 8 / NR_BITS);

	  for (unsigned x = 0; x < (NR_BITS == 4 ? 8 : 4); x ++)
	    load_matrix_sync(bFrag[x], &bSamples[buffer][/*recvXoffset*/ 24 * z + NR_RECEIVERS_PER_TCM_X * x][0][0][minorTime][0], sizeof(bSamples[0][0][0][0]) * 8 / NR_BITS);

	  for (unsigned y = 0; y < (NR_BITS == 4 ? 4 : 2); y ++)
	    for (unsigned x = 0; x < 2 + 2 * y; x ++, i ++)
	      mma_sync(sum[i], aFrag[y], bFrag[x], sum[i]);
	}
      }
    }
  }

#if defined PORTABLE
  __syncthreads();
#endif

  if (warp != 0)
    for (unsigned y = 0, i = 0; y < nrFragmentsY; y ++)
      for (unsigned x = 0; x < nrFragmentsX; x ++, i ++)
	storeVisibilities(visibilities, channel, firstReceiver, firstReceiver, recvYoffset, recvXoffset, y, x, fullTriangle, x < 2 * y + (NR_BITS == 4 ? 8 : 4), sum[i], scratchSpace, warp);
  else
    for (unsigned z = 0, i = 0; z < 3; z ++)
      for (unsigned y = 0; y < (NR_BITS == 4 ? 4 : 2); y ++)
	for (unsigned x = 0; x < 2 * y + 2; x ++, i ++)
	  storeVisibilities(visibilities, channel, firstReceiver, firstReceiver, 24 * z, 24 * z, y, x, fullTriangle, x < 2 * y, sum[i], scratchSpace, warp);
}

#endif


template <unsigned nrFragmentsY, bool skipLoadYcheck, bool skipLoadXcheck, bool skipStoreYcheck, bool skipStoreXcheck> __device__ void doCorrelateRectangle(Visibilities visibilities, const Samples samples, unsigned firstReceiverY, unsigned firstReceiverX, SharedData<>::Asamples &aSamples, SharedData<NR_RECEIVERS_PER_BLOCK == 64 ? 32 : NR_RECEIVERS_PER_BLOCK>::Bsamples &bSamples, ScratchSpace scratchSpace[NR_WARPS])
{
  const unsigned nrFragmentsX = NR_RECEIVERS_PER_BLOCK / NR_RECEIVERS_PER_TCM_X / 2 / (NR_RECEIVERS_PER_BLOCK == 64 ? 2 : 1);

  Sum sum[nrFragmentsY][nrFragmentsX];

  for (unsigned y = 0; y < nrFragmentsY; y ++)
    for (unsigned x = 0; x < nrFragmentsX; x ++)
      fill_fragment(sum[y][x], 0);

  unsigned tid     = warpSize * (blockDim.y * threadIdx.z + threadIdx.y) + threadIdx.x;
  unsigned channel = blockIdx.y;

  unsigned recvXoffset = nrFragmentsX * NR_RECEIVERS_PER_TCM_X * threadIdx.y;
  unsigned recvYoffset = nrFragmentsY * NR_RECEIVERS_PER_TCM_Y * threadIdx.z;

  FetchData<int4> tmpY0((tid >> 2)     , (tid >> 1) & 1, 64 / NR_BITS * (tid & 1));
  FetchData<int4> tmpX0((tid >> 2)     , (tid >> 1) & 1, 64 / NR_BITS * (tid & 1));
#if NR_RECEIVERS_PER_BLOCK == 48
  FetchData<int2> tmpY1((tid >> 3) + 32, (tid >> 2) & 1, 32 / NR_BITS * (tid & 3));
  FetchData<int2> tmpX1((tid >> 3) + 32, (tid >> 2) & 1, 32 / NR_BITS * (tid & 3));
#elif NR_RECEIVERS_PER_BLOCK == 64
  FetchData<int4> tmpY1((tid >> 2) + 32, (tid >> 1) & 1, 64 / NR_BITS * (tid & 1));
#endif

#if defined ASYNC_COPIES
  using namespace nvcuda::experimental;
  pipeline pipe;

  for (unsigned majorTime = 0; majorTime < READ_AHEAD; majorTime ++) {
    unsigned fetchBuffer = majorTime;

    tmpY0.copyAsyncA(pipe, aSamples[fetchBuffer], samples, channel, majorTime, firstReceiverY, skipLoadYcheck);
#if NR_RECEIVERS_PER_BLOCK == 48 || NR_RECEIVERS_PER_BLOCK == 64
    tmpY1.copyAsyncA(pipe, aSamples[fetchBuffer], samples, channel, majorTime, firstReceiverY, skipLoadYcheck);
#endif
    tmpX0.copyAsyncB(pipe, bSamples[fetchBuffer], samples, channel, majorTime, firstReceiverX, skipLoadXcheck);
#if NR_RECEIVERS_PER_BLOCK == 48
    tmpX1.copyAsyncB(pipe, bSamples[fetchBuffer], samples, channel, majorTime, firstReceiverX, skipLoadXcheck);
#endif

    pipe.commit();
  }
#else
  tmpY0.load(samples, channel, 0, firstReceiverY, skipLoadYcheck);
#if NR_RECEIVERS_PER_BLOCK == 48 || NR_RECEIVERS_PER_BLOCK == 64
  tmpY1.load(samples, channel, 0, firstReceiverY, skipLoadYcheck);
#endif
  tmpX0.load(samples, channel, 0, firstReceiverX, skipLoadXcheck);
#if NR_RECEIVERS_PER_BLOCK == 48
  tmpX1.load(samples, channel, 0, firstReceiverX, skipLoadXcheck);
#endif
#endif

  for (unsigned majorTime = 0; majorTime < NR_SAMPLES_PER_CHANNEL / NR_TIMES_PER_BLOCK; majorTime ++) {
    unsigned buffer = majorTime % NR_SHARED_BUFFERS;

#if !defined ASYNC_COPIES
    tmpY0.storeA(aSamples[buffer]);
#if NR_RECEIVERS_PER_BLOCK == 48 || NR_RECEIVERS_PER_BLOCK == 64
    tmpY1.storeA(aSamples[buffer]);
#endif
    tmpX0.storeB(bSamples[buffer]);
#if NR_RECEIVERS_PER_BLOCK == 48
    tmpX1.storeB(bSamples[buffer]);
#endif
#endif

    unsigned majorReadTime = majorTime + READ_AHEAD;

    if (majorReadTime < NR_SAMPLES_PER_CHANNEL / NR_TIMES_PER_BLOCK) {
#if defined ASYNC_COPIES
      unsigned fetchBuffer = (buffer + READ_AHEAD) % NR_SHARED_BUFFERS;

      tmpY0.copyAsyncA(pipe, aSamples[fetchBuffer], samples, channel, majorReadTime, firstReceiverY, skipLoadYcheck);
#if NR_RECEIVERS_PER_BLOCK == 48 || NR_RECEIVERS_PER_BLOCK == 64
      tmpY1.copyAsyncA(pipe, aSamples[fetchBuffer], samples, channel, majorReadTime, firstReceiverY, skipLoadYcheck);
#endif
      tmpX0.copyAsyncB(pipe, bSamples[fetchBuffer], samples, channel, majorReadTime, firstReceiverX, skipLoadXcheck);
#if NR_RECEIVERS_PER_BLOCK == 48
      tmpX1.copyAsyncB(pipe, bSamples[fetchBuffer], samples, channel, majorReadTime, firstReceiverX, skipLoadXcheck);
#endif
#else
      tmpY0.load(samples, channel, majorReadTime, firstReceiverY, skipLoadYcheck);
#if NR_RECEIVERS_PER_BLOCK == 48 || NR_RECEIVERS_PER_BLOCK == 64
      tmpY1.load(samples, channel, majorReadTime, firstReceiverY, skipLoadYcheck);
#endif
      tmpX0.load(samples, channel, majorReadTime, firstReceiverX, skipLoadXcheck);
#if NR_RECEIVERS_PER_BLOCK == 48
      tmpX1.load(samples, channel, majorReadTime, firstReceiverX, skipLoadXcheck);
#endif
#endif
    }

#if defined ASYNC_COPIES
    pipe.commit();
    pipe.wait_prior<READ_AHEAD>();

    tmpX0.fixB(bSamples[buffer]);
#if NR_RECEIVERS_PER_BLOCK == 48
    tmpX1.fixB(bSamples[buffer]);
#endif
#endif

    __syncthreads();

#pragma unroll
    for (unsigned minorTime = 0; minorTime < NR_TIMES_PER_BLOCK; minorTime += ((NR_BITS) == 4 ? 16 : 8)) {
      Afrag aFrag[nrFragmentsY];
      Bfrag bFrag[nrFragmentsX];

      for (unsigned y = 0; y < nrFragmentsY; y ++)
	load_matrix_sync(aFrag[y], &aSamples[buffer][recvYoffset + NR_RECEIVERS_PER_TCM_Y * y][0][minorTime][0], sizeof(aSamples[0][0][0]) * 8 / NR_BITS);

      for (unsigned x = 0; x < nrFragmentsX; x ++)
	load_matrix_sync(bFrag[x], &bSamples[buffer][recvXoffset + NR_RECEIVERS_PER_TCM_X * x][0][0][minorTime][0], sizeof(bSamples[0][0][0][0]) * 8 / NR_BITS);

      for (unsigned y = 0; y < nrFragmentsY; y ++)
	for (unsigned x = 0; x < nrFragmentsX; x ++)
	  mma_sync(sum[y][x], aFrag[y], bFrag[x], sum[y][x]);
    }
  }

#if 0
  for (unsigned y = 0; y < nrFragmentsY; y ++)
    for (unsigned x = 0; x < nrFragmentsX; x ++)
      for (unsigned i = 0; i < sum[0][0].num_storage_elements; i ++)
	if (sum[y][x].x[i] != 0)
#if NR_BITS == 4 || NR_BITS == 8
	  printf("blockIdx=(%d,%d,%d) tid=%u y=%u x=%u i=%u v=%d\n", blockIdx.x, blockIdx.y, blockIdx.z, tid, y, x, i, sum[y][x].x[i]);
#else
	  printf("blockIdx=(%d,%d,%d) tid=%u y=%u x=%u i=%u v=%f\n", blockIdx.x, blockIdx.y, blockIdx.z, tid, y, x, i, sum[y][x].x[i]);
#endif
#endif

#if defined PORTABLE
  __syncthreads();
#endif

  for (unsigned y = 0; y < nrFragmentsY; y ++)
    for (unsigned x = 0; x < nrFragmentsX; x ++)
      storeVisibilities(visibilities, channel, firstReceiverY, firstReceiverX, recvYoffset, recvXoffset, y, x, skipStoreYcheck, skipStoreXcheck, sum[y][x], scratchSpace, tid / warpSize);
}


extern "C" __global__
__launch_bounds__(NR_WARPS * 32, NR_RECEIVERS_PER_BLOCK == 32 ? 4 : 2)
void correlate(Visibilities visibilities, const Samples samples)
{
  const unsigned nrFragmentsY = NR_RECEIVERS_PER_BLOCK / NR_RECEIVERS_PER_TCM_Y / 2;

  unsigned block = blockIdx.x;

#if NR_RECEIVERS_PER_BLOCK == 32 || NR_RECEIVERS_PER_BLOCK == 48
  unsigned blockY = (unsigned) (sqrtf(8 * block + 1) - .99999f) / 2;
  unsigned blockX = block - blockY * (blockY + 1) / 2;
  unsigned firstReceiverX = blockX * NR_RECEIVERS_PER_BLOCK;
#elif NR_RECEIVERS_PER_BLOCK == 64
  unsigned blockY = (unsigned) sqrtf(block);
  unsigned blockX = block - blockY * blockY;
  unsigned firstReceiverX = blockX * (NR_RECEIVERS_PER_BLOCK / 2);
#endif
  unsigned firstReceiverY = blockY * NR_RECEIVERS_PER_BLOCK;

  union shared {
    struct {
      SharedData<>::Asamples aSamples;
      SharedData<NR_RECEIVERS_PER_BLOCK == 64 ? 32 : NR_RECEIVERS_PER_BLOCK>::Bsamples bSamples;
    } rectangle;
    struct {
      SharedData<>::Bsamples samples;
    } triangle;
    ScratchSpace scratchSpace[NR_WARPS];
  };

  // the following hack is necessary to run the correlator in the OpenCL environment,
  // as the maximum local memory size is 48K - 16 bytes.  Due to padding in bSamples,
  // the last 16 bytes are not used, so allocate 16 fewer bytes.
  __shared__ char rawbuffer[sizeof(union shared) - 16] __attribute__((aligned(16)));
  union shared &u = (union shared &) rawbuffer;

  if (firstReceiverX == firstReceiverY)
#if NR_RECEIVERS_PER_BLOCK == 32 || NR_RECEIVERS_PER_BLOCK == 48
    doCorrelateRectangle<nrFragmentsY, NR_RECEIVERS % NR_RECEIVERS_PER_BLOCK == 0, NR_RECEIVERS % NR_RECEIVERS_PER_BLOCK == 0, NR_RECEIVERS % NR_RECEIVERS_PER_BLOCK == 0, false>(visibilities, samples, firstReceiverY, firstReceiverX, u.rectangle.aSamples, u.rectangle.bSamples, u.scratchSpace);
#elif NR_RECEIVERS_PER_BLOCK == 64
    if (NR_RECEIVERS % NR_RECEIVERS_PER_BLOCK != 0 && (NR_RECEIVERS < NR_RECEIVERS_PER_BLOCK || firstReceiverX >= NR_RECEIVERS / NR_RECEIVERS_PER_BLOCK * NR_RECEIVERS_PER_BLOCK))
      doCorrelateTriangle<false>(visibilities, samples, firstReceiverX, 2 * threadIdx.z + threadIdx.y, 64 * threadIdx.z + 32 * threadIdx.y + threadIdx.x, u.triangle.samples, u.scratchSpace);
    else
      doCorrelateTriangle<true>(visibilities, samples, firstReceiverX, 2 * threadIdx.z + threadIdx.y, 64 * threadIdx.z + 32 * threadIdx.y + threadIdx.x, u.triangle.samples, u.scratchSpace);
#endif
#if NR_RECEIVERS % NR_RECEIVERS_PER_BLOCK != 0
  else if (NR_RECEIVERS < NR_RECEIVERS_PER_BLOCK || firstReceiverY >= NR_RECEIVERS / NR_RECEIVERS_PER_BLOCK * NR_RECEIVERS_PER_BLOCK)
    doCorrelateRectangle<(NR_RECEIVERS % NR_RECEIVERS_PER_BLOCK + 2 * NR_RECEIVERS_PER_TCM_Y - 1) / NR_RECEIVERS_PER_TCM_Y / 2, false, true, NR_RECEIVERS % (2 * NR_RECEIVERS_PER_TCM_Y) == 0, true>(visibilities, samples, firstReceiverY, firstReceiverX, u.rectangle.aSamples, u.rectangle.bSamples, u.scratchSpace);
#endif
  else
    doCorrelateRectangle<nrFragmentsY, true, true, true, true>(visibilities, samples, firstReceiverY, firstReceiverX, u.rectangle.aSamples, u.rectangle.bSamples, u.scratchSpace);
}

} // extern "C++"
