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
 * https://git.astron.nl/RD/tensor-core-correlator/-/blob/fbd512a3/libtcc/kernel/TCCorrelator.cu
 *
 * See https://developer.nvidia.com/gtc/2019/video/s9306 for a high-level overview.
 * Lower-level details are in the doc/xbgpu.tcc.rst (and built by Sphinx with
 * the rest of the documentation).
 *
 * It has been modified by SARAO:
 * - Wrap the file in extern "C++" to make it work with PyCUDA (see below)
 * - Add results to the output instead of overwriting, to allow accumulation
 *   across multiple calls; results use 64-bit integers to avoid overflow.
 * - Take the input axes in a different order.
 * - Remove the asynchronous copy code (it would not have worked well with
 *   the previous point).
 * - Guarantee 32-byte alignment of the shared data (required by
 *   load_matrix_sync / store_matrix_sync).
 * - Parallelise over multiple problem instances.
 * - Write the output
 * - Remove trailing whitespace.
 *
 * SARAO's modification is licenced as follows:
 *******************************************************************************
 * Copyright (c) 2020-2022, 2024, National Research Foundation (SARAO)
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
 */
extern "C++" {

#include <mma.h>

#define NR_BASELINES		 (NR_RECEIVERS * (NR_RECEIVERS + 1) / 2)
#define ALIGN(A,N)		 (((A)+(N)-1)/(N)*(N))

#define NR_TIMES_PER_BLOCK	 (128 / (NR_BITS))
#define NR_RECEIVERS_PER_TCM_X	 ((NR_BITS) == 4 ? 2 : 4)
#define NR_RECEIVERS_PER_TCM_Y	 8
#define NR_RECEIVERS_PER_BLOCK_X (NR_RECEIVERS_PER_BLOCK == 64 ? 32 : NR_RECEIVERS_PER_BLOCK)

#define COMPLEX			 2

#if __CUDA_ARCH__ < (NR_BITS == 4 ? 730 : NR_BITS == 8 ? 720 : NR_BITS == 16 ? 700 : 0)
#error this architecture has no suitable tensor cores
#endif

#if __CUDA_ARCH__ != 700 && __CUDA_ARCH__ != 720 && __CUDA_ARCH__ != 750 && __CUDA_ARCH__ != 800 && __CUDA_ARCH__ != 860 && __CUDA_ARCH__ != 870 && __CUDA_ARCH__ != 890 && __CUDA_ARCH__ != 900
#define PORTABLE // unknown architecture -> write visibilities in portable way (via shared memory)
#endif

#if NR_RECEIVERS_PER_BLOCK != 32 && NR_RECEIVERS_PER_BLOCK != 48 && NR_RECEIVERS_PER_BLOCK != 64
#error unsupported NR_RECEIVERS_PER_BLOCK
#endif

#if NR_SAMPLES_PER_CHANNEL % NR_TIMES_PER_BLOCK != 0
#error NR_SAMPLES_PER_CHANNEL should be a multiple of NR_TIMES_PER_BLOCK
#endif

#define MIN(A,B) ((A)<(B)?(A):(B))


inline __device__ unsigned laneid()
{
#if 0
  unsigned laneid;

  asm ("mov.u32 %0, %%laneid;" : "=r" (laneid));
  return laneid;
#else
  return threadIdx.x;
#endif
}


namespace nvcuda {
  namespace wmma {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 730
    template<> class fragment<matrix_a, 16, 8, 64, experimental::precision::s4, row_major> : public __frag_base<experimental::precision::s4, 32, 4> {};
    template<> class fragment<matrix_b, 16, 8, 64, experimental::precision::s4, col_major> : public __frag_base<experimental::precision::s4, 16, 2> {};
    template<> class fragment<accumulator, 16, 8, 64, int> : public __frag_base<int, 4> {};

    inline __device__ void mma_sync(fragment<accumulator, 16, 8, 64, int>& d,
				    const fragment<matrix_a, 16, 8, 64, experimental::precision::s4, row_major>& a,
				    const fragment<matrix_b, 16, 8, 64, experimental::precision::s4, col_major>& b,
				    const fragment<accumulator, 16, 8, 64, int>& c)
    {
      asm ("mma.sync.aligned.m16n8k64.row.col.satfinite.s32.s4.s4.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};" :
	   "=r" (d.x[0]), "=r" (d.x[1]), "=r" (d.x[2]), "=r" (d.x[3]) :
	   "r" (a.x[0]), "r" (a.x[1]), "r" (a.x[2]), "r" (a.x[3]),
	   "r" (b.x[0]), "r" (b.x[1]),
	   "r" (c.x[0]), "r" (c.x[1]), "r" (c.x[2]), "r" (c.x[3])
	  );
    }

    inline __device__ void load_matrix_sync(fragment<matrix_a, 16, 8, 64, experimental::precision::s4, row_major> &a, const void *p, unsigned ldm)
    {
      a.x[0] = ((const int *) p)[ldm / 8 * (laneid() / 4    ) + laneid() % 4    ];
      a.x[1] = ((const int *) p)[ldm / 8 * (laneid() / 4 + 8) + laneid() % 4    ];
      a.x[2] = ((const int *) p)[ldm / 8 * (laneid() / 4    ) + laneid() % 4 + 4];
      a.x[3] = ((const int *) p)[ldm / 8 * (laneid() / 4 + 8) + laneid() % 4 + 4];
    }

    inline __device__ void load_matrix_sync(fragment<matrix_b, 16, 8, 64, experimental::precision::s4, col_major> &b, const void *p, unsigned ldm)
    {
      b.x[0] = ((const int *) p)[ldm / 8 * (laneid() / 4) + laneid() % 4    ];
      b.x[1] = ((const int *) p)[ldm / 8 * (laneid() / 4) + laneid() % 4 + 4];
    }

    inline __device__ void store_matrix_sync(int *p, const fragment<accumulator, 16, 8, 64, int>& d, unsigned ldm, layout_t layout)
    {
      // FIXME: only row-major supported
      ((int2 *) p)[ldm / 2 * (laneid() / 4    ) + laneid() % 4] = make_int2(d.x[0], d.x[1]);
      ((int2 *) p)[ldm / 2 * (laneid() / 4 + 8) + laneid() % 4] = make_int2(d.x[2], d.x[3]);
    }
#endif
  }
}


using namespace nvcuda::wmma;

#if NR_BITS == 4
typedef char    Sample;
typedef int2    Visibility;
#elif NR_BITS == 8
typedef char2   Sample;
typedef int2    Visibility;
#elif NR_BITS == 16
typedef __half2 Sample;
typedef float2  Visibility;
#endif


inline __device__ Visibility operator += (Visibility &a, Visibility b)
{
  a.x += b.x, a.y += b.y;
  return a;
}


typedef Sample Samples[NR_RECEIVERS][NR_CHANNELS][NR_SAMPLES_PER_CHANNEL / NR_TIMES_PER_BLOCK][NR_TIMES_PER_BLOCK][NR_POLARIZATIONS];

#if !defined CUSTOM_STORE_VISIBILITY
typedef Visibility Visibilities[NR_CHANNELS][NR_BASELINES][NR_POLARIZATIONS][NR_POLARIZATIONS];
#endif


#if NR_BITS == 4
typedef fragment<matrix_a, 16, 8, 64, experimental::precision::s4, row_major> Afrag;
typedef fragment<matrix_b, 16, 8, 64, experimental::precision::s4, col_major> Bfrag;
typedef fragment<accumulator, 16, 8, 64, int>                                 Sum;
#elif NR_BITS == 8
typedef fragment<matrix_a, 16, 16, 16, signed char, row_major>               Afrag;
typedef fragment<matrix_b, 16, 16, 16, signed char, col_major>               Bfrag;
typedef fragment<accumulator, 16, 16, 16, int>                               Sum;
#elif NR_BITS == 16
typedef fragment<matrix_a, 16, 16, 16, __half, row_major>                    Afrag;
typedef fragment<matrix_b, 16, 16, 16, __half, col_major>                    Bfrag;
typedef fragment<accumulator, 16, 16, 16, float>                             Sum;
#endif


typedef Visibility ScratchSpace[NR_RECEIVERS_PER_TCM_Y][NR_POLARIZATIONS][NR_RECEIVERS_PER_TCM_X][NR_POLARIZATIONS];


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


#define READ_AHEAD        1
#define NR_SHARED_BUFFERS 2


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
  __device__ FetchData(unsigned loadRecv, unsigned loadTime)
  :
    loadRecv(loadRecv), loadTime(loadTime), data({0})
  {
  }

  __device__ void load(const Samples samples, unsigned channel, unsigned time, unsigned firstReceiver, bool skipLoadCheck = NR_RECEIVERS % NR_RECEIVERS_PER_BLOCK == 0)
  {
    if (skipLoadCheck || firstReceiver + loadRecv < NR_RECEIVERS)
    {
      data = * (T *) &samples[firstReceiver + loadRecv][channel][time][loadTime][0];
      //memcpy(&data, &samples[firstReceiver + loadRecv][channel][time][loadTime][0], sizeof(T));
    }
  }

  template <typename SharedData> __device__ void storeA(SharedData samples) const
  {
#pragma unroll
    for (unsigned i = 0; i < sizeof(T) / sizeof(Sample); i++)
      *(Sample *) &samples[loadRecv][i & 1][loadTime + (i >> 1)][0] = ((const Sample *) &data)[i];
  }

  template <typename SharedData> __device__ void storeB(SharedData samples) const
  {
    //* ((T *) &samples[loadRecv][loadPol][0][loadTime][0]) = data;
    //* ((T *) &samples[loadRecv][loadPol][1][loadTime][0]) = conj_perm(data);
    T tmp = conj_perm(data);
#pragma unroll
    for (unsigned i = 0; i < sizeof(T) / sizeof(Sample); i++)
    {
      unsigned time = loadTime + (i >> 1);
      *(Sample *) &samples[loadRecv][i & 1][0][time][0] = ((const Sample *) &data)[i];
      *(Sample *) &samples[loadRecv][i & 1][1][time][0] = ((const Sample *) &tmp)[i];
    }
  }

  unsigned loadRecv, loadTime;
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


#if defined CUSTOM_STORE_VISIBILITY
// Upstream pastes in CUSTOM_STORE_VISIBILITY here, but we just paste in
// the code we want here.
// CUSTOM_STORE_VISIBILITY
typedef long2 StoredVisibility;
typedef StoredVisibility Visibilities[NR_CHANNELS][NR_BASELINES][NR_POLARIZATIONS][NR_POLARIZATIONS];

inline __device__ StoredVisibility operator += (StoredVisibility &a, Visibility b)
{
  a.x += b.x;
  a.y += b.y;
  return a;
}

template <bool add> __device__ inline void storeVisibility(Visibilities visibilities, unsigned channel, unsigned baseline, unsigned polY, unsigned polX, Visibility visibility)
{
  // Ignore 'add': we always want it to be true.
  // NB: polX/polY are swapped compared to the upstream version
  visibilities[channel][baseline][polX][polY] += visibility;
}
#else

template <bool add> __device__ inline void storeVisibility(Visibilities visibilities, unsigned channel, unsigned baseline, unsigned polY, unsigned polX, Visibility visibility)
{
  if (add)
    visibilities[channel][baseline][polY][polX] += visibility;
  else
    visibilities[channel][baseline][polY][polX] =  visibility;
}

#endif


template <bool add, typename T> __device__ inline void storeVisibility(Visibilities visibilities, unsigned channel, unsigned baseline, unsigned recvY, unsigned recvX, unsigned tcY, unsigned tcX, unsigned polY, unsigned polX, bool skipCheckY, bool skipCheckX, T sumR, T sumI)
{
  if ((skipCheckY || recvY + tcY <= recvX + tcX) && (skipCheckX || recvX + tcX < NR_RECEIVERS))
    storeVisibility<add>(visibilities, channel, baseline + tcX * recvX + tcX * (tcX + 1) / 2 + tcY, polY, polX, make_complex(sumR, sumI));
}


template <bool add>__device__ inline void storeVisibilities(Visibilities visibilities, unsigned channel, unsigned firstReceiverY, unsigned firstReceiverX, unsigned y, unsigned x, bool skipCheckY, bool skipCheckX, const Sum &sum, ScratchSpace scratchSpace[], unsigned warp)
{
#if defined PORTABLE
 store_matrix_sync(&scratchSpace[warp][0][0][0][0].x, sum, NR_RECEIVERS_PER_TCM_X * NR_POLARIZATIONS * COMPLEX, mem_row_major);
  __syncwarp();

#if 0
  if (threadIdx.x == 0)
    for (unsigned _y = 0; _y < NR_RECEIVERS_PER_TCM_Y; _y ++)
      for (unsigned pol_y = 0; pol_y < NR_POLARIZATIONS; pol_y ++)
        for (unsigned _x = 0; _x < NR_RECEIVERS_PER_TCM_X; _x ++)
          for (unsigned pol_x = 0; pol_x < NR_POLARIZATIONS; pol_x ++)
            if (scratchSpace[warp][_y][pol_y][_x][pol_x].x != 0 || scratchSpace[warp][_y][pol_y][_x][pol_x].y != 0)
              printf("firstY=%u firstX=%u warp=%u y=%u x=%u _y=%u pol_y=%u _x=%u pol_x=%u val=(%f,%f)\n", firstReceiverY, firstReceiverX, warp, y, x, _y, pol_y, _x, pol_x, (float) scratchSpace[warp][_y][pol_y][_x][pol_x].x, (float) scratchSpace[warp][_y][pol_y][_x][pol_x].y);
#endif

#if NR_BITS == 4
  unsigned _y       = threadIdx.x >> 2;
  unsigned _x       = (threadIdx.x >> 1) & 1;
  unsigned polY     = threadIdx.x & 1;
#elif NR_BITS == 8 || NR_BITS == 16
  unsigned _y       = threadIdx.x >> 2;
  unsigned _x       = threadIdx.x & 3;
#endif

  unsigned recvY    = firstReceiverY + NR_RECEIVERS_PER_TCM_Y * y + _y;
  unsigned recvX    = firstReceiverX + NR_RECEIVERS_PER_TCM_X * x + _x;
  unsigned baseline = (recvX * (recvX + 1) / 2) + recvY;

  if ((skipCheckY || recvY <= recvX) && (skipCheckX || recvX < NR_RECEIVERS))
#if NR_BITS == 4
    for (unsigned polX = 0; polX < NR_POLARIZATIONS; polX ++)
      visibilities[channel][baseline][polY][polX] = scratchSpace[warp][_y][polY][_x][polX];
#elif NR_BITS == 8 || NR_BITS == 16
    for (unsigned polY = 0; polY < NR_POLARIZATIONS; polY ++)
      for (unsigned polX = 0; polX < NR_POLARIZATIONS; polX ++)
        visibilities[channel][baseline][polY][polX] = scratchSpace[warp][_y][polY][_x][polX];
#endif
#else
#if __CUDA_ARCH__ == 700 || (__CUDA_ARCH__ == 720 && NR_BITS == 16)
  unsigned recvY    = firstReceiverY + NR_RECEIVERS_PER_TCM_Y * y + ((threadIdx.x >> 3) & 2) + (threadIdx.x & 4);
  unsigned recvX    = firstReceiverX + NR_RECEIVERS_PER_TCM_X * x + ((threadIdx.x >> 2) & 2);
  unsigned polY     = threadIdx.x & 1;
  unsigned polX     = (threadIdx.x >> 1) & 1;
#elif (__CUDA_ARCH__ == 720 && NR_BITS == 8) || __CUDA_ARCH__ == 750 || __CUDA_ARCH__ == 800 || __CUDA_ARCH__ == 860 || __CUDA_ARCH__ == 870 || __CUDA_ARCH__ == 890 || __CUDA_ARCH__ == 900
  unsigned recvY    = firstReceiverY + NR_RECEIVERS_PER_TCM_Y * y + ((threadIdx.x >> 3) & 3);
  unsigned recvX    = firstReceiverX + NR_RECEIVERS_PER_TCM_X * x + ((threadIdx.x >> 1) & 1);
  unsigned polY     = (threadIdx.x >> 2) & 1;
  unsigned polX     = threadIdx.x & 1;
#endif

  unsigned baseline = (recvX * (recvX + 1) / 2) + recvY;

#if __CUDA_ARCH__ == 700 || (__CUDA_ARCH__ == 720 && NR_BITS == 16)
  storeVisibility<add>(visibilities, channel, baseline, recvY, recvX, 0, 0, polY, polX, skipCheckY, skipCheckX, sum.x[0], sum.x[1]);
  storeVisibility<add>(visibilities, channel, baseline, recvY, recvX, 0, 1, polY, polX, skipCheckY, skipCheckX, sum.x[4], sum.x[5]);
  storeVisibility<add>(visibilities, channel, baseline, recvY, recvX, 1, 0, polY, polX, skipCheckY, skipCheckX, sum.x[2], sum.x[3]);
  storeVisibility<add>(visibilities, channel, baseline, recvY, recvX, 1, 1, polY, polX, skipCheckY, skipCheckX, sum.x[6], sum.x[7]);
#elif (__CUDA_ARCH__ == 720 && NR_BITS == 8) || __CUDA_ARCH__ == 750 || __CUDA_ARCH__ == 800 || __CUDA_ARCH__ == 860 || __CUDA_ARCH__ == 870 || __CUDA_ARCH__ == 890 || __CUDA_ARCH__ == 900
  storeVisibility<add>(visibilities, channel, baseline, recvY, recvX, 0, 0, polY, polX, skipCheckY, skipCheckX, sum.x[0], sum.x[1]);
#if NR_BITS == 8 || NR_BITS == 16
  storeVisibility<add>(visibilities, channel, baseline, recvY, recvX, 0, 2, polY, polX, skipCheckY, skipCheckX, sum.x[4], sum.x[5]);
#endif
  storeVisibility<add>(visibilities, channel, baseline, recvY, recvX, 4, 0, polY, polX, skipCheckY, skipCheckX, sum.x[2], sum.x[3]);
#if NR_BITS == 8 || NR_BITS == 16
  storeVisibility<add>(visibilities, channel, baseline, recvY, recvX, 4, 2, polY, polX, skipCheckY, skipCheckX, sum.x[6], sum.x[7]);
#endif
#endif
#endif
}


#define NR_WARPS 4

#if NR_RECEIVERS_PER_BLOCK == 64

template <bool add, bool fullTriangle> __device__ void doCorrelateTriangle(Visibilities visibilities, const Samples samples, unsigned firstReceiver, unsigned warp, unsigned tid, SharedData<>::Bsamples &bSamples, ScratchSpace scratchSpace[NR_WARPS])
{
  const unsigned nrFragmentsX = 24 / NR_RECEIVERS_PER_TCM_X;
  const unsigned nrFragmentsY = 24 / NR_RECEIVERS_PER_TCM_Y;
  Sum            sum[nrFragmentsX * nrFragmentsY];

  for (auto &s : sum)
    fill_fragment(s, 0);

  unsigned channel = blockIdx.y;

  const uchar2 offsets[] = {
    make_uchar2( 0,  0),
    make_uchar2(16,  0),
    make_uchar2(40,  0),
    make_uchar2(40, 24),
  };

  unsigned recvXoffset = offsets[warp].x;
  unsigned recvYoffset = offsets[warp].y;

  FetchData<int4> tmp0((tid >> 2)                             , 32 / NR_BITS * (tid & 3));
  FetchData<int4> tmp1((tid >> 2) + NR_RECEIVERS_PER_BLOCK / 2, 32 / NR_BITS * (tid & 3));

  tmp0.load(samples, channel, 0, firstReceiver, fullTriangle);
  tmp1.load(samples, channel, 0, firstReceiver, fullTriangle);

  for (unsigned majorTime = 0; majorTime < NR_SAMPLES_PER_CHANNEL / NR_TIMES_PER_BLOCK; majorTime ++) {
    unsigned buffer = majorTime % NR_SHARED_BUFFERS;

    tmp0.storeB(bSamples[buffer]);
    tmp1.storeB(bSamples[buffer]);

    unsigned majorReadTime = majorTime + READ_AHEAD;

    if (majorReadTime < NR_SAMPLES_PER_CHANNEL / NR_TIMES_PER_BLOCK) {
      tmp0.load(samples, channel, majorReadTime, firstReceiver, fullTriangle);
      tmp1.load(samples, channel, majorReadTime, firstReceiver, fullTriangle);
    }

    __syncthreads();

#pragma unroll
    for (unsigned minorTime = 0; minorTime < NR_TIMES_PER_BLOCK; minorTime += ((NR_BITS) == 4 ? 32 : 8)) {
      Afrag aFrag;
      Bfrag bFrag[nrFragmentsX];

      if (warp != 0) {
	for (unsigned x = 0; x < nrFragmentsX; x ++)
	  load_matrix_sync(bFrag[x], &bSamples[buffer][recvXoffset + NR_RECEIVERS_PER_TCM_X * x][0][0][minorTime][0], sizeof(bSamples[0][0][0][0]) * 8 / NR_BITS);

	for (unsigned y = 0, i = 0; y < nrFragmentsY; y ++) {
	  load_matrix_sync(aFrag, &bSamples[buffer][recvYoffset + NR_RECEIVERS_PER_TCM_Y * y][0][0][minorTime][0], sizeof(bSamples[0][0][0]) * 8 / NR_BITS);

	  for (unsigned x = 0; x < nrFragmentsX; x ++, i ++)
	    mma_sync(sum[i], aFrag, bFrag[x], sum[i]);
	}
      } else {
	for (unsigned z = 0, i = 0; z < 3; z ++) {
	  for (unsigned x = 0; x < 16 / NR_RECEIVERS_PER_TCM_X; x ++)
	    load_matrix_sync(bFrag[x], &bSamples[buffer][/*recvXoffset*/ 24 * z + NR_RECEIVERS_PER_TCM_X * x][0][0][minorTime][0], sizeof(bSamples[0][0][0][0]) * 8 / NR_BITS);

	  for (unsigned y = 0; y < 2; y ++) {
	    load_matrix_sync(aFrag, &bSamples[buffer][/*recvYoffset*/ 24 * z + NR_RECEIVERS_PER_TCM_Y * y][0][0][minorTime][0], sizeof(bSamples[0][0][0]) * 8 / NR_BITS);

	    for (unsigned x = 8 * y / NR_RECEIVERS_PER_TCM_X; x < 16 / NR_RECEIVERS_PER_TCM_X; x ++, i ++)
	      mma_sync(sum[i], aFrag, bFrag[x], sum[i]);
	  }
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
	storeVisibilities<add>(visibilities, channel, firstReceiver + recvYoffset, firstReceiver + recvXoffset, y, x, y < 2 || x > (NR_BITS == 4 ? 4 : 2), fullTriangle, sum[i], scratchSpace, warp);
  else
    for (unsigned z = 0, i = 0; z < 3; z ++)
      for (unsigned y = 0; y < 2; y ++)
	for (unsigned x = 8 * y / NR_RECEIVERS_PER_TCM_X; x < 16 / NR_RECEIVERS_PER_TCM_X; x ++, i ++)
	  storeVisibilities<add>(visibilities, channel, firstReceiver + 24 * z, firstReceiver + 24 * z, y, x, (y + 1) * NR_RECEIVERS_PER_TCM_Y <= x * NR_RECEIVERS_PER_TCM_X, fullTriangle, sum[i], scratchSpace, warp);
}

#endif


template <bool add, unsigned nrFragmentsY, unsigned nrFragmentsX, bool skipLoadYcheck, bool skipLoadXcheck, bool skipStoreYcheck, bool skipStoreXcheck> __device__ void doCorrelateRectangle(Visibilities visibilities, const Samples samples, unsigned firstReceiverY, unsigned firstReceiverX, SharedData<>::Asamples &aSamples, SharedData<NR_RECEIVERS_PER_BLOCK_X>::Bsamples &bSamples, ScratchSpace scratchSpace[NR_WARPS])
{
  Sum sum[nrFragmentsY][nrFragmentsX];

  for (unsigned y = 0; y < nrFragmentsY; y ++)
    for (unsigned x = 0; x < nrFragmentsX; x ++)
      fill_fragment(sum[y][x], 0);

  unsigned tid     = warpSize * (blockDim.y * threadIdx.z + threadIdx.y) + threadIdx.x;
  unsigned channel = blockIdx.y;

  unsigned recvXoffset = nrFragmentsX * NR_RECEIVERS_PER_TCM_X * threadIdx.y;
  unsigned recvYoffset = nrFragmentsY * NR_RECEIVERS_PER_TCM_Y * threadIdx.z;

  FetchData<int4> tmpY0((tid >> 2)     , 32 / NR_BITS * (tid & 3));
  FetchData<int4> tmpX0((tid >> 2)     , 32 / NR_BITS * (tid & 3));
#if NR_RECEIVERS_PER_BLOCK == 48
  FetchData<int2> tmpY1((tid >> 3) + 32, 16 / NR_BITS * (tid & 7));
  FetchData<int2> tmpX1((tid >> 3) + 32, 16 / NR_BITS * (tid & 7));
#elif NR_RECEIVERS_PER_BLOCK == 64
  FetchData<int4> tmpY1((tid >> 2) + 32, 32 / NR_BITS * (tid & 3));
#endif

  tmpY0.load(samples, channel, 0, firstReceiverY, skipLoadYcheck);
#if NR_RECEIVERS_PER_BLOCK == 48 || NR_RECEIVERS_PER_BLOCK == 64
  tmpY1.load(samples, channel, 0, firstReceiverY, skipLoadYcheck);
#endif
  tmpX0.load(samples, channel, 0, firstReceiverX, skipLoadXcheck);
#if NR_RECEIVERS_PER_BLOCK == 48
  tmpX1.load(samples, channel, 0, firstReceiverX, skipLoadXcheck);
#endif

  for (unsigned majorTime = 0; majorTime < NR_SAMPLES_PER_CHANNEL / NR_TIMES_PER_BLOCK; majorTime ++) {
    unsigned buffer = majorTime % NR_SHARED_BUFFERS;

    tmpY0.storeA(aSamples[buffer]);
#if NR_RECEIVERS_PER_BLOCK == 48 || NR_RECEIVERS_PER_BLOCK == 64
    tmpY1.storeA(aSamples[buffer]);
#endif
    tmpX0.storeB(bSamples[buffer]);
#if NR_RECEIVERS_PER_BLOCK == 48
    tmpX1.storeB(bSamples[buffer]);
#endif

    unsigned majorReadTime = majorTime + READ_AHEAD;

    if (majorReadTime < NR_SAMPLES_PER_CHANNEL / NR_TIMES_PER_BLOCK) {
      tmpY0.load(samples, channel, majorReadTime, firstReceiverY, skipLoadYcheck);
#if NR_RECEIVERS_PER_BLOCK == 48 || NR_RECEIVERS_PER_BLOCK == 64
      tmpY1.load(samples, channel, majorReadTime, firstReceiverY, skipLoadYcheck);
#endif
      tmpX0.load(samples, channel, majorReadTime, firstReceiverX, skipLoadXcheck);
#if NR_RECEIVERS_PER_BLOCK == 48
      tmpX1.load(samples, channel, majorReadTime, firstReceiverX, skipLoadXcheck);
#endif
    }

    __syncthreads();

#pragma unroll
    for (unsigned minorTime = 0; minorTime < NR_TIMES_PER_BLOCK; minorTime += ((NR_BITS) == 4 ? 32 : 8)) {
      Afrag aFrag;
      Bfrag bFrag[nrFragmentsX];

      for (unsigned x = 0; x < nrFragmentsX; x ++)
	load_matrix_sync(bFrag[x], &bSamples[buffer][recvXoffset + NR_RECEIVERS_PER_TCM_X * x][0][0][minorTime][0], sizeof(bSamples[0][0][0][0]) * 8 / NR_BITS);

      for (unsigned y = 0; y < nrFragmentsY; y ++) {
	load_matrix_sync(aFrag, &aSamples[buffer][recvYoffset + NR_RECEIVERS_PER_TCM_Y * y][0][minorTime][0], sizeof(aSamples[0][0][0]) * 8 / NR_BITS);

	for (unsigned x = 0; x < nrFragmentsX; x ++)
	  mma_sync(sum[y][x], aFrag, bFrag[x], sum[y][x]);
      }
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
      storeVisibilities<add>(visibilities, channel, firstReceiverY + recvYoffset, firstReceiverX + recvXoffset, y, x, skipStoreYcheck, skipStoreXcheck, sum[y][x], scratchSpace, tid / warpSize);
}


union shared {
  struct {
    alignas(32) SharedData<>::Asamples aSamples;
    alignas(32) SharedData<NR_RECEIVERS_PER_BLOCK_X>::Bsamples bSamples;
  } rectangle;
  struct {
    alignas(32) SharedData<>::Bsamples samples;
  } triangle;
  ScratchSpace scratchSpace[NR_WARPS];
};


template <bool add> __device__ void doCorrelate(Visibilities visibilities, const Samples samples, union shared &u)
{
  constexpr unsigned nrFragmentsX = NR_RECEIVERS_PER_BLOCK_X / NR_RECEIVERS_PER_TCM_X / 2;
  constexpr unsigned nrFragmentsY = NR_RECEIVERS_PER_BLOCK   / NR_RECEIVERS_PER_TCM_Y / 2;

  unsigned block = blockIdx.x;

#if NR_RECEIVERS_PER_BLOCK == 32 || NR_RECEIVERS_PER_BLOCK == 48
  unsigned blockX = (unsigned) (sqrtf(8 * block + 1) - .99999f) / 2;
  unsigned blockY = block - blockX * (blockX + 1) / 2;
  unsigned firstReceiverY = blockY * NR_RECEIVERS_PER_BLOCK;
  unsigned firstReceiverX = blockX * NR_RECEIVERS_PER_BLOCK;
#elif NR_RECEIVERS_PER_BLOCK == 64
  unsigned blockX = (unsigned) sqrtf(block);
  unsigned blockY = block - blockX * blockX;
  unsigned firstReceiverY = blockY / 2 * NR_RECEIVERS_PER_BLOCK;
  //unsigned firstReceiverX = (2 * blockX + blockY % 2) * (NR_RECEIVERS_PER_BLOCK / 2);
  unsigned firstReceiverX = blockX * NR_RECEIVERS_PER_BLOCK + (blockY % 2) * NR_RECEIVERS_PER_BLOCK / 2;

  if (firstReceiverX >= NR_RECEIVERS)
    return;
#endif

  if (firstReceiverX == firstReceiverY)
#if NR_RECEIVERS_PER_BLOCK == 32 || NR_RECEIVERS_PER_BLOCK == 48
    doCorrelateRectangle<add, nrFragmentsY, nrFragmentsX, NR_RECEIVERS % NR_RECEIVERS_PER_BLOCK == 0, NR_RECEIVERS % NR_RECEIVERS_PER_BLOCK == 0, false, NR_RECEIVERS % NR_RECEIVERS_PER_BLOCK == 0>(visibilities, samples, firstReceiverY, firstReceiverX, u.rectangle.aSamples, u.rectangle.bSamples, u.scratchSpace); // TODO: smaller nrFragments[XY]
#elif NR_RECEIVERS_PER_BLOCK == 64
    if (NR_RECEIVERS % NR_RECEIVERS_PER_BLOCK != 0 && (NR_RECEIVERS < NR_RECEIVERS_PER_BLOCK || firstReceiverX >= NR_RECEIVERS / NR_RECEIVERS_PER_BLOCK * NR_RECEIVERS_PER_BLOCK))
      doCorrelateTriangle<add, false>(visibilities, samples, firstReceiverX, 2 * threadIdx.z + threadIdx.y, 64 * threadIdx.z + 32 * threadIdx.y + threadIdx.x, u.triangle.samples, u.scratchSpace);
    else
      doCorrelateTriangle<add, true>(visibilities, samples, firstReceiverX, 2 * threadIdx.z + threadIdx.y, 64 * threadIdx.z + 32 * threadIdx.y + threadIdx.x, u.triangle.samples, u.scratchSpace);
#endif
#if NR_RECEIVERS % NR_RECEIVERS_PER_BLOCK_X != 0
  else if (firstReceiverX >= NR_RECEIVERS / NR_RECEIVERS_PER_BLOCK_X * NR_RECEIVERS_PER_BLOCK_X)
    doCorrelateRectangle<add, nrFragmentsY, (NR_RECEIVERS % NR_RECEIVERS_PER_BLOCK_X + 2 * NR_RECEIVERS_PER_TCM_X - 1) / (2 * NR_RECEIVERS_PER_TCM_X), true, false, true, NR_RECEIVERS % (2 * NR_RECEIVERS_PER_TCM_X) == 0>(visibilities, samples, firstReceiverY, firstReceiverX, u.rectangle.aSamples, u.rectangle.bSamples, u.scratchSpace);
#endif
  else
    doCorrelateRectangle<add, nrFragmentsY, nrFragmentsX, true, true, true, true>(visibilities, samples, firstReceiverY, firstReceiverX, u.rectangle.aSamples, u.rectangle.bSamples, u.scratchSpace);

#if 0
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    //((uint64_t *) visibilities)[blockIdx. y * 81 + blockIdx.x] = clock64();
    unsigned sm;
    asm("mov.u32 %0, %smid;" : "=r"(sm) );
    printf("block %u channel %u sm %u at %llu\n", blockIdx.x, blockIdx.y, sm, clock64());
  }
#endif
}


extern "C" __global__
__launch_bounds__(NR_WARPS * 32, NR_RECEIVERS_PER_BLOCK == 32 ? 4 : 2)
void correlate(Visibilities *visibilities, const Samples *samples, unsigned batchOffset)
{
  const bool add = true;
  // the following hack is necessary to run the correlator in the OpenCL environment,
  // as the maximum local memory size is 48K - 16 bytes.  Due to padding in bSamples,
  // the last 16 bytes are not used, so allocate 16 fewer bytes.
  __shared__ char rawbuffer[sizeof(union shared) - 16] __attribute__((aligned(16)));
  union shared &u = (union shared &) rawbuffer;

  unsigned batch = batchOffset + blockIdx.z;
  visibilities += batch;
  samples += batch;

  if (add)
    doCorrelate<true>(*visibilities, *samples, u);
  else
    doCorrelate<false>(*visibilities, *samples, u);
}


// Clamp x to [-INT_MAX, INT_MAX] and return true if it was clamped
__device__ bool saturate(long *x)
{
  if (*x < -INT_MAX) {
    *x = -INT_MAX;
    return true;
  }
  if (*x > INT_MAX) {
    *x = INT_MAX;
    return true;
  }
  return false;
}

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

// TODO: generalise to other values of NR_BITS
extern "C" __global__
__launch_bounds__(NR_WARPS * 32)
void reduce(int2 * __restrict__ out, unsigned int * __restrict__ out_saturated, const long2 * __restrict__ in, unsigned batches)
{
  namespace cg = cooperative_groups;
  const unsigned int stride = NR_CHANNELS * NR_BASELINES * NR_POLARIZATIONS * NR_POLARIZATIONS;
  const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= stride)
    return;
  long2 sum = make_long2(0, 0);
  for (unsigned i = 0; i < batches; i++) {
    long2 value = in[i * stride + idx];
    sum.x += value.x;
    sum.y += value.y;
  }
  // Apply saturation (note: | not || to avoid short-circuiting)
  unsigned int saturated = saturate(&sum.x) | saturate(&sum.y);
  /* Do some in-kernel reduction of the saturation count to reduce the number
   * of atomic operations. Ideally this would be done for the whole thread
   * block rather than just warp, but CUDA doesn't yet make that a one-liner,
   * and the performance is not that critical.
   */
  auto group = cg::tiled_partition<32>(cg::this_thread_block());
  saturated = cg::reduce(group, saturated, cg::plus<unsigned int>());
  out[idx] = make_complex((int) sum.x, (int) sum.y);
  if (group.thread_rank() == 0)
    atomicAdd(out_saturated, saturated);
}

} // extern "C++"
