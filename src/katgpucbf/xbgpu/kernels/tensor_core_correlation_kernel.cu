/* Yes, the first line of a file is a closing brace. Dont run away, dont panic, there is a reason for this. By default
 * pycuda surrounds a source file with `extern "C" {}`. However if #include <mma.h> is within an `extern "C"{}` block
 * it throws all sorts of errors. The actual solution to this is to pass a the `no_extern_c=True` argument to the 
 * `pycuda.compiler.SourceModule(...)` function, however, katsdpsigproc does not provide the ability to do this at the 
 * moment. Therefore the quickest fix is to just add a } at the start of the file to close the extern function. The last
 * } in this file has also been commented out to compensate for the closing brace of the added `extern "C"{}`.
 *
 * When this code gets closer to production, the suggested fix is to modify the accel.build() and context.compile() 
 * functions in katsdpsigproc to take a no_extern_c flag as these ones are the ones that will call the SourceModule(...)
 * constructor.
 */
}
#include <mma.h>

<%include file="/port.mako"/>
#define NR_STATIONS ${dual_pol_ants}
#define NR_STATIONS_PER_BLOCK ${ants_per_block}
#define NR_BITS ${sample_bitwidth}
#define NR_CHANNELS ${channels}
#define NR_SAMPLES_PER_CHANNEL ${samples_per_channel}
#define NR_POLARIZATIONS ${polarizastions}

#define NR_BASELINES		(NR_STATIONS * (NR_STATIONS + 1) / 2)
#define ALIGN(A,N)		(((A)+(N)-1)/(N)*(N))

#define NR_TIMES_PER_BLOCK	(128 / (NR_BITS))
#define NR_STATIONS_PER_TCM_X	((NR_BITS) == 4 ? 2 : 4)
#define NR_STATIONS_PER_TCM_Y	((NR_BITS) == 4 ? 4 : 8)

#define COMPLEX			2

#if __CUDA_ARCH__ < (NR_BITS == 4 ? 730 : NR_BITS == 8 ? 720 : NR_BITS == 16 ? 700 : 0)
#error this architecture has no tensor cores
#endif

#if __CUDA_ARCH__ != 700 && __CUDA_ARCH__ != 720 && __CUDA_ARCH__ != 750
#define PORTABLE // unknown architecture -> write visibilities in portable way (via shared memory)
#endif

#if NR_STATIONS_PER_BLOCK != 48 && NR_STATIONS_PER_BLOCK != 64
#error unsupported NR_STATIONS_PER_BLOCK
#endif


using namespace nvcuda::wmma;

#if NR_BITS == 4
typedef char    Samples[NR_CHANNELS][NR_SAMPLES_PER_CHANNEL / NR_TIMES_PER_BLOCK][NR_STATIONS][NR_POLARIZATIONS][NR_TIMES_PER_BLOCK];
typedef int2    Visibilities[NR_CHANNELS][NR_BASELINES][NR_POLARIZATIONS][NR_POLARIZATIONS];
#elif NR_BITS == 8
typedef char2   Samples[NR_CHANNELS][NR_SAMPLES_PER_CHANNEL / NR_TIMES_PER_BLOCK][NR_STATIONS][NR_POLARIZATIONS][NR_TIMES_PER_BLOCK];
typedef int2    Visibilities[NR_CHANNELS][NR_BASELINES][NR_POLARIZATIONS][NR_POLARIZATIONS];
#elif NR_BITS == 16
typedef __half2 Samples[NR_CHANNELS][NR_SAMPLES_PER_CHANNEL / NR_TIMES_PER_BLOCK][NR_STATIONS][NR_POLARIZATIONS][NR_TIMES_PER_BLOCK];
typedef float2  Visibilities[NR_CHANNELS][NR_BASELINES][NR_POLARIZATIONS][NR_POLARIZATIONS];
#endif


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
  return ((v & 0x0F0F0F0F) << 4) | (__vnegss4(v >> 4) & 0x0F0F0F0F);
#elif NR_BITS == 8
  return __byte_perm(v, __vnegss4(v), 0x2705);
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


template <unsigned nrStationsPerBlock = NR_STATIONS_PER_BLOCK> struct SharedData
{
#if NR_BITS == 4
  typedef char        Asamples[nrStationsPerBlock][NR_POLARIZATIONS][NR_TIMES_PER_BLOCK][1];
  typedef char        Bsamples[nrStationsPerBlock][NR_POLARIZATIONS][COMPLEX][NR_TIMES_PER_BLOCK + 16][1];
#elif NR_BITS == 8
  typedef signed char Asamples[nrStationsPerBlock][NR_POLARIZATIONS][NR_TIMES_PER_BLOCK][COMPLEX];
  typedef signed char Bsamples[nrStationsPerBlock][NR_POLARIZATIONS][COMPLEX][NR_TIMES_PER_BLOCK + 8][COMPLEX];
#elif NR_BITS == 16
  typedef __half      Asamples[nrStationsPerBlock][NR_POLARIZATIONS][NR_TIMES_PER_BLOCK][COMPLEX];
  typedef __half      Bsamples[nrStationsPerBlock][NR_POLARIZATIONS][COMPLEX][NR_TIMES_PER_BLOCK + 4][COMPLEX];
#endif
};


template <typename T> struct FetchData
{
  __device__ FetchData(unsigned loadStat, unsigned loadPol, unsigned loadTime)
  :
    loadStat(loadStat), loadPol(loadPol), loadTime(loadTime), data({0})
  {
  }

  __device__ void load(const Samples samples, unsigned channel, unsigned time, unsigned firstStation, bool skipLoadCheck = NR_STATIONS % NR_STATIONS_PER_BLOCK == 0)
  {
    if (skipLoadCheck || firstStation + loadStat < NR_STATIONS)
      data = * (T *) &samples[channel][time][firstStation + loadStat][loadPol][loadTime];
  }

  template <typename SharedData> __device__ void storeA(SharedData samples) const
  {
    * ((T *) &samples[loadStat][loadPol][loadTime][0]) = data;
  }

  template <typename SharedData> __device__ void storeB(SharedData samples) const
  {
    * ((T *) &samples[loadStat][loadPol][0][loadTime][0]) = data;
    * ((T *) &samples[loadStat][loadPol][1][loadTime][0]) = conj_perm(data);
  }

  unsigned loadStat, loadPol, loadTime;
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


template <typename T> __device__ inline void storeVisibility(Visibilities visibilities, unsigned channel, unsigned baseline, unsigned statY, unsigned statX, unsigned tcY, unsigned tcX, unsigned polY, unsigned polX, bool skipCheckY, bool skipCheckX, T sumR, T sumI)
{
  if ((skipCheckX || statX + tcX <= statY + tcY) && (skipCheckY || statY + tcY < NR_STATIONS))
    visibilities[channel][baseline + tcY * statY + tcY * (tcY + 1) / 2 + tcX][polY][polX] = make_complex(sumR, sumI);
}


__device__ inline void storeVisibilities(Visibilities visibilities, unsigned channel, unsigned firstStationY, unsigned firstStationX, unsigned statYoffset, unsigned statXoffset, unsigned y, unsigned x, bool skipCheckY, bool skipCheckX, const Sum &sum, ScratchSpace scratchSpace[], unsigned warp)
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
              printf("firstY=%u firstX=%u warp=%u y=%u x=%u _y=%u pol_y=%u _x=%u pol_x=%u val=(%f,%f)\n", firstStationY, firstStationX, warp, y, x, _y, pol_y, _x, pol_x, scratchSpace[warp][_y][pol_y][_x][pol_x].x, scratchSpace[warp][_y][pol_y][_x][pol_x].y);
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

  unsigned statY    = firstStationY + statYoffset + NR_STATIONS_PER_TCM_Y * y + _y;
  unsigned statX    = firstStationX + statXoffset + NR_STATIONS_PER_TCM_X * x + _x;
  unsigned baseline = (statY * (statY + 1) / 2) + statX;

  if ((skipCheckX || statX <= statY) && (skipCheckY || statY < NR_STATIONS))
#if NR_BITS == 4
    visibilities[channel][baseline][polY][polX] = scratchSpace[warp][_y][polY][_x][polX];
#elif NR_BITS == 8 || NR_BITS == 16
    for (unsigned polY = 0; polY < NR_POLARIZATIONS; polY ++)
      for (unsigned polX = 0; polX < NR_POLARIZATIONS; polX ++)
        visibilities[channel][baseline][polY][polX] = scratchSpace[warp][_y][polY][_x][polX];
#endif
#else
#if __CUDA_ARCH__ == 700 || (__CUDA_ARCH__ == 720 && NR_BITS == 16)
  unsigned statY    = firstStationY + statYoffset + NR_STATIONS_PER_TCM_Y * y + ((threadIdx.x >> 3) & 2) + (threadIdx.x & 4);
  unsigned statX    = firstStationX + statXoffset + NR_STATIONS_PER_TCM_X * x + ((threadIdx.x >> 2) & 2);
  unsigned polY     = threadIdx.x & 1;
  unsigned polX     = (threadIdx.x >> 1) & 1;
#elif (__CUDA_ARCH__ == 720 && NR_BITS == 8) || __CUDA_ARCH__ == 750
  unsigned statY    = firstStationY + statYoffset + NR_STATIONS_PER_TCM_Y * y + ((threadIdx.x >> 3) & 3);
  unsigned statX    = firstStationX + statXoffset + NR_STATIONS_PER_TCM_X * x + ((threadIdx.x >> 1) & 1);
  unsigned polY     = (threadIdx.x >> 2) & 1;
  unsigned polX     = threadIdx.x & 1;
#endif

  unsigned baseline = (statY * (statY + 1) / 2) + statX;

#if __CUDA_ARCH__ == 700 || (__CUDA_ARCH__ == 720 && NR_BITS == 16)
  storeVisibility(visibilities, channel, baseline, statY, statX, 0, 0, polY, polX, skipCheckY, skipCheckX, sum.x[0], sum.x[1]);
  storeVisibility(visibilities, channel, baseline, statY, statX, 0, 1, polY, polX, skipCheckY, skipCheckX, sum.x[4], sum.x[5]);
  storeVisibility(visibilities, channel, baseline, statY, statX, 1, 0, polY, polX, skipCheckY, skipCheckX, sum.x[2], sum.x[3]);
  storeVisibility(visibilities, channel, baseline, statY, statX, 1, 1, polY, polX, skipCheckY, skipCheckX, sum.x[6], sum.x[7]);
#elif (__CUDA_ARCH__ == 720 && NR_BITS == 8) || __CUDA_ARCH__ == 750
  storeVisibility(visibilities, channel, baseline, statY, statX, 0, 0, polY, polX, skipCheckY, skipCheckX, sum.x[0], sum.x[1]);
#if NR_BITS == 8 || NR_BITS == 16
  storeVisibility(visibilities, channel, baseline, statY, statX, 0, 2, polY, polX, skipCheckY, skipCheckX, sum.x[4], sum.x[5]);
  storeVisibility(visibilities, channel, baseline, statY, statX, 4, 0, polY, polX, skipCheckY, skipCheckX, sum.x[2], sum.x[3]);
  storeVisibility(visibilities, channel, baseline, statY, statX, 4, 2, polY, polX, skipCheckY, skipCheckX, sum.x[6], sum.x[7]);
#endif
#endif
#endif
}


#define NR_WARPS 4

#if NR_STATIONS_PER_BLOCK == 64

template <bool fullTriangle> __device__ void doCorrelateTriangle(Visibilities visibilities, const Samples samples, unsigned firstStation, unsigned warp, unsigned tid, SharedData<>::Bsamples &bSamples, ScratchSpace scratchSpace[NR_WARPS])
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

  unsigned statXoffset = offsets[warp].x;
  unsigned statYoffset = offsets[warp].y;

  FetchData<int4> tmp0((tid >> 2)                            , (tid >> 1) & 1, 64 / NR_BITS * (tid & 1));
  FetchData<int4> tmp1((tid >> 2) + NR_STATIONS_PER_BLOCK / 2, (tid >> 1) & 1, 64 / NR_BITS * (tid & 1));

  tmp0.load(samples, channel, 0, firstStation, fullTriangle);
  tmp1.load(samples, channel, 0, firstStation, fullTriangle);

#pragma unroll 1
  for (unsigned majorTime = 0; majorTime < NR_SAMPLES_PER_CHANNEL / NR_TIMES_PER_BLOCK; majorTime ++) {
    __syncthreads();

    tmp0.storeB(bSamples);
    tmp1.storeB(bSamples);

    if (majorTime + 1 != NR_SAMPLES_PER_CHANNEL / NR_TIMES_PER_BLOCK) {
      tmp0.load(samples, channel, majorTime + 1, firstStation, fullTriangle);
      tmp1.load(samples, channel, majorTime + 1, firstStation, fullTriangle);
    }

    __syncthreads();

    for (unsigned minorTime = 0; minorTime < NR_TIMES_PER_BLOCK; minorTime += ((NR_BITS) == 4 ? 16 : 8)) {
      Afrag aFrag[nrFragmentsY];
      Bfrag bFrag[nrFragmentsX];

      if (warp != 0) {
	for (unsigned y = 0; y < nrFragmentsY; y ++)
	  load_matrix_sync(aFrag[y], &bSamples[statYoffset + NR_STATIONS_PER_TCM_Y * y][0][0][minorTime][0], sizeof(bSamples[0][0]) * 8 / NR_BITS);
											    
	for (unsigned x = 0; x < nrFragmentsX; x ++)
	  load_matrix_sync(bFrag[x], &bSamples[statXoffset + NR_STATIONS_PER_TCM_X * x][0][0][minorTime][0], sizeof(bSamples[0][0][0]) * 8 / NR_BITS);

	for (unsigned y = 0, i = 0; y < nrFragmentsY; y ++)
	  for (unsigned x = 0; x < nrFragmentsX; x ++, i ++)
	    mma_sync(sum[i], aFrag[y], bFrag[x], sum[i]);
      } else {
	for (unsigned z = 0, i = 0; z < 3; z ++) {
	  for (unsigned y = 0; y < (NR_BITS == 4 ? 4 : 2); y ++)
	    load_matrix_sync(aFrag[y], &bSamples[/*statYoffset*/ 24 * z + NR_STATIONS_PER_TCM_Y * y][0][0][minorTime][0], sizeof(bSamples[0][0]) * 8 / NR_BITS);
											    
	  for (unsigned x = 0; x < (NR_BITS == 4 ? 8 : 4); x ++)
	    load_matrix_sync(bFrag[x], &bSamples[/*statXoffset*/ 24 * z + NR_STATIONS_PER_TCM_X * x][0][0][minorTime][0], sizeof(bSamples[0][0][0]) * 8 / NR_BITS);

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
	storeVisibilities(visibilities, channel, firstStation, firstStation, statYoffset, statXoffset, y, x, fullTriangle, x < 2 * y + (NR_BITS == 4 ? 8 : 4), sum[i], scratchSpace, warp);
  else
    for (unsigned z = 0, i = 0; z < 3; z ++)
      for (unsigned y = 0; y < (NR_BITS == 4 ? 4 : 2); y ++)
	for (unsigned x = 0; x < 2 * y + 2; x ++, i ++)
	  storeVisibilities(visibilities, channel, firstStation, firstStation, 24 * z, 24 * z, y, x, fullTriangle, x < 2 * y, sum[i], scratchSpace, warp);
}

#endif


template <unsigned nrFragmentsY, bool skipLoadYcheck, bool skipLoadXcheck, bool skipStoreYcheck, bool skipStoreXcheck> __device__ void doCorrelateRectangle(Visibilities visibilities, const Samples samples, unsigned firstStationY, unsigned firstStationX, SharedData<>::Asamples &aSamples, SharedData<NR_STATIONS_PER_BLOCK == 64 ? 32 : NR_STATIONS_PER_BLOCK>::Bsamples &bSamples, ScratchSpace scratchSpace[NR_WARPS])
{
  const unsigned nrFragmentsX = NR_STATIONS_PER_BLOCK / NR_STATIONS_PER_TCM_X / 2 / (NR_STATIONS_PER_BLOCK == 64 ? 2 : 1);

  Sum sum[nrFragmentsY][nrFragmentsX];

  for (unsigned y = 0; y < nrFragmentsY; y ++)
    for (unsigned x = 0; x < nrFragmentsX; x ++)
      fill_fragment(sum[y][x], 0);

  unsigned tid     = warpSize * (blockDim.y * threadIdx.z + threadIdx.y) + threadIdx.x;
  unsigned channel = blockIdx.y;

  unsigned statXoffset = nrFragmentsX * NR_STATIONS_PER_TCM_X * threadIdx.y;
  unsigned statYoffset = nrFragmentsY * NR_STATIONS_PER_TCM_Y * threadIdx.z;

  FetchData<int4> tmpY0((tid >> 2)     , (tid >> 1) & 1, 64 / NR_BITS * (tid & 1));
  FetchData<int4> tmpX0((tid >> 2)     , (tid >> 1) & 1, 64 / NR_BITS * (tid & 1));
#if NR_STATIONS_PER_BLOCK == 48
  FetchData<int2> tmpY1((tid >> 3) + 32, (tid >> 2) & 1, 32 / NR_BITS * (tid & 3));
  FetchData<int2> tmpX1((tid >> 3) + 32, (tid >> 2) & 1, 32 / NR_BITS * (tid & 3));
#elif NR_STATIONS_PER_BLOCK == 64
  FetchData<int4> tmpY1((tid >> 2) + 32, (tid >> 1) & 1, 64 / NR_BITS * (tid & 1));
#endif

  tmpY0.load(samples, channel, 0, firstStationY, skipLoadYcheck);
  tmpY1.load(samples, channel, 0, firstStationY, skipLoadYcheck);
  tmpX0.load(samples, channel, 0, firstStationX, skipLoadXcheck);
#if NR_STATIONS_PER_BLOCK == 48
  tmpX1.load(samples, channel, 0, firstStationX, skipLoadXcheck);
#endif

#pragma unroll 1
  for (unsigned majorTime = 0; majorTime < NR_SAMPLES_PER_CHANNEL / NR_TIMES_PER_BLOCK; majorTime ++) {
    __syncthreads();

    tmpY0.storeA(aSamples);
    tmpY1.storeA(aSamples);
    tmpX0.storeB(bSamples);
#if NR_STATIONS_PER_BLOCK == 48
    tmpX1.storeB(bSamples);
#endif

    if (majorTime + 1 != NR_SAMPLES_PER_CHANNEL / NR_TIMES_PER_BLOCK) {
      tmpY0.load(samples, channel, majorTime + 1, firstStationY, skipLoadYcheck);
      tmpY1.load(samples, channel, majorTime + 1, firstStationY, skipLoadYcheck);
      tmpX0.load(samples, channel, majorTime + 1, firstStationX, skipLoadXcheck);
#if NR_STATIONS_PER_BLOCK == 48
      tmpX1.load(samples, channel, majorTime + 1, firstStationX, skipLoadXcheck);
#endif
    }

    __syncthreads();

    for (unsigned minorTime = 0; minorTime < NR_TIMES_PER_BLOCK; minorTime += ((NR_BITS) == 4 ? 16 : 8)) {
      Afrag aFrag[nrFragmentsY];
      Bfrag bFrag[nrFragmentsX];

      for (unsigned y = 0; y < nrFragmentsY; y ++)
	load_matrix_sync(aFrag[y], &aSamples[statYoffset + NR_STATIONS_PER_TCM_Y * y][0][minorTime][0], sizeof(aSamples[0][0]) * 8 / NR_BITS);

      for (unsigned x = 0; x < nrFragmentsX; x ++)
	load_matrix_sync(bFrag[x], &bSamples[statXoffset + NR_STATIONS_PER_TCM_X * x][0][0][minorTime][0], sizeof(bSamples[0][0][0]) * 8 / NR_BITS);

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
      storeVisibilities(visibilities, channel, firstStationY, firstStationX, statYoffset, statXoffset, y, x, skipStoreYcheck, skipStoreXcheck, sum[y][x], scratchSpace, tid / warpSize);
}


extern "C" {

__global__
__launch_bounds__(NR_WARPS * 32)
void correlate(Visibilities visibilities, const Samples samples)
{
  
  const unsigned nrFragmentsY = NR_STATIONS_PER_BLOCK / NR_STATIONS_PER_TCM_Y / 2;

  unsigned block = blockIdx.x;
  if(block == 0 && threadIdx.x == 0){
    int iSamples = NR_CHANNELS * NR_SAMPLES_PER_CHANNEL / NR_TIMES_PER_BLOCK * NR_STATIONS * NR_POLARIZATIONS * NR_TIMES_PER_BLOCK * sizeof(char2);
    int iVisibilities = NR_CHANNELS * NR_BASELINES * NR_POLARIZATIONS * NR_POLARIZATIONS*sizeof(int2);
    printf("%d %d\n",iSamples,iVisibilities);
  }

#if NR_STATIONS_PER_BLOCK == 48
  unsigned blockY = (unsigned) (sqrtf(8 * block + 1) - .99999f) / 2;
  unsigned blockX = block - blockY * (blockY + 1) / 2;
  unsigned firstStationX = blockX * NR_STATIONS_PER_BLOCK;
#elif NR_STATIONS_PER_BLOCK == 64
  unsigned blockY = (unsigned) sqrtf(block);
  unsigned blockX = block - blockY * blockY;
  unsigned firstStationX = blockX * (NR_STATIONS_PER_BLOCK / 2);
#endif
  unsigned firstStationY = blockY * NR_STATIONS_PER_BLOCK;

  __shared__ union {
    struct {
      SharedData<>::Asamples aSamples;
      SharedData<NR_STATIONS_PER_BLOCK == 64 ? 32 : NR_STATIONS_PER_BLOCK>::Bsamples bSamples;
    } rectangle;
    struct {
      SharedData<>::Bsamples samples;
    } triangle;
    ScratchSpace scratchSpace[NR_WARPS];
  } u;

  if (firstStationX == firstStationY)
#if NR_STATIONS_PER_BLOCK == 48
    doCorrelateRectangle<nrFragmentsY, NR_STATIONS % NR_STATIONS_PER_BLOCK == 0, NR_STATIONS % NR_STATIONS_PER_BLOCK == 0, NR_STATIONS % NR_STATIONS_PER_BLOCK == 0, false>(visibilities, samples, firstStationY, firstStationX, u.rectangle.aSamples, u.rectangle.bSamples, u.scratchSpace);
#elif NR_STATIONS_PER_BLOCK == 64
    if (NR_STATIONS % NR_STATIONS_PER_BLOCK != 0 && (NR_STATIONS < NR_STATIONS_PER_BLOCK || firstStationX >= NR_STATIONS / NR_STATIONS_PER_BLOCK * NR_STATIONS_PER_BLOCK))
      doCorrelateTriangle<false>(visibilities, samples, firstStationX, 2 * threadIdx.z + threadIdx.y, 64 * threadIdx.z + 32 * threadIdx.y + threadIdx.x, u.triangle.samples, u.scratchSpace);
    else
      doCorrelateTriangle<true>(visibilities, samples, firstStationX, 2 * threadIdx.z + threadIdx.y, 64 * threadIdx.z + 32 * threadIdx.y + threadIdx.x, u.triangle.samples, u.scratchSpace);
#endif
#if NR_STATIONS % NR_STATIONS_PER_BLOCK != 0
  else if (NR_STATIONS < NR_STATIONS_PER_BLOCK || firstStationY >= NR_STATIONS / NR_STATIONS_PER_BLOCK * NR_STATIONS_PER_BLOCK)
    doCorrelateRectangle<(NR_STATIONS % NR_STATIONS_PER_BLOCK + 2 * NR_STATIONS_PER_TCM_Y - 1) / NR_STATIONS_PER_TCM_Y / 2, false, true, NR_STATIONS % (2 * NR_STATIONS_PER_TCM_Y) == 0, true>(visibilities, samples, firstStationY, firstStationX, u.rectangle.aSamples, u.rectangle.bSamples, u.scratchSpace);
#endif
  else
    doCorrelateRectangle<nrFragmentsY, true, true, true, true>(visibilities, samples, firstStationY, firstStationX, u.rectangle.aSamples, u.rectangle.bSamples, u.scratchSpace);
}

//Normally this open brace should be uncommented, but because we are exploting that pycuda surrounds the source with `extern "C"{}`, this last curly bracket introduced by pycuda means we dont actually need to add this bracket
//}