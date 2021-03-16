/*
    This kernel aims to carry out the reorder functionality required by katxgpu.
    This GPU-side reorder transforms a 1D block of data in the following matrix format:
    - uint16_t [n_antennas] [n_channels] [n_samples_per_channel] [polarizations]
      transposed to
      uint16_t [n_channels] [n_samples_per_channel//times_per_block]
               [n_antennas] [polarizations] [times_per_block]
    - Typical values for the dimensions
        - n_antennas (a) = 64
        - n_channels (c) = 128
        - n_samples_per_channel (t) = 256
        - polarisations (p) = 2, always
        - times_per_block = 16, always
*/

// Includes
#include <stdint.h>
#include <stdlib.h>
#include <sys/cdefs.h>

<%include file="/port.mako"/>
// Defines, now using mako parametrisation
#define NR_STATIONS ${n_ants}
#define NR_CHANNELS ${n_channels}
#define NR_SAMPLES_PER_CHANNEL ${n_samples_per_channel}
#define NR_POLARISATIONS ${n_polarisations}
#define NR_TIMES_PER_BLOCK ${n_times_per_block}
#define NR_BATCHES ${n_batches}

// Defines
// - Altered for ease of visualisation
/*
#define NR_STATIONS 64
#define NR_CHANNELS 128
#define NR_SAMPLES_PER_CHANNEL 256
#define NR_POLARISATIONS 2
#define NR_TIMES_PER_BLOCK 16
#define NR_BATCHES 10
*/

// Maximum number of threads per block, as per:
// - https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications__technical-specifications-per-compute-capability
#define THREADS_PER_BLOCK 1024

/*  \brief Kernel that implements a naive reorder of F-Engine data.

    The following CUDA kernel implements a naive (i.e. unrefined) reorder of data ingested by the GPU X-Engine from the F-Engine.
    As mentioned at the top of this document, data is received as an array in the format of:
    - uint16_t [n_antennas] [n_channels] [n_samples_per_channel] [polarisations]
    And is required to be reordered into an array of format:
    - uint16_t [n_channels] [n_samples_per_channel // times_per_block] [n_antennas] [polarizations] [times_per_block]

    Currently, all dimension-strides are calculated within the kernel itself.
    - Granted, there are some redudancies/inefficiences in variable usage; however,
    - The kernel itself is operating as required, and will be refined as necessary.
    
    \param[in]  pu16Array           Pointer to a pre-populated input data array. The input array is one-dimensional but stores
                                    multidimensional data according to the following indices:
                                    - [n_antennas] [n_channels] [n_samples_per_channel] [polarisations]
    \param[out] pu16ArrayReordered  Pointer to the memory allocated for the reordered output data. Once more, this 1D output array
                                    represents multidimensional data in the following format:
                                    - [n_channels] [n_samples_per_channel // times_per_block] [n_antennas] [polarisations] [times_per_block]

*/

__global__
void reorder_naive(uint16_t *pu16Array, uint16_t *pu16ArrayReordered)
{
    // 1. Declare indices used for reorder
    int iThreadIndex_x = blockIdx.x * blockDim.x + threadIdx.x;
    int iRemIndex, iBatchCounter;
    iBatchCounter = blockIdx.y;

    int iAntIndex, iChanIndex, iTimeIndex, iPolIndex;
    // iPolStride = 1;

    // - Declaring in their order of dimensionality for the new matrix
    int iNewIndex, iNewChanOffset, iTimeOuterOffset, iNewAntOffset, iNewPolOffset;
    int iTimeOuterIndex, iTimeInnerIndex, iMatrixStride_y;
    
    // 2. Calculate indices for reorder
    // 2.1. Calculate 'current'/original indices for each dimension
    //      - Matrix Stride should be the same value for Original and Reordered matrices
    iMatrixStride_y = iBatchCounter * NR_STATIONS * NR_CHANNELS * NR_SAMPLES_PER_CHANNEL * NR_POLARISATIONS;
    iAntIndex = iThreadIndex_x / (NR_CHANNELS * NR_SAMPLES_PER_CHANNEL * NR_POLARISATIONS);
    iRemIndex = iThreadIndex_x % (NR_CHANNELS * NR_SAMPLES_PER_CHANNEL * NR_POLARISATIONS);

    iChanIndex = iRemIndex / (NR_SAMPLES_PER_CHANNEL * NR_POLARISATIONS);
    iRemIndex = iRemIndex % (NR_SAMPLES_PER_CHANNEL * NR_POLARISATIONS);

    iTimeIndex = iRemIndex / NR_POLARISATIONS;
    iRemIndex = iRemIndex % NR_POLARISATIONS;
    // 0 = Even = Pol-0, 1 = Odd = Pol-1
    iPolIndex = iRemIndex;

    // 2.2. Calculate reordered matrix's indices and stride accordingly
    iNewChanOffset = iChanIndex * (NR_SAMPLES_PER_CHANNEL/NR_TIMES_PER_BLOCK)*NR_STATIONS*NR_POLARISATIONS*NR_TIMES_PER_BLOCK;
    iTimeOuterIndex = iTimeIndex / NR_TIMES_PER_BLOCK;
    iTimeOuterOffset = iTimeOuterIndex * NR_STATIONS*NR_POLARISATIONS*NR_TIMES_PER_BLOCK;
    iNewAntOffset = iAntIndex * NR_POLARISATIONS*NR_TIMES_PER_BLOCK;
    iNewPolOffset = iPolIndex * NR_TIMES_PER_BLOCK;
    iTimeInnerIndex = iTimeIndex % NR_TIMES_PER_BLOCK; // ?
    
    iNewIndex = iNewChanOffset + iTimeOuterOffset + iNewAntOffset + iNewPolOffset + iTimeInnerIndex;

    // 3. Perform the reorder (where necessary)
    uint16_t u16InputSample;
    if (iThreadIndex_x < (NR_STATIONS * NR_CHANNELS * NR_SAMPLES_PER_CHANNEL * NR_POLARISATIONS))
    {
        // 3.1. Read out from the original array
        u16InputSample = *(pu16Array + iThreadIndex_x + iMatrixStride_y);
        // 3.2. Store at its reordered index
        *(pu16ArrayReordered + iNewIndex + iMatrixStride_y) = u16InputSample;
    }
}
