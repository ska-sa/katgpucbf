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
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <sys/cdefs.h>

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
#define NR_STATIONS ${n_ants}
#define NR_CHANNELS ${n_channels}
#define NR_SAMPLES_PER_CHANNEL ${n_samples_per_channel}
#define NR_POLARIZATIONS ${n_polarizastions}
#define NR_TIMES_PER_BLOCK $(n_times_per_block)

// Defines
// - Altered for ease of visualisation
/*
#define ANTS 64
#define CHANS 128
#define TIME_SAMPLES_PER_CHAN 256
#define POLS 2
#define TIMES_PER_BLOCK 16
#define BATCH 10
*/

#define INPUT_RATE 28    // Gbps
#define ONE_MB 1000000.0 // For ease of calc

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
    iMatrixStride_y = iBatchCounter * ANTS * CHANS * TIME_SAMPLES_PER_CHAN * POLS;
    iAntIndex = iThreadIndex_x / (CHANS * TIME_SAMPLES_PER_CHAN * POLS);
    iRemIndex = iThreadIndex_x % (CHANS * TIME_SAMPLES_PER_CHAN * POLS);

    iChanIndex = iRemIndex / (TIME_SAMPLES_PER_CHAN * POLS);
    iRemIndex = iRemIndex % (TIME_SAMPLES_PER_CHAN * POLS);

    iTimeIndex = iRemIndex / POLS;
    iRemIndex = iRemIndex % POLS;
    // 0 = Even = Pol-0, 1 = Odd = Pol-1
    iPolIndex = iRemIndex;

    // 2.2. Calculate reordered matrix's indices and stride accordingly
    iNewChanOffset = iChanIndex * (TIME_SAMPLES_PER_CHAN/TIMES_PER_BLOCK)*ANTS*POLS*TIMES_PER_BLOCK;
    iTimeOuterIndex = iTimeIndex / TIMES_PER_BLOCK;
    iTimeOuterOffset = iTimeOuterIndex * ANTS*POLS*TIMES_PER_BLOCK;
    iNewAntOffset = iAntIndex * POLS*TIMES_PER_BLOCK;
    iNewPolOffset = iPolIndex * TIMES_PER_BLOCK;
    iTimeInnerIndex = iTimeIndex % TIMES_PER_BLOCK; // ?
    
    iNewIndex = iNewChanOffset + iTimeOuterOffset + iNewAntOffset + iNewPolOffset + iTimeInnerIndex;

    // 3. Perform the reorder (where necessary)
    uint16_t u16InputSample;
    if (iThreadIndex_x < (ANTS * CHANS * TIME_SAMPLES_PER_CHAN * POLS))
    {
        // 3.1. Read out from the original array
        u16InputSample = *(pu16Array + iThreadIndex_x + iMatrixStride_y);
        // 3.2. Store at its reordered index
        *(pu16ArrayReordered + iNewIndex + iMatrixStride_y) = u16InputSample;
    }
}

// Function declarations
void display_1d_strided(uint16_t *pu16Array, bool bDisplayHex);
void display_1d_reordered(uint16_t *pu16Array, bool bDisplayHex);
int verify_reorder(uint16_t *pu16Array, uint16_t *pu16ArrayReordered, int iMatrixSize);

int main()
{
    printf("Starting...\n\n");

    #pragma region Variable Declaration

    // Working outside the default stream (stream 'ID' == 0)
    cudaStream_t stream1;
    cudaStreamCreate(&stream1);

    // For timing the kernel execution
    cudaEvent_t eventKernelStart, eventKernelStop;         // To time the kernel
    cudaEvent_t eventMemcpyH2DStart, eventMemcpyH2DStop;   // To time the H2D Memcpy
    cudaEvent_t eventMemcpyD2HStart, eventMemcpyD2HStop;   // To time the D2H Memcpy
    cudaEventCreate(&eventKernelStart);
    cudaEventCreate(&eventKernelStop);
    cudaEventCreate(&eventMemcpyH2DStart);
    cudaEventCreate(&eventMemcpyH2DStop);
    cudaEventCreate(&eventMemcpyD2HStart);
    cudaEventCreate(&eventMemcpyD2HStop);

    int iMatrixSize = ANTS * CHANS * TIME_SAMPLES_PER_CHAN * POLS;
    int iTotalElements = ANTS * CHANS * TIME_SAMPLES_PER_CHAN * POLS * BATCH;
    // int iTotalElementsOut = CHANS * (TIME_SAMPLES_PER_CHAN / TIMES_PER_BLOCK) * ANTS * POLS * TIMES_PER_BLOCK;
    // - iTotalElementsOut = iTotalElements! Just with different dimensions

    uint16_t *pu16Array_host, *pu16ArrayReordered_host, *pu16Array_device, *pu16ArrayReordered_device;

    #pragma endregion

    #pragma region Memory Setup and Configuration

    // Pinned host memory required
    cudaMallocHost(&pu16Array_host, iTotalElements*sizeof(uint16_t));
    cudaMallocHost(&pu16ArrayReordered_host, iTotalElements*sizeof(uint16_t));

    cudaMalloc(&pu16Array_device, iTotalElements*sizeof(uint16_t));
    cudaMalloc(&pu16ArrayReordered_device, iTotalElements*sizeof(uint16_t));

    // Fill the host memory array
    int i = 0;
    uint16_t u16ArrayValue = 0;
    for (i = 0; i < iTotalElements; i += 2)
    {
        // Increment by two to assign each polarisation the same value
        pu16Array_host[i] = u16ArrayValue;
        pu16Array_host[i+1] = u16ArrayValue++;
    }
    memset(pu16ArrayReordered_host, 0, iTotalElements*sizeof(uint16_t));

    // Copy host memory into device memory buffer - asynchronously - and time it!
    cudaEventRecord(eventMemcpyH2DStart, stream1);
    cudaMemcpyAsync(pu16Array_device, pu16Array_host, iTotalElements*sizeof(uint16_t),\
                    cudaMemcpyHostToDevice, stream1);
    cudaEventRecord(eventMemcpyH2DStop, stream1);

    // Seeing as our output host memory is empty/to be populated by the kernel,
    // We can simply cudaMemset the output device memory to zero.
    cudaMemsetAsync(pu16ArrayReordered_device, 0, iTotalElements*sizeof(uint16_t), stream1);

    #pragma endregion

    #pragma region Kernel Execution and Timing

    // For ease of visualisation the [polarisation] data fields as a pair of uint8_t's
    // bool bDisplayHex = false;
    // display_1d_strided(pu16Array_host, bDisplayHex);

    printf("\n=================================================\n");
    printf("\nReordering...\n");
    printf("\n=================================================\n");
    
    // Launch kernel on iTotalElements elements
    // - Need to calculate the number of blocks required at 1024 threads-per-block
    int iNumBlocks_x = ((iMatrixSize + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    int iNumBlocks_y = BATCH;
    dim3 gridSize(iNumBlocks_x, iNumBlocks_y);

    cudaEventRecord(eventKernelStart, stream1);
    // Now executing on a non-default cudaStream
    reorder_naive<<<gridSize, THREADS_PER_BLOCK, 0, stream1>>>(pu16Array_device, pu16ArrayReordered_device);
    cudaEventRecord(eventKernelStop, stream1);

    // Copy device memory back to host, and time it!
    cudaEventRecord(eventMemcpyD2HStart, stream1);
    cudaMemcpyAsync(pu16ArrayReordered_host, pu16ArrayReordered_device, iTotalElements*sizeof(uint16_t),\
                    cudaMemcpyDeviceToHost, stream1);
    cudaEventRecord(eventMemcpyD2HStop, stream1);

    // Make sure we're all on the same page
    cudaStreamSynchronize(stream1);
    
    // display_1d_reordered(pu16ArrayReordered_host, bDisplayHex);

    #pragma endregion

    printf("\n=================================================\n");
    printf("\nVerifying...\n");
    int iResult = verify_reorder(pu16Array_host, pu16ArrayReordered_host, iMatrixSize);

    if (iResult)
        printf("\nReorder was successful!\n");
    else
        printf("\nReorder failed.\n");
    
    printf("\n=================================================\n");
    #pragma region Calculate GPU Utilisation
    cudaEventSynchronize(eventMemcpyH2DStop);
    cudaEventSynchronize(eventKernelStop);
    cudaEventSynchronize(eventMemcpyD2HStop);
    
    float fMemcpyH2D_ms = 0;
    float fKernelTime_ms = 0;
    float fMemcpyD2H_ms = 0;
    
    cudaEventElapsedTime(&fMemcpyH2D_ms, eventMemcpyH2DStart, eventMemcpyH2DStop);
    cudaEventElapsedTime(&fKernelTime_ms, eventKernelStart, eventKernelStop);
    cudaEventElapsedTime(&fMemcpyD2H_ms, eventMemcpyD2HStart, eventMemcpyD2HStop);

    float fGpuUtilRatio;
    float fChunkSize_MB = sizeof(uint16_t) * iTotalElements / ONE_MB;
    float fTransferTime_s = (fChunkSize_MB / 1000) / (INPUT_RATE / 8.0);
    
    fGpuUtilRatio = (fKernelTime_ms / 1000) / fTransferTime_s;

    #pragma endregion

    // Free at last
    cudaFree(pu16Array_device);
    cudaFree(pu16ArrayReordered_device);
    // Now have to free pinned host memory using cudaFreeHost
    cudaFreeHost(pu16Array_host);
    cudaFreeHost(pu16ArrayReordered_host);

    printf("\nWorking with a Chunk size of ~%.0f MB\n", fChunkSize_MB);
    printf("\nHost-to-Device Memcpy executed in %.3f milliseconds.\n", fMemcpyH2D_ms);
    printf("\nKernel executed in %.3f milliseconds.\n", fKernelTime_ms);
    printf("\nDevice-to-Host Memcpy executed in %.3f milliseconds.\n\n", fMemcpyD2H_ms);
    printf("\n=================================================\n");
    printf("\nGPU Utilisation Ratio (%%) \n\t= [Kernel Exection time (s)]/[Data Transfer time (s)]\n\
            = %.2f %%\n\n", (fGpuUtilRatio * 100));

    return 0;
}

#pragma region Display Arrays

void display_1d_strided(uint16_t *pu16Array, bool bDisplayHex)
{
    int iAntIndex, iChanIndex, iTimeIndex, iPolIndex, iBatchCounter;
    // Treating each polarisation as a single uint16_t

    int iAntOffset, iChanOffset, iTimeOffset, iMatrixOffset, iOffset;
    
    for (iBatchCounter = 0; iBatchCounter < BATCH; iBatchCounter++)
    {
        printf("\n\tBatch %d\n\n", iBatchCounter);
        iMatrixOffset = iBatchCounter * ANTS * CHANS * TIME_SAMPLES_PER_CHAN * POLS;
        for (iAntIndex = 0; iAntIndex < ANTS; iAntIndex++)
        {
            iAntOffset = iAntIndex * CHANS * TIME_SAMPLES_PER_CHAN * POLS;
            for (iChanIndex = 0; iChanIndex < CHANS; iChanIndex++)
            {
                iChanOffset = iChanIndex * TIME_SAMPLES_PER_CHAN * POLS;
                for (iTimeIndex = 0; iTimeIndex < TIME_SAMPLES_PER_CHAN; iTimeIndex++)
                {
                    iTimeOffset = iTimeIndex * POLS;
                    for (iPolIndex = 0; iPolIndex < POLS; iPolIndex++)
                    {
                        iOffset = iAntOffset + iChanOffset + iTimeOffset + iPolIndex;
                        if (bDisplayHex)
                        {
                            printf("0x%X ", *(pu16Array + iOffset + iMatrixOffset));
                        }
                        else
                        {
                            printf("%d ", *(pu16Array + iOffset + iMatrixOffset));
                        }
                    }
                }
                printf("\n");
            }
            printf("\n----------------------------------------------------\n");
        }
        printf("\n========================================================\n");
    }
}

void display_1d_reordered(uint16_t *pu16Array, bool bDisplayHex)
{
    int iChanIndex, iTimeOuterIndex, iAntIndex, iPolIndex, iTimeInnerIndex, iBatchCounter;
    
    int iChanOffset, iTimeOffset, iAntOffset, iPolOffset, iMatrixOffset, iOffset;

    for (iBatchCounter = 0; iBatchCounter < BATCH; iBatchCounter++)
    {
        printf("\n\tBatch %d\n\n", iBatchCounter);
        iMatrixOffset = iBatchCounter * ANTS * CHANS * TIME_SAMPLES_PER_CHAN * POLS;
        for (iChanIndex = 0; iChanIndex < CHANS; iChanIndex++)
        {
            iChanOffset = iChanIndex * (TIME_SAMPLES_PER_CHAN / TIMES_PER_BLOCK) * ANTS * POLS * TIMES_PER_BLOCK;
            for (iTimeOuterIndex = 0; iTimeOuterIndex < (TIME_SAMPLES_PER_CHAN / TIMES_PER_BLOCK); iTimeOuterIndex++)
            {
                iTimeOffset = iTimeOuterIndex * ANTS * POLS * TIMES_PER_BLOCK;
                for (iAntIndex = 0; iAntIndex < ANTS; iAntIndex++)
                {
                    iAntOffset = iAntIndex * POLS * TIMES_PER_BLOCK;
                    for (iPolIndex = 0; iPolIndex < POLS; iPolIndex++)
                    {
                        iPolOffset = iPolIndex * TIMES_PER_BLOCK;
                        for (iTimeInnerIndex = 0; iTimeInnerIndex < TIMES_PER_BLOCK; iTimeInnerIndex++)
                        {
                            iOffset = iChanOffset + iTimeOffset + iAntOffset + iPolOffset + iTimeInnerIndex;
                            if (bDisplayHex)
                            {
                                printf("0x%X ", *(pu16Array + iOffset + iMatrixOffset));
                            }
                            else
                            {
                                printf("%d ", *(pu16Array + iOffset + iMatrixOffset));
                            }
                        }
                        printf("\n\t");    
                    }
                    printf("\n\n");
                }
                printf("\n-------------------------------------------------\n");
            }
            printf("\n===================================================\n");
        }
        printf("\n*********************************************************************\n");
    }
}

#pragma endregion

int verify_reorder(uint16_t *pu16Array, uint16_t *pu16ArrayReordered, int iMatrixSize)
{
    int iCurrIndex, iRemIndex, iBatchCounter;
    int iAntIndex, iChanIndex, iTimeIndex, iPolIndex;
    int iAntStride, iChanStride, iTimeStride, iMatrixStride;
    iAntStride = CHANS * TIME_SAMPLES_PER_CHAN * POLS;
    iChanStride = TIME_SAMPLES_PER_CHAN * POLS;
    iTimeStride = POLS;
    // iPolStride = 1;

    // Declaring in their order of dimensionality for the new matrix
    int iNewIndex, iNewChanOffset, iTimeOuterOffset, iNewAntOffset, iNewPolOffset;
    int iTimeIndexOuter, iTimeIndexInner;
    int iNewAntStride, iNewChanStride, iNewSampleStride, iNewPolStride;
    iNewChanStride = (TIME_SAMPLES_PER_CHAN / TIMES_PER_BLOCK) * ANTS * POLS * TIMES_PER_BLOCK;
    iNewSampleStride = ANTS * POLS * TIMES_PER_BLOCK;
    iNewAntStride = POLS * TIMES_PER_BLOCK;
    iNewPolStride = TIMES_PER_BLOCK;

    for (iBatchCounter = 0; iBatchCounter < BATCH; iBatchCounter++)
    {
        iMatrixStride = iBatchCounter * ANTS * CHANS * TIME_SAMPLES_PER_CHAN * POLS;
        for (iCurrIndex = 0; iCurrIndex < iMatrixSize; iCurrIndex++)
        {
            iAntIndex = iCurrIndex / iAntStride;
            iRemIndex = iCurrIndex % iAntStride;

            iChanIndex = iRemIndex / iChanStride;
            iRemIndex = iRemIndex % iChanStride;

            iTimeIndex = iRemIndex / iTimeStride;
            iRemIndex = iRemIndex % iTimeStride;
            // 0 = Even = Pol-0, 1 = Odd = Pol-1
            iPolIndex = iRemIndex;

            // Now, stride them according to the new matrix's dimensions
            iNewChanOffset = iChanIndex * iNewChanStride;
            iTimeIndexOuter = iTimeIndex / TIMES_PER_BLOCK;
            iTimeOuterOffset = iTimeIndexOuter * iNewSampleStride;
            iNewAntOffset = iAntIndex * iNewAntStride;
            iNewPolOffset = iPolIndex * iNewPolStride;
            iTimeIndexInner = iTimeIndex % TIMES_PER_BLOCK; // ?
            
            iNewIndex = iNewChanOffset + iTimeOuterOffset + iNewAntOffset + \
                        iNewPolOffset + iTimeIndexInner + iMatrixStride;

            if (pu16ArrayReordered[iNewIndex] != pu16Array[iCurrIndex + iMatrixStride])
            {
                // Problem
                printf("\nReordered: %d at index %d\n != Original: %d at index %d",\
                        pu16ArrayReordered[iNewIndex], iNewIndex,\
                        pu16Array[iCurrIndex + iMatrixStride], (iCurrIndex + iMatrixStride));
                return 0;
            }
        }
    }
    
    // If we get here, all is well
    return 1;
}
