/* C library containing methods for kernel data verification.
 * 
 * This was created as it speeds up the verification process massively, when compared to its python-equivalent.
 *
 * The following verificaiton functions have been migrated to C to take advantage of this preformance increase.
 * 1. verify_reorder:
 *  - For verification of the data output by the precorrelation_reorder_kernel.
 */

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

/*  \brief Function that implements verification of data output by the pre-correlation reorder kernel.
 *
 *  As mentioned in the corresponding kernel, data is received by the kernel as an array in the format of:
 *   - uint16_t [n_antennas] [n_channels] [n_samples_per_channel] [polarisations]
 *   And is reordered into an array of format:
 *   - uint16_t [n_channels] [n_samples_per_channel // times_per_block] [n_antennas] [polarizations] [times_per_block]
 *
 *   This function accept the input and output data as well as all matrix dimensions as individual parameters.
 *   Typical values for matrix dimensions are as follows:
 *   - Antennas = 64,
 *   - Channels = 128,
 *   - Samples-per-channel = 256,
 *   - Polarisations = 2 (hardcoded),
 *   - Times-per-block = 16.
 *   
 *   \param[in] pi8Array            Pointer to a pre-populated input data array. The input array is one-dimensional but stores
 *                                   multidimensional data according to the following indices:
 *                                   - [n_antennas] [n_channels] [n_samples_per_channel] [polarisations]
 *   \param[in] pi8ArrayReordered   Pointer to array holding reordered data. Once more, this 1D output array
 *                                   represents multidimensional data in the following format:
 *                                   - [n_channels] [n_samples_per_channel // times_per_block] [n_antennas] [polarisations] [times_per_block]
 *   \parma[in] iNumBatches         Number of batches of data that has been reordered.
 *   \parma[in] iNumAnts            Number of antennas.
 *   \parma[in] iNumChans           Number of channels, per antenna.
 *   \parma[in] iNumSamplesPerChan  Number of samples per channel.
 *   \parma[in] iNumPols            Number of polarisations per sample.
 *   \param[in] iNumTimesPerBlock   Number of times per block.
 */

int verify_precorrelation_reorder(
    int8_t *pi8Array,
    int8_t *pi8ArrayReordered,
    int iNumBatches,
    int iNumAnts,
    int iNumChans,
    int iNumSamplesPerChan,
    int iNumPols,
    int iNumTimesPerBlock)
{
    // 1. Input matrix
    // 1.1. Declare variables for input matrix strides and indices
    int iCurrIndex, iRemIndex, iBatchCounter, iMatrixSize;
    int iAntIndex, iChanIndex, iTimeIndex, iPolIndex;
    int iAntStride, iChanStride, iTimeStride, iMatrixStride;

    // 1.2. Calculate input matrix's strides
    iMatrixSize = iNumAnts * iNumChans * iNumSamplesPerChan * iNumPols;
    iAntStride = iNumChans * iNumSamplesPerChan * iNumPols;
    iChanStride = iNumSamplesPerChan * iNumPols;
    iTimeStride = iNumPols;
    // iPolStride = 1;

    // 2. Output, reordered matrix
    // 2.1. Declaring in their order of dimensionality for the new matrix
    int iNewIndex, iNewChanOffset, iTimeOuterOffset, iNewAntOffset, iNewPolOffset;
    int iTimeIndexOuter, iTimeIndexInner;
    int iNewAntStride, iNewChanStride, iNewSampleStride, iNewPolStride;
    
    // 2.2. Calculate output matrix's strides
    iNewChanStride = (iNumSamplesPerChan / iNumTimesPerBlock) * iNumAnts * iNumPols * iNumTimesPerBlock;
    iNewSampleStride = iNumAnts * iNumPols * iNumTimesPerBlock;
    iNewAntStride = iNumPols * iNumTimesPerBlock;
    iNewPolStride = iNumTimesPerBlock;

    // 3. Scroll through the entire data set for comparison
    //  - Ultimately it is a batch of matrices, hence separate for-loops
    for (iBatchCounter = 0; iBatchCounter < iNumBatches; iBatchCounter++)
    {
        iMatrixStride = iBatchCounter * iMatrixSize;
        for (iCurrIndex = 0; iCurrIndex < iMatrixSize; iCurrIndex++)
        {
            // 3. Index calculations
            // 3.1. Calculate specific input indices per dimension
            iAntIndex = iCurrIndex / iAntStride;
            iRemIndex = iCurrIndex % iAntStride;

            iChanIndex = iRemIndex / iChanStride;
            iRemIndex = iRemIndex % iChanStride;

            iTimeIndex = iRemIndex / iTimeStride;
            iRemIndex = iRemIndex % iTimeStride;
            // 0 = Even = Pol-0, 1 = Odd = Pol-1
            iPolIndex = iRemIndex;

            // 3.2. Stride them according to the new matrix's dimensions
            iNewChanOffset = iChanIndex * iNewChanStride;
            iTimeIndexOuter = iTimeIndex / iNumTimesPerBlock;
            iTimeOuterOffset = iTimeIndexOuter * iNewSampleStride;
            iNewAntOffset = iAntIndex * iNewAntStride;
            iNewPolOffset = iPolIndex * iNewPolStride;
            iTimeIndexInner = iTimeIndex % iNumTimesPerBlock;

            iNewIndex = iNewChanOffset + iTimeOuterOffset + iNewAntOffset + \
                        iNewPolOffset + iTimeIndexInner + iMatrixStride;
            
            // 4. Compare the input data to the output, reordered data
            if (pi8ArrayReordered[iNewIndex] != pi8Array[iCurrIndex + iMatrixStride])
            {
                // Problem
                printf("\nReordered: %d at index %d\n != Original: %d at index %d\n",\
                        pi8ArrayReordered[iNewIndex], iNewIndex,\
                        pi8Array[iCurrIndex + iMatrixStride], (iCurrIndex + iMatrixStride));
                
                return 0;
            }
        }
    }
    
    // If we get here, all is well
    return 1;
}
