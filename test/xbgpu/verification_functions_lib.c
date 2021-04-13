/* C library containing methods for kernel data verification.
 *
 * This was created as it speeds up the verification process massively, when compared to its python-equivalent.
 *
 * The following verificaiton functions have been migrated to C to take advantage of this preformance increase.
 * 1. verify_reorder:
 *  - For verification of the data output by the precorrelation_reorder_kernel.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

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
 *   \param[in] pi8Array            Pointer to a pre-populated input data array. The input array is one-dimensional but
 *                                  stores multidimensional data according to the following indices:
 *                                   - [n_antennas] [n_channels] [n_samples_per_channel] [polarisations]
 *   \param[in] pi8ArrayReordered   Pointer to array holding reordered data. Once more, this 1D output array
 *                                  represents multidimensional data in the following format:
 *                                   - [n_channels] [n_samples_per_channel // times_per_block] [n_antennas]
 *                                     [polarisations] [times_per_block] 
 *   \parma[in] iNumBatches         Number of batches of data that has been reordered.
 *   \parma[in] iNumAnts            Number of antennas.
 *   \parma[in] iNumChans           Number of channels, per antenna.
 *   \parma[in] iNumSamplesPerChan  Number of samples per channel.
 *   \parma[in] iNumPols            Number of polarisations per sample.
 *   \param[in] iNumTimesPerBlock   Number of times per block.
 */

int verify_precorrelation_reorder(int8_t *pi8Array, int8_t *pi8ArrayReordered, int iNumBatches, int iNumAnts,
                                  int iNumChans, int iNumSamplesPerChan, int iNumPols, int iNumTimesPerBlock)
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

            iNewIndex =
                iNewChanOffset + iTimeOuterOffset + iNewAntOffset + iNewPolOffset + iTimeIndexInner + iMatrixStride;

            // 4. Compare the input data to the output, reordered data
            if (pi8ArrayReordered[iNewIndex] != pi8Array[iCurrIndex + iMatrixStride])
            {
                // Problem
                printf("\nReordered: %d at index %d\n != Original: %d at index %d\n", pi8ArrayReordered[iNewIndex],
                       iNewIndex, pi8Array[iCurrIndex + iMatrixStride], (iCurrIndex + iMatrixStride));

                return 0;
            }
        }
    }

    // If we get here, all is well
    return 1;
}

struct si32Complex
{
    int32_t real;
    int32_t imag;
};

struct si32Complex createBoundedComplex(int8_t i8Real, int8_t i8Imag)
{
    if (i8Real == -128)
        i8Real = -127;
    if (i8Imag == -128)
        i8Imag = -127;
    struct si32Complex ret = {i8Real, i8Imag};
    return ret;
}

int get_baseline_index(int iAnt1, int iAnt2)
{
    if (iAnt2 > iAnt1)
        return -1;
    return iAnt1 * (iAnt1 + 1) / 2 + iAnt2;
}

// c = a * b * alpha
void complex_multiply_scale_accumulate(struct si32Complex *c, struct si32Complex a, struct si32Complex b, int alpha)
{
    c->real += (a.real * b.real + a.imag * b.imag) * alpha;
    c->imag += (a.imag * b.real - a.real * b.imag) * alpha;
}

int assert_complex_samples(struct si32Complex calculatedSample, uint64_t u64ActualSample)
{
    int32_t i32ActualSampleReal = (int32_t)u64ActualSample;
    int32_t i32ActualSampleImag = (int32_t)(u64ActualSample >> 32);

    if (calculatedSample.real != i32ActualSampleReal)
        return 0;
    if (calculatedSample.imag != i32ActualSampleImag)
        return 0;
    return 1;
}

int verify_antpair_visibilities(int iBatchStartIndex, int iNumBatches, int iNumSamplesPerChan, int iAnt1Index,
                                int iAnt2Index, uint64_t u64Pol00, uint64_t u64Pol01, uint64_t u64Pol10,
                                uint64_t u64Pol11)
{
    struct si32Complex sGeneratedPol00 = {0, 0};
    struct si32Complex sGeneratedPol01 = {0, 0};
    struct si32Complex sGeneratedPol10 = {0, 0};
    struct si32Complex sGeneratedPol11 = {0, 0};

    for (size_t ulBatchIndex = 0; ulBatchIndex < iNumBatches; ulBatchIndex++)
    {
        struct si32Complex ant1Pol0 = {(int8_t)iAnt1Index, (int8_t)iAnt1Index + 1};
        struct si32Complex ant1Pol1 = {(int8_t)iAnt1Index, (int8_t)iAnt1Index + 2};
        struct si32Complex ant2Pol0 = {(int8_t)iAnt2Index, (int8_t)iAnt2Index + 1};
        struct si32Complex ant2Pol1 = {(int8_t)iAnt2Index, (int8_t)iAnt2Index + 2};

        complex_multiply_scale_accumulate(&sGeneratedPol00, ant1Pol0, ant2Pol0, iNumSamplesPerChan);
        complex_multiply_scale_accumulate(&sGeneratedPol01, ant1Pol0, ant2Pol1, iNumSamplesPerChan);
        complex_multiply_scale_accumulate(&sGeneratedPol10, ant1Pol1, ant2Pol0, iNumSamplesPerChan);
        complex_multiply_scale_accumulate(&sGeneratedPol11, ant1Pol1, ant2Pol1, iNumSamplesPerChan);
    }

    //printf("%d %d %d %d\n", iAnt1Index, iAnt2Index, sGeneratedPol00.real, sGeneratedPol00.imag);

    if (assert_complex_samples(sGeneratedPol00, u64Pol00) == 0)
    {
        printf("a\n");
        return 0;
    }
    if (assert_complex_samples(sGeneratedPol01, u64Pol01) == 0)
    {
        printf("b\n");
        return 0;
    }
    if (assert_complex_samples(sGeneratedPol10, u64Pol10) == 0)
    {
        printf("c\n");
        return 0;
    }
    if (assert_complex_samples(sGeneratedPol11, u64Pol11) == 0)
    {
        printf("d\n");
        return 0;
    }
    // int complex z2 = 1 + 2 * I;

    return 1;
}

// Print somewhere a failure message
int verify_xbengine_proc_loop(uint64_t *pu64Baselines, int iBatchStartIndex, int iNumBatches, int iNumAnts,
                              int iNumChans, int iNumSamplesPerChan, int iNumPols)
{
    //printf("********************* %d **** %d **********************\n", iBatchStartIndex, iNumBatches);

    const int iNumBaselines = iNumAnts * (iNumAnts + 1) / 2; // Change to Ul
    const int iChannelStride = iNumBaselines * iNumPols;
    const int iBaselineStride = iNumPols * iNumPols;
    int iChannelIndex = 0;
    for (size_t iAnt1Index = 0; iAnt1Index < iNumAnts; iAnt1Index++)
    {
        for (size_t iAnt2Index = 0; iAnt2Index < iAnt1Index + 1; iAnt2Index++)
        {
            int iBaselineIndex = get_baseline_index(iAnt1Index, iAnt2Index);
            int iSampleIndex = iChannelIndex * iChannelStride + iBaselineStride * iBaselineIndex;
            int iSampleReal = (int32_t)pu64Baselines[iSampleIndex];
            int iSampleImag = (int32_t)(pu64Baselines[iSampleIndex] >> 32);
            // printf("%d %d %ld %ld %d %d\n", iBaselineIndex, iSampleIndex, iAnt1Index, iAnt2Index, iSampleReal,
            //        iSampleImag);
            int success = verify_antpair_visibilities(
                iBaselineIndex, iNumBatches, iNumSamplesPerChan, iAnt1Index, iAnt2Index, pu64Baselines[iSampleIndex],
                pu64Baselines[iSampleIndex + 1], pu64Baselines[iSampleIndex + 2], pu64Baselines[iSampleIndex + 3]);
            if (success == 0)
            {
                printf("We failed\n");
                return 0;
            }
        }
    }

    return 1;
}