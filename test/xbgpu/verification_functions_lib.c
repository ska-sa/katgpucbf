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
    int32_t i32Real;
    int32_t i32Imag;
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
    c->i32Real += (a.i32Real * b.i32Real + a.i32Imag * b.i32Imag) * alpha;
    c->i32Imag += (a.i32Imag * b.i32Real - a.i32Real * b.i32Imag) * alpha;
}

int assert_complex_samples(struct si32Complex sCalculatedSample, struct si32Complex sActualSample)
{
    if (sCalculatedSample.i32Real != sActualSample.i32Real)
        return 0;
    if (sCalculatedSample.i32Imag != sActualSample.i32Imag)
        return 0;
    return 1;
}

int verify_antpair_visibilities(uint uBatchStartIndex, uint ulNumBatches, uint ulChannelIndex, uint ulNumSamplesPerChan,
                                uint uAnt1Index, uint uAnt2Index, uint64_t u64Pol00, uint64_t u64Pol01,
                                uint64_t u64Pol10, uint64_t u64Pol11)
{
    struct si32Complex sActualPol00 = {(int32_t)u64Pol00, (int32_t)(u64Pol00 >> 32)};
    struct si32Complex sActualPol01 = {(int32_t)u64Pol01, (int32_t)(u64Pol01 >> 32)};
    struct si32Complex sActualPol10 = {(int32_t)u64Pol10, (int32_t)(u64Pol10 >> 32)};
    struct si32Complex sActualPol11 = {(int32_t)u64Pol11, (int32_t)(u64Pol11 >> 32)};

    struct si32Complex sGeneratedPol00 = {0, 0};
    struct si32Complex sGeneratedPol01 = {0, 0};
    struct si32Complex sGeneratedPol10 = {0, 0};
    struct si32Complex sGeneratedPol11 = {0, 0};

    for (size_t ulBatchIndex = uBatchStartIndex; ulBatchIndex < uBatchStartIndex + ulNumBatches; ulBatchIndex++)
    {
        int iSign = ulBatchIndex % 2 == 0 ? 1 : -1;

        struct si32Complex sAnt1Pol0 =
            createBoundedComplex((int8_t)(iSign * ulBatchIndex), (int8_t)(iSign * ulChannelIndex));
        struct si32Complex sAnt1Pol1 =
            createBoundedComplex((int8_t)(-iSign * uAnt1Index), (int8_t)(-iSign * ulChannelIndex));
        struct si32Complex sAnt2Pol0 =
            createBoundedComplex((int8_t)(iSign * ulBatchIndex), (int8_t)(iSign * ulChannelIndex));
        struct si32Complex sAnt2Pol1 =
            createBoundedComplex((int8_t)(-iSign * uAnt2Index), (int8_t)(-iSign * ulChannelIndex));

        complex_multiply_scale_accumulate(&sGeneratedPol00, sAnt1Pol0, sAnt2Pol0, ulNumSamplesPerChan);
        complex_multiply_scale_accumulate(&sGeneratedPol01, sAnt1Pol0, sAnt2Pol1, ulNumSamplesPerChan);
        complex_multiply_scale_accumulate(&sGeneratedPol10, sAnt1Pol1, sAnt2Pol0, ulNumSamplesPerChan);
        complex_multiply_scale_accumulate(&sGeneratedPol11, sAnt1Pol1, sAnt2Pol1, ulNumSamplesPerChan);
    }

    if (assert_complex_samples(sGeneratedPol00, sActualPol00) == 0)
    {
        printf("Ant 1 %d, Ant 2 %d, Polarisation product 00 is incorrect. Expected: %d + %dj, Received %d + %dj\n",
               uAnt1Index, uAnt2Index, sGeneratedPol00.i32Real, sGeneratedPol00.i32Imag, sActualPol00.i32Real,
               sActualPol00.i32Imag);
        return 0;
    }
    if (assert_complex_samples(sGeneratedPol01, sActualPol01) == 0)
    {
        printf("Ant 1 %d, Ant 2 %d, Polarisation product 01 is incorrect. Expected: %d + %dj, Received %d + %dj\n",
               uAnt1Index, uAnt2Index, sGeneratedPol01.i32Real, sGeneratedPol01.i32Imag, sActualPol01.i32Real,
               sActualPol01.i32Imag);
        return 0;
    }
    if (assert_complex_samples(sGeneratedPol10, sActualPol10) == 0)
    {
        printf("Ant 1 %d, Ant 2 %d, Polarisation product 10 is incorrect. Expected: %d + %dj, Received %d + %dj\n",
               uAnt1Index, uAnt2Index, sGeneratedPol10.i32Real, sGeneratedPol10.i32Imag, sActualPol10.i32Real,
               sActualPol10.i32Imag);
        return 0;
    }
    if (assert_complex_samples(sGeneratedPol11, sActualPol11) == 0)
    {
        printf("Ant 1 %d, Ant 2 %d, Polarisation product 11 is incorrect. Expected: %d + %dj, Received %d + %dj\n",
               uAnt1Index, uAnt2Index, sGeneratedPol11.i32Real, sGeneratedPol11.i32Imag, sActualPol11.i32Real,
               sActualPol11.i32Imag);
        return 0;
    }

    return 1;
}

int verify_xbengine(uint64_t *pu64Baselines, uint uBatchStartIndex, size_t ulNumBatches, size_t ulNumAnts,
                    size_t ulNumChans, size_t ulNumSamplesPerChan, size_t uNumPols)
{
    const size_t ulNumBaselines = ulNumAnts * (ulNumAnts + 1) / 2;
    const size_t ulBaselineStride = uNumPols * uNumPols;
    const size_t ulChannelStride = ulNumBaselines * ulBaselineStride;

    for (size_t ulChannelIndex = 0; ulChannelIndex < ulNumChans; ulChannelIndex++)
    {
        for (size_t ulAnt1Index = 0; ulAnt1Index < ulNumAnts; ulAnt1Index++)
        {
            for (size_t ulAnt2Index = 0; ulAnt2Index < ulAnt1Index + 1; ulAnt2Index++)
            {
                uint uBaselineIndex = get_baseline_index(ulAnt1Index, ulAnt2Index);
                uint uSampleIndex = ulChannelIndex * ulChannelStride + ulBaselineStride * uBaselineIndex;

                int iSuccess = verify_antpair_visibilities(
                    uBatchStartIndex, ulNumBatches, ulChannelIndex, ulNumSamplesPerChan, ulAnt1Index, ulAnt2Index,
                    pu64Baselines[uSampleIndex], pu64Baselines[uSampleIndex + 1], pu64Baselines[uSampleIndex + 2],
                    pu64Baselines[uSampleIndex + 3]);
                if (iSuccess == 0)
                {
                    return 0;
                }
            }
        }
    }

    return 1;
}