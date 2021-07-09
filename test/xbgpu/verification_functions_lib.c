/* C library containing methods for kernel data verification.
 *
 * This was created as it speeds up the verification process massively, when compared to its python-equivalent.
 *
 * The following verificaiton functions have been migrated to C to take advantage of this preformance increase.
 * 1. verify_reorder:
 *  - For verification of the data output by the precorrelation_reorder_kernel.
 * 2. verify_xbengine
 *  - Verification of the entire XBEngine pipeline.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#pragma region precorrelation_reorder_test

/**
 *  \brief Function that implements verification of data output by the pre-correlation reorder kernel.
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

#pragma endregion precorrelation_reorder_test

#pragma region xbengine_test

/**
 * Struct representing a complex number where the real and complex components are each stored as a 32-bit integer.
 *
 * This struct is used within the verify_xbengine(...) verification function scope.
 */
struct si32Complex
{
    int32_t i32Real; // Real component of the complex number.
    int32_t i32Imag; // IMaginary component of the comple number.
};

/**
 * Create an si32Complex complex struct from two 8-bit integers. If either of the given values are equal to -128, they
 * are changed to -127 as -128 is not supported by Nvidia Tensor cores.
 *
 * This function is used within in the verify_xbengine(...) verification function scope.
 *
 * \param[in] i8Real    Real component of complex number.
 * \param[in] i8Imag    Imaginary component of complex number.
 *
 * \return Struct representing the complex number.
 */
struct si32Complex createBoundedComplex(int8_t i8Real, int8_t i8Imag)
{
    if (i8Real == -128)
        i8Real = -127;
    if (i8Imag == -128)
        i8Imag = -127;
    struct si32Complex sReturn = {i8Real, i8Imag};
    return sReturn;
}

/**
 * Return the index in the visibilities matrix of the visibility produced by ant1 and ant2.
 *
 * This function is a C port of the katxbgpu.tensorcore_xengine_core.TensorCoreXEngineCore.get_baseline_index(...)
 * function.
 *
 * This function is used within in the verify_xbengine(...) verification function scope.
 *
 * \param[in] uAnt1    First antenna in the correlation pair.
 * \param[in] uAnt2    Second antenna in the correlation pair.
 *
 * \return Index of the visibilities matrix of the correlation pair.
 */
uint get_baseline_index(uint uAnt1, uint uAnt2)
{
    if (uAnt2 > uAnt1)
        return -1;
    return uAnt1 * (uAnt1 + 1) / 2 + uAnt2;
}

/**
 * Performs complex multiplication and accumulation: c = c + a*b*alpha. Where a,b and c are complex values and alpha
 * is a scalar.
 *
 * This function is used within in the verify_xbengine(...) verification function scope.
 *
 * \param[out]  sC     Pointer to complex struct c where calculated data is accumulated.
 * \param[in]   sA      Struct containing complex input value a.
 * \param[in]   sB      Struct containing complex input value a.
 * \param[in]   iAlpha  Scalar value alpha.
 */
void complex_multiply_scale_accumulate(struct si32Complex *sC, struct si32Complex sA, struct si32Complex sB, int iAlpha)
{
    sC->i32Real += (sA.i32Real * sB.i32Real + sA.i32Imag * sB.i32Imag) * iAlpha;
    sC->i32Imag += (sA.i32Imag * sB.i32Real - sA.i32Real * sB.i32Imag) * iAlpha;
}

/**
 * Compare two complex values.
 *
 * This function is used within in the verify_xbengine(...) verification function scope.
 *
 * \param[in] sComplexValue1    First antenna in the correlation pair.
 * \param[in] sComplexValue2    Second antenna in the correlation pair.
 *
 * \return Returns 1 if the values are equal and 0 if not.
 */
int compare_complex_values(struct si32Complex sComplexValue1, struct si32Complex sComplexValue2)
{
    if (sComplexValue1.i32Real != sComplexValue2.i32Real)
        return 0;
    if (sComplexValue1.i32Imag != sComplexValue2.i32Imag)
        return 0;
    return 1;
}

/**
 * Calculates the expected visibility data for all polarisation products for a pair of antennas and compares them to
 * the actual values generate by the xbengine pipeline. This function only checks the values for a single channel.
 *
 * This function assumes that the X-Engine input data is equal to the values assigned in the createHeaps(...) function
 * in xbengine_test.py
 *
 * This function is used within in the verify_xbengine(...) verification function scope.
 *
 * \param[in] ulBatchStartIndex     Index of the first batch of heaps in the accumulation epoch.
 * \param[in] ulNumBatches          Number of heaps per antenna in the accumulation epoch.
 * \param[in] ulChannelIndex        Index of the channel being considered.
 * \param[in] ulNumSamplesPerChan   Number of samples per channel in a single heap.
 * \param[in] ulAnt1Index           First antenna in the correlation pair.
 * \param[in] ulAnt2Index           Second antenna in the correlation pair.
 * \param[in] u64Pol00              Packed 32-bit complex value containing the pol0pol0 product of the correlation pair.
 * \param[in] u64Pol01              Packed 32-bit complex value containing the pol0pol1 product of the correlation pair.
 * \param[in] u64Pol10              Packed 32-bit complex value containing the pol1pol0 product of the correlation pair.
 * \param[in] u64Pol11              Packed 32-bit complex value containing the pol1pol1 product of the correlation pair.
 *
 * \return Returns 1 if the four polarisation poducts are all correct or 0 otherwise.
 */
int verify_antpair_visibilities(size_t ulBatchStartIndex, size_t ulNumBatches, size_t ulChannelIndex,
                                size_t ulNumSamplesPerChan, size_t ulAnt1Index, size_t ulAnt2Index, uint64_t u64Pol00,
                                uint64_t u64Pol01, uint64_t u64Pol10, uint64_t u64Pol11)
{
    // 1. Convert the 64 bit uint containing the actual complex samples into an si32Complex as it is easier to work
    // with,
    struct si32Complex sActualPol00 = {(int32_t)u64Pol00, (int32_t)(u64Pol00 >> 32)};
    struct si32Complex sActualPol01 = {(int32_t)u64Pol01, (int32_t)(u64Pol01 >> 32)};
    struct si32Complex sActualPol10 = {(int32_t)u64Pol10, (int32_t)(u64Pol10 >> 32)};
    struct si32Complex sActualPol11 = {(int32_t)u64Pol11, (int32_t)(u64Pol11 >> 32)};

    // 2. Calculate the values that we expect the visibilities to equal.
    // 2.1 Initialise complex values to store the generated data.
    struct si32Complex sGeneratedPol00 = {0, 0};
    struct si32Complex sGeneratedPol01 = {0, 0};
    struct si32Complex sGeneratedPol10 = {0, 0};
    struct si32Complex sGeneratedPol11 = {0, 0};

    // 2.2 For a specific channel in a heap, the values of all antenna samples are kept constant and can be multiplied
    // by the number of samples per channel (instead of generating them for each sample in the channel and added them
    // individually). This reduces computation time significantly. However for different batches the samples
    // change so we need iterate over each batch in the epoch and add all these values together.
    for (size_t ulBatchIndex = ulBatchStartIndex; ulBatchIndex < ulBatchStartIndex + ulNumBatches; ulBatchIndex++)
    {
        // 2.2.1 Generate the samples for the specific channel in this specific batch for each polarisation for each
        // antenna.
        int iSign = ulBatchIndex % 2 == 0 ? 1 : -1;
        struct si32Complex sAnt1Pol0 =
            createBoundedComplex((int8_t)(iSign * ulBatchIndex), (int8_t)(iSign * ulChannelIndex));
        struct si32Complex sAnt1Pol1 =
            createBoundedComplex((int8_t)(-iSign * ulAnt1Index), (int8_t)(-iSign * ulChannelIndex));
        struct si32Complex sAnt2Pol0 =
            createBoundedComplex((int8_t)(iSign * ulBatchIndex), (int8_t)(iSign * ulChannelIndex));
        struct si32Complex sAnt2Pol1 =
            createBoundedComplex((int8_t)(-iSign * ulAnt2Index), (int8_t)(-iSign * ulChannelIndex));

        // 2.2.2 Multiply the samples of the two antennas. There are ulNumSamplesPerChan of these identical samples in
        // a batch so we multiply the output by ulNumSamplesPerChan. This is then added to the value in sGeneratedPol00
        // which accumulates this across batches. This is repeated for all four polarisation pairs..
        complex_multiply_scale_accumulate(&sGeneratedPol00, sAnt1Pol0, sAnt2Pol0, ulNumSamplesPerChan);
        complex_multiply_scale_accumulate(&sGeneratedPol01, sAnt1Pol0, sAnt2Pol1, ulNumSamplesPerChan);
        complex_multiply_scale_accumulate(&sGeneratedPol10, sAnt1Pol1, sAnt2Pol0, ulNumSamplesPerChan);
        complex_multiply_scale_accumulate(&sGeneratedPol11, sAnt1Pol1, sAnt2Pol1, ulNumSamplesPerChan);
    }

    // 3. Compare the actual values of the polarisation produts to the generated complex values. Print an error message
    // and exit if the value is not as expected. Return 1 if all values are correct.
    if (compare_complex_values(sGeneratedPol00, sActualPol00) == 0)
    {
        printf("Ant 1 %ld, Ant 2 %ld, Polarisation product 00 is incorrect. Expected: %d + %dj, Received %d + %dj\n",
               ulAnt1Index, ulAnt2Index, sGeneratedPol00.i32Real, sGeneratedPol00.i32Imag, sActualPol00.i32Real,
               sActualPol00.i32Imag);
        return 0;
    }
    if (compare_complex_values(sGeneratedPol01, sActualPol01) == 0)
    {
        printf("Ant 1 %ld, Ant 2 %ld, Polarisation product 01 is incorrect. Expected: %d + %dj, Received %d + %dj\n",
               ulAnt1Index, ulAnt2Index, sGeneratedPol01.i32Real, sGeneratedPol01.i32Imag, sActualPol01.i32Real,
               sActualPol01.i32Imag);
        return 0;
    }
    if (compare_complex_values(sGeneratedPol10, sActualPol10) == 0)
    {
        printf("Ant 1 %ld, Ant 2 %ld, Polarisation product 10 is incorrect. Expected: %d + %dj, Received %d + %dj\n",
               ulAnt1Index, ulAnt2Index, sGeneratedPol10.i32Real, sGeneratedPol10.i32Imag, sActualPol10.i32Real,
               sActualPol10.i32Imag);
        return 0;
    }
    if (compare_complex_values(sGeneratedPol11, sActualPol11) == 0)
    {
        printf("Ant 1 %ld, Ant 2 %ld, Polarisation product 11 is incorrect. Expected: %d + %dj, Received %d + %dj\n",
               ulAnt1Index, ulAnt2Index, sGeneratedPol11.i32Real, sGeneratedPol11.i32Imag, sActualPol11.i32Real,
               sActualPol11.i32Imag);
        return 0;
    }

    return 1;
}

/**
 * Function called by the xbengine unit test to check that the data out of the engine is correct.
 *
 * This function assumes that the X-Engine input data is equal to the values assigned in the createHeaps(...) function
 * in xbengine_test.py and that the output visibilities data is formatted as described in the
 * katxbgpu.tensorcore_xengine_core module.
 *
 * This function is called directly in the xbengine_test.py module.
 *
 * \param[in] pu64Baselines         Pointer to the visibilities matrix generated by the xbengine pipeline.
 * \param[in] ulBatchStartIndex     Index of the first batch of heaps in the accumulation epoch.
 * \param[in] ulNumBatches          Number of batches (heaps per antenna) in the accumulation epoch.
 * \param[in] ulNumChannels         Number of channels within a single heap.
 * \param[in] ulNumSamplesPerChan   Number of samples per channel in a single heap.
 * \param[in] ulNumPols             Number of polarisations per antenna (only 2 is supported at the moment).
 *
 * \return Returns 1 the visibilities are all correct or 0 otherwise.
 */
int verify_xbengine(uint64_t *pu64Baselines, size_t ulBatchStartIndex, size_t ulNumBatches, size_t ulNumAnts,
                    size_t ulNumChans, size_t ulNumSamplesPerChan, size_t uNumPols)
{
    // 1. Determine the different strides used when calculating the location of the correlation produts for an antenna
    // pair in the pu64Baselines visibilities matrix. This matrix will have the shape as described in the
    // katxbgpu.tensorcore_xengine_core module and the strides generated here correspond to the strides in that matrix.
    const size_t ulNumBaselines = ulNumAnts * (ulNumAnts + 1) / 2;
    const size_t ulBaselineStride = uNumPols * uNumPols;
    const size_t ulChannelStride = ulNumBaselines * ulBaselineStride;

    // 2. We need to iterate through all the different channels and baselines in the visibilities matrix and ensure that
    // they are equal to the calculated values.
    for (size_t ulChannelIndex = 0; ulChannelIndex < ulNumChans; ulChannelIndex++)
    {
        for (size_t ulAnt1Index = 0; ulAnt1Index < ulNumAnts; ulAnt1Index++)
        {
            for (size_t ulAnt2Index = 0; ulAnt2Index < ulAnt1Index + 1; ulAnt2Index++)
            {
                // 2.1 For a specific antenna pair and channel, determine the location of the polarisation products in
                // the visibilities matrix.
                size_t ulBaselineIndex = get_baseline_index(ulAnt1Index, ulAnt2Index);
                size_t ulSampleIndex = ulChannelIndex * ulChannelStride + ulBaselineIndex * ulBaselineStride;

                // 2.2 Check that the polarsiation products for the specific antenna pair are equal to what we expect
                // them to be. If the values are incorrect this test immediately exits.
                int iSuccess = verify_antpair_visibilities(
                    ulBatchStartIndex, ulNumBatches, ulChannelIndex, ulNumSamplesPerChan, ulAnt1Index, ulAnt2Index,
                    pu64Baselines[ulSampleIndex], pu64Baselines[ulSampleIndex + 1], pu64Baselines[ulSampleIndex + 2],
                    pu64Baselines[ulSampleIndex + 3]);
                if (iSuccess == 0)
                {
                    return 0;
                }
            }
        }
    }

    return 1;
}

#pragma endregion xbengine_test