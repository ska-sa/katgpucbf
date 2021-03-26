/*
    C library containing methods for post-kernel execution:
    1. verify_reorder
        - For verification of the data output by the precorrelation_reorder_kernel


 */

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

/**
 * Reorder verification method
 * - Implemented in C to (hopefully) run quicker/more efficiently
 * 
 * @param 
 */

int verify_reorder(int8_t *pi8Array, int8_t *pi8ArrayReordered, int iNumBatches, int iNumAnts, int iNumChans, int iNumSamples, int iNumPols, int iTimesPerBlock)
{
    int iCurrIndex, iRemIndex, iBatchCounter, iMatrixSize;
    int iAntIndex, iChanIndex, iTimeIndex, iPolIndex;
    int iAntStride, iChanStride, iTimeStride, iMatrixStride;
    iMatrixSize = iNumAnts * iNumChans * iNumSamples * iNumPols;
    iAntStride = iNumChans * iNumSamples * iNumPols;
    iChanStride = iNumSamples * iNumPols;
    iTimeStride = iNumPols;
    // iPolStride = 1;

    // Declaring in their order of dimensionality for the new matrix
    int iNewIndex, iNewChanOffset, iTimeOuterOffset, iNewAntOffset, iNewPolOffset;
    int iTimeIndexOuter, iTimeIndexInner;
    int iNewAntStride, iNewChanStride, iNewSampleStride, iNewPolStride;
    iNewChanStride = (iNumSamples / iTimesPerBlock) * iNumAnts * iNumPols * iTimesPerBlock;
    iNewSampleStride = iNumAnts * iNumPols * iTimesPerBlock;
    iNewAntStride = iNumPols * iTimesPerBlock;
    iNewPolStride = iTimesPerBlock;

    for (iBatchCounter = 0; iBatchCounter < iNumBatches; iBatchCounter++)
    {
        iMatrixStride = iBatchCounter * iNumAnts * iNumChans * iNumSamples * iNumPols;
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
            iTimeIndexOuter = iTimeIndex / iTimesPerBlock;
            iTimeOuterOffset = iTimeIndexOuter * iNewSampleStride;
            iNewAntOffset = iAntIndex * iNewAntStride;
            iNewPolOffset = iPolIndex * iNewPolStride;
            iTimeIndexInner = iTimeIndex % iTimesPerBlock; // ?

            iNewIndex = iNewChanOffset + iTimeOuterOffset + iNewAntOffset + \
                        iNewPolOffset + iTimeIndexInner + iMatrixStride;

            if (pi8ArrayReordered[iNewIndex] != pi8Array[iCurrIndex + iMatrixStride])
            // if (iOrigData != iNewData)
            {
                // Problem
                // printf("\nReordered: %d at index %d\n != Original: %d at index %d",\
                //         pi8ArrayReordered[iNewIndex], iNewIndex,\
                //         pi8Array[iCurrIndex + iMatrixStride], (iCurrIndex + iMatrixStride));
                printf("\n========================================\n");
                printf("Original indices:\n");
                printf("\t Batch: %d | Ant: %d | Chan: %d | Sample: %d | Pol: %d\n", \
                            iBatchCounter, iAntIndex, iChanIndex, iTimeIndex, iPolIndex);
                printf("\n---------------------------------------\n");
                printf("Reorder indices:\n");
                printf("\t Batch: %d | Chan: %d | Sample/TpB: %d | Ant: %d | Pol: %d | TpB: %d\n", \
                        iBatchCounter, iChanIndex, iTimeIndexOuter, iAntIndex, iPolIndex, iTimeIndexInner);
                return 0;
            }
        }
    }
    
    // If we get here, all is well
    return 1;
}
