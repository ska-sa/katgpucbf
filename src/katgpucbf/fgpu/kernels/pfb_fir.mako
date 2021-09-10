/*******************************************************************************
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

<%include file="/port.mako"/>

#define WGS ${wgs}
#define TAPS ${taps}

/* Return the byte index in the chunk where the start of your 10-bit sample will
 * be. The sample will be spread over two successive bytes but we just need to
 * know where the first one is.
 */
DEVICE_FN int samples_to_bytes(int samples)
{
    return samples + (samples >> 2);
}

/* Get the 10-bit sample at the given index from the chunk of samples, shake off
 * the unwanted surrounding pieces, and return as a float.
 */
DEVICE_FN float get_sample_10bit(const GLOBAL uchar * RESTRICT in, int idx)
{
    // We were given the sample number. Get the byte index.
    int byte_idx = samples_to_bytes(idx);

    // We need two bytes to make sure we get all 10 bits of the sample.
    // TODO: probably better to make this `int32_t` instead of just `int`?
    int raw = (in[byte_idx] << 8) + in[byte_idx + 1];

    //TODO replace the magic numbers 16 and 22 with expressions based on sizeof(int) * CHAR_BIT

    /* Bitwise AND with 3 (0b0000011) is equivalent to modulo 4. The position of
     * the MSB of the sample we want in the byte follows a pattern that repeats
     * every 4 bytes. (0, 2, 4, 6, 0, 2, 4, 6, etc)
     * The extra 16 is necessary because we use a 32-bit int value and the top
     * 16 bits are unused.
     */
    int shift_left = 2 * (idx & 3) + 16;

    /* Shift left to get rid of the stuff above, and right by 22 to get the
     * 10 bits that we are actually interested in. This relies on the
     * shift-right operation doing sign extension if the sample is negative.
     * This int gets promoted to a float in the return. Since we are going from
     * 10 bits to 24, we won't lose any precision, but if things change later
     * (e.g. a digitiser with more precision, or casting to __half) then this
     * might be a concern.
     */
    return (raw << shift_left) >> 22;
}

/* Each work-item is responsible for a run of input values with stride `step`.
 *
 * This approach becomes very register-heavy as the number of taps increases.
 * A better approach may be to have the work group cooperatively load a
 * rectangle of data into local memory, transpose, and work from there. While
 * local memory is smaller than the register file, multiple threads will read
 * the same value.
 */
KERNEL REQD_WORK_GROUP_SIZE(WGS, 1, 1) void pfb_fir(
    GLOBAL float * RESTRICT out,          // Output memory
    const GLOBAL uchar * RESTRICT in,     // Input data (digitiser samples)
    const GLOBAL float * RESTRICT weights,// Weights for the PFB-FIR filter.
    int n,                                // Size of the `out` array, to avoid going out-of-bounds.
    int step,                             // Number of input samples needed for each spectrum, i.e. 2*channels.
    int stepy,                            // Size of data that will be worked on by a single thread-block.
    int in_offset,                        // Number of samples to skip from the start of *in.
    int out_offset           // Number of samples to skip from the start of *out. Must be a multiple of `step` to make sense.
)
{
    // Figure out where our thread block has to work.
    int group_x = get_group_id(0);
    int group_y = get_group_id(1);

    // Figure out where this thread has to work.
    int lid = get_local_id(0);
    int pos = group_x * WGS + lid; // pos is the position within the step (i.e. spectrum) that this thread will work on.
    int offset = group_y * stepy + pos; // This thread probably doesn't work on the very beginning of the data, so we
                                        // make the indexing easier for ourselves later.

    //Increment this pointer because this thread may not need to write to the
    // beginning of the block.
    out += offset + out_offset;

    // can't skip individual (input) samples with pointer arithmetic, so track in_offset
    in_offset += offset;
    n -= offset;

    /* Here we fill up the taps of the FIR before we bother to do any outputs.
     * We assume we are not interested in the initial transient spectra.
     * We prime all but one of the taps with samples of data. The last one will
     * be filled in later as part of the main loop.
     */
    float samples[TAPS];
    for (int i = 0; i < TAPS - 1; i++)
    {
        samples[i] = get_sample_10bit(in, in_offset);
        in_offset += step;  // and we shift the in_offset along, this makes the indexing simpler later.
    }

    // Load the relevant weights for this branch of the PFB-FIR.
    float rweights[TAPS];
#pragma unroll
    for (int i = 0; i < TAPS; i++)
        rweights[i] = weights[i * step + pos];

    // This thread will process the same equivalent sample in `rows` successive
    // output "spectra" worth of data.
    int rows = stepy / step;

    // Unrolling by factor of TAPS makes the sample index known at compile time.
#pragma unroll ${taps}
    for (int i = 0; i < rows; i++)  // We'll be at our most memory-bandwidth-efficient if rows >> TAPS. Launching ~256K threads should ensure this.
    {
        int idx = i * step;
        if (idx >= n) // Edge case - chunk size might not be a perfect multiple of number of rows.
            break;

        /* Each FIR output sample only needs one new sample, and TAPS-1 old ones.
         * This line reads the new one in over the oldest one (or the one we
         * missed above, in the case of the first loop) using this modulo trick.
         * This, combined with the way they are then read out below, avoids
         * having to manually shift things along in the array each loop.
         */
        samples[(i + TAPS - 1) % TAPS] = get_sample_10bit(in, in_offset + idx);

        // Implement the actual FIR filter by multiplying samples by weights and summing.
        float sum = 0.0f;
        for (int j = 0; j < TAPS; j++)
            // [(i + j) % TAPS] matches the sample with the appropriate weight.
            sum += rweights[j] * samples[(i + j) % TAPS];
        // Sum written out to global memory.
        out[idx] = sum;
    }
}
