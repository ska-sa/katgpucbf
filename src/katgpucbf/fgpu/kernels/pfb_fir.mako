/*******************************************************************************
 * Copyright (c) 2020-2023, National Research Foundation (SARAO)
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
<%namespace name="wg_reduce" file="/wg_reduce.mako"/>

#define WGS ${wgs}
#define TAPS ${taps}
#define CHANNELS ${channels}
#define UNZIP_FACTOR ${unzip_factor}
#define DIG_SAMPLE_BITS ${dig_sample_bits}

<%include file="unpack.mako"/>

${wg_reduce.define_scratch('unsigned long long', wgs, 'scratch_t', allow_shuffle=True)}
${wg_reduce.define_function('unsigned long long', wgs, 'reduce', 'scratch_t', wg_reduce.op_plus, allow_shuffle=True, broadcast=False)}

/* Apply unzipping to an output index.
 */
DEVICE_FN static unsigned int shuffle_index(unsigned int idx)
{
    // x.bit_length - 1 is log2(x) when x is a power of 2
    const int low_bits = ${unzip_factor.bit_length() - 1};
    const int high_bits = ${channels.bit_length() - unzip_factor.bit_length()};
    // Bits to modify
    const int mask = (2 << (low_bits + high_bits)) - 2;
    unsigned int orig = idx & mask;
    unsigned int swapped = ((orig >> low_bits) | (orig << high_bits)) & mask;
    return (idx & ~mask) | swapped;
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
    GLOBAL unsigned long long * RESTRICT out_total_power,  // Sum of squares of samples (incremented)
    const GLOBAL uchar * RESTRICT in,     // Input data (digitiser samples)
    const GLOBAL float * RESTRICT weights,// Weights for the PFB-FIR filter.
    int n,                                // Size of the `out` array, to avoid going out-of-bounds.
    int stepy,                            // Size of data that will be worked on by a single thread-block.
    int in_offset,                        // Number of samples to skip from the start of *in.
    int out_offset           // Number of samples to skip from the start of *out. Must be a multiple of `step` to make sense.
)
{
    const int step = 2 * CHANNELS;
    // Figure out where our thread block has to work.
    int group_x = get_group_id(0);
    int group_y = get_group_id(1);

    // Figure out where this thread has to work.
    int lid = get_local_id(0);
    int pos = group_x * WGS + lid; // pos is the position within the step (i.e. spectrum) that this thread will work on.
    int offset = group_y * stepy + pos; // This thread probably doesn't work on the very beginning of the data, so we
                                        // make the indexing easier for ourselves later.

    // Increment this pointer because this thread may not need to write to the
    // beginning of the block.
    out += shuffle_index(offset + out_offset);

    // can't skip individual (input) samples with pointer arithmetic, so track in_offset
    in_offset += offset;
    n -= offset;

    /* Here we fill up the taps of the FIR before we bother to do any outputs.
     * We assume we are not interested in the initial transient spectra.
     * We prime all but one of the taps with samples of data. The last one will
     * be filled in later as part of the main loop.
     *
     * These samples are deliberately not included in total_power, because they
     * have already been counted by a previous workgroup (except for the very
     * first samples in the stream, or after lost data, but that's a corner
     * case not worth worrying about).
     */
    float samples[TAPS];
    unpack_t unpack;
    unpack_init(&unpack, in, in_offset);
    for (int i = 0; i < TAPS - 1; i++)
    {
        samples[i] = unpack_read(&unpack);
        unpack_advance(&unpack, step);
    }

    // Load the relevant weights for this branch of the PFB-FIR.
    float rweights[TAPS];
#pragma unroll
    for (int i = 0; i < TAPS; i++)
        rweights[i] = weights[i * step + pos];

    // This thread will process the same equivalent sample in `rows` successive
    // output "spectra" worth of data.
    int rows = stepy / step;

    unsigned long long total_power = 0;
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
        int sample = unpack_read(&unpack);
        unpack_advance(&unpack, step);
        total_power += sample * sample;
        samples[(i + TAPS - 1) % TAPS] = (float) sample;

        // Implement the actual FIR filter by multiplying samples by weights and summing.
        float sum = 0.0f;
        for (int j = 0; j < TAPS; j++)
            // [(i + j) % TAPS] matches the sample with the appropriate weight.
            sum += rweights[j] * samples[(i + j) % TAPS];
        // Sum written out to global memory.
        out[idx] = sum;
    }

    // Reduce total_power across work items, to reduce the number of atomics needed.
    LOCAL_DECL scratch_t scratch;
    total_power = reduce(total_power, lid, &scratch);
    if (lid == 0)
        atomicAdd(out_total_power, total_power);
}
