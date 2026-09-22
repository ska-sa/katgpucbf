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

<% do_total_power = not complex_input %>
#define WGS_X ${wgs_x}
#define WGS_Y ${wgs_y}
#define AMP_Y ${amp_y}
#define TAPS ${taps}
#define CHANNELS ${channels}
#define TOTAL_POWER_SPECTRA ${total_power_spectra}
#define UNZIP_FACTOR ${unzip_factor}
#define INPUT_SAMPLE_BITS ${input_sample_bits}
#define WEIGHT_INDEX_SHIFT ${1 if complex_input else 0}
#define N_POLS ${n_pols}

% if complex_input:

<%include file="unpack_float.mako"/>

% else:

<%include file="unpack.mako"/>
<%namespace name="wg_reduce" file="/wg_reduce.mako"/>
${wg_reduce.define_scratch('unsigned long long', wgs_x * wgs_y, 'scratch_t', allow_shuffle=True)}
${wg_reduce.define_function('unsigned long long', wgs_x * wgs_y, 'reduce', 'scratch_t', wg_reduce.op_plus, allow_shuffle=True, broadcast=False)}

% endif

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
 * WGS_Y work-items will collaboratively load the necessary data.
 *
 * This approach becomes very register-heavy as the number of taps increases.
 * A better approach may be to have the work group cooperatively load a
 * rectangle of data into local memory, transpose, and work from there. While
 * local memory is smaller than the register file, multiple threads will read
 * the same value.
 *
 * When the input is complex, all the parameters and internal indexing treat
 * it as if it were real, just with each pair of adjacent reals using the same
 * weight.
 */
KERNEL REQD_WORK_GROUP_SIZE(WGS_X, WGS_Y, 1) void pfb_fir(
    GLOBAL float * RESTRICT out,          // Output memory
% if complex_input:
    const GLOBAL float * RESTRICT in,     // Input data (down-converted samples)
% else:
    GLOBAL unsigned long long * RESTRICT out_total_power,  // Sum of squares of samples (incremented)
    int out_total_power_stride,           // Offset to `out_total_power` between pols
    const GLOBAL unsigned char * RESTRICT in,     // Input data (digitiser samples)
% endif
    const GLOBAL float * RESTRICT weights,// Weights for the PFB-FIR filter.
    int out_stride,                       // Offset to `out` between pols
    int in_stride,                        // Offset to `in` between pols
    int n,                                // Size of the `out` array (in spectra), to avoid going out-of-bounds.
% for pol in range(n_pols):
    int in_offset${pol},                  // Number of samples to skip from the start of *in
% endfor
    // Number of samples to skip from the start of *out, in spectra.
    int out_offset
)
{
    const unsigned int step = 2 * CHANNELS;
    const unsigned int rows_in = WGS_Y * AMP_Y;
    const unsigned int rows_out = rows_in - (TAPS - 1);

    LOCAL_DECL short int raw_samples[rows_in][WGS_X];

    // Figure out where our thread block has to work.
    int group_y = get_group_id(1) * rows_out;
    int pol = get_group_id(2);
    int in_offset;
    switch (pol)
    {
% for pol in range(n_pols):
    case ${pol}:
        in_offset = in_offset${pol};
        break;
% endfor
    }

    // Figure out where this thread has to work.
    int lid_x = get_local_id(0);
    int lid_y = get_local_id(1);
    // pos is the position within the step (i.e. spectrum) that this thread will work on.
    // This is equivalent to get_global_id(0) but reuses known values
    int pos = get_group_id(0) * WGS_X + lid_x;

    // can't skip individual (input) samples with pointer arithmetic, so track in_offset
    // (the first sample to be loaded by this thread)
    in_offset += (group_y + lid_y) * step + pos;
    in += pol * in_stride;

    // Increment this pointer because this thread may not need to write to the
    // beginning of the block.
    out += pol * out_stride + shuffle_index(pos);
% if do_total_power:
    out_total_power += pol * out_total_power_stride;
% endif

    /* Load the data. There are probably a few ways to optimise this:
     * - Convert to float here, so that we don't have to the (expensive)
     *   int->float conversion multiple times. That will double storage.
     * - On the other extreme, could just load the raw bytes (with a
     *   tensor memory unit if available) to save memory, and do all
     *   decoding later.
     * - Some work could start before all the memory is loaded, if we
     *   use finer-grained split barriers.
     */
    unpack_t unpack;
    unpack_init(&unpack, in, in_offset);
    for (int i = 0; i < rows_in; i += WGS_Y)
    {
        raw_samples[i + lid_y][lid_x] = unpack_read(&unpack); // TODO: avoid reading past the end
        unpack_advance(&unpack, step * WGS_Y);
    }
    BARRIER();

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
    int local_row = lid_y * AMP_Y;

#pragma unroll
    for (int i = 0; i < TAPS - 1; i++)
    {
        // Load the sample (write to i + 1 because we start the main loop by shuffling down)
        samples[i + 1] = raw_samples[local_row++][lid_x];
    }

    // Load the relevant weights for this branch of the PFB-FIR.
    // With complex_input, we shift the index to allow a single weight
    // to apply to both the real and imaginary components.
    // TODO: could load them through shared memory first?
    float rweights[TAPS];
#pragma unroll
    for (int i = 0; i < TAPS; i++)
        rweights[i] = weights[(i * step + pos) >> WEIGHT_INDEX_SHIFT];

    // This work-group will process up to (but excluding) this spectrum.
    int out_group_start_y = group_y + out_offset;  // first spectrum number to write in output
    int out_group_stop_y = min(n, out_group_start_y + rows_out);
    // This work-item will process this range of spectra
    // TODO: rebalance the work since rows_out < rows_in?
    int out_start_y = out_group_start_y + lid_y * AMP_Y;
    int out_stop_y = min(out_group_stop_y, out_start_y + AMP_Y);

% if not do_total_power:
    for (int i = out_start_y; i < out_stop_y; i++)
    {
        {  // Block just to balance things with the not complex_input case.
% else:
    unsigned long long total_power = 0;
    // Note: this while loop must be group-uniform, because we do a group-wise
    // reduction inside it
    int out_group_y = out_group_start_y;
    while (out_group_y < out_group_stop_y)
    {
        // Determine the next boundary at which we need to emit
        // accumulated total_power.
        // TODO: rewrite without using division
        int group_stop = min(out_group_stop_y, (out_group_y / TOTAL_POWER_SPECTRA + 1) * TOTAL_POWER_SPECTRA);
        int stop = min(group_stop, out_stop_y);
        for (int i = max(out_start_y, out_group_y); i < stop; i++)
        {
% endif
            // Load the raw data for the sample
            sample_t sample = raw_samples[local_row++][lid_x];
            // Shuffle down the samples to make room for the new one
            for (int j = 0; j < TAPS - 1; j++)
                samples[j] = samples[j + 1];

            /* Each FIR output sample only needs one new sample, and TAPS-1 old
             * ones. Read the new one into the array, and also use it to compute
             * total power.
             */
% if do_total_power:
            total_power += sample * sample;
% endif
            samples[TAPS - 1] = (float) sample;

            // Implement the actual FIR filter by multiplying samples by weights and summing.
            float sum = rweights[0] * samples[0];
            for (int j = 1; j < TAPS; j++)
                sum += rweights[j] * samples[j];
            // Sum written out to global memory.
            out[i * step] = sum;
        }
% if do_total_power:
        // Reduce total_power across work items, to reduce the number of atomics needed.
        // TODO: use 32-bit reduction when sample_bits is small enough.
        LOCAL_DECL scratch_t scratch;
        int lid = lid_y * WGS_X + lid_x;
        total_power = reduce(total_power, lid, &scratch);
        if (lid == 0)
            atomicAdd(&out_total_power[out_group_y / TOTAL_POWER_SPECTRA], total_power);
        total_power = 0;
        out_group_y = group_stop;
% endif
    }
}
