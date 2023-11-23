/*******************************************************************************
 * Copyright (c) 2023, National Research Foundation (SARAO)
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

#define N_BEAMS ${len(beam_pols)}
// Number of spectra processed in each work group
#define BLOCK_SPECTRA ${block_spectra}
// Number of antennas whose data is loaded and processed at a time
#define BATCH_ANTENNAS 16
#define QMAX 127

// Quantise, and return saturation flag
// TODO: this is copied from postproc.py. Create a shared implementation.
DEVICE_FN bool quant(float value, int *out)
{
    float clamped = fminf(fmaxf(value, -QMAX), QMAX);
    bool saturated = value != clamped;

    int q;
    // Convert to s8, round to nearest integer, and saturate
    // (saturation is redundant due to the test above, but automatic
    // for float-to-int conversions on CUDA hardware).
#ifdef __OPENCL_VERSION__
    q = convert_char_sat_rte(clamped);
#else
    asm("cvt.rni.sat.s8.f32 %0, %1;" : "=r" (q) : "f"(clamped));
#endif
    *out = q;
    return saturated;
}

// Compute a * b in complex numbers
DEVICE_FN float2 cmul(float2 a, float2 b)
{
    return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// Compute a * b + c in complex numbers
DEVICE_FN float2 cmad(float2 a, float2 b, float2 c)
{
    return make_float2(c.x + a.x * b.x - a.y * b.y, c.y + a.x * b.y + a.y * b.x);
}

// Each thread computes all beams for one (channel, time)
KERNEL REQD_WORK_GROUP_SIZE(BLOCK_SPECTRA, 1, 1) void beamform(
    GLOBAL char2 * RESTRICT out,             // shape frame, beam, channel, time
    GLOBAL const char4 * RESTRICT in,        // shape frame, antenna, channel, time, pol
    GLOBAL const float2 * RESTRICT weights,  // shape antenna, beam, tightly packed
    GLOBAL const float * RESTRICT delays,    // shape antenna, beam, tightly packed
    int out_stride,                          // elements between channels
    int out_beam_stride,                     // elements between beams
    int out_frame_stride,                    // elements between frames
    int in_stride,                           // elements between channels
    int in_antenna_stride,                   // elements between antennas
    int in_frame_stride,                     // elements between frames
    int n_ants,
    int n_spectra
)
{
    /* Local storage for computed weights. The logical shape is
     * [BATCH_ANTENNAS][N_BEAMS] but some parts of the code index
     * it as a linear array.
     */
    LOCAL_DECL float2 l_weights[BATCH_ANTENNAS * N_BEAMS];

    int lid = get_local_id(0);
    int spectrum = get_global_id(0);
    int channel = get_global_id(1);
    int frame = get_global_id(2);
    /* Whether this thread works on actual input/output values. Some work
     * items will hang off the end of the data but need to keep running to
     * participate in the coefficient calculations.
     */
    bool valid = (spectrum < n_spectra);
    // Point to the first input/output handled by this work item
    in += frame * in_frame_stride + channel * in_stride + spectrum;
    out += frame * out_frame_stride + channel * out_stride + spectrum;

    // Zero out the accumulators
    float2 accum[N_BEAMS];
    for (int i = 0; i < N_BEAMS; i++)
        accum[i] = make_float2(0.0f, 0.0f);

    for (int a_batch = 0; a_batch < n_ants; a_batch += BATCH_ANTENNAS)
    {
        int batch_size = min(BATCH_ANTENNAS, n_ants - a_batch);
        /* Precompute the weights for all beams for this antenna batch.
         * The work items collectively iterate over the interval
         * [0, N_BEAMS * batch_size).
         */
        for (int i = lid; i < N_BEAMS * batch_size; i += BLOCK_SPECTRA)
        {
            // Address into the global memory
            int addr = a_batch * N_BEAMS + i;
            float2 w = weights[addr];
            float cd = channel * delays[addr];
            float2 rot;
            sincospif(cd, &rot.y, &rot.x);
            l_weights[i] = cmul(w, rot);
        }
        BARRIER(); // Complete all weight calculations before using them

        if (valid)
        {
            for (int a = 0; a < batch_size; a++)
            {
                char4 sample = *in;
                // Convert to float, and split the polarisations into an
                // array.
                float2 sample_pols[2] = {make_float2(sample.x, sample.y), make_float2(sample.z, sample.w)};
                in += in_antenna_stride;

                /* Iterate over all beams. This is done with a mako loop
                 * rather than a C loop to ensure that the polarisation is
                 * resolved at compile time rather than retrieved from an
                 * array.
                 */
% for i, pol in enumerate(beam_pols):
                {
                    int i = ${i};
                    int pol = ${pol};
                    accum[i] = cmad(l_weights[a * N_BEAMS + i], sample_pols[pol], accum[i]);
                }
% endfor
            }
        }
        BARRIER();  // protect against next loop iteration overwriting l_weights
    }

    if (!valid)
        return;
    // Write accumulators to output
    for (int i = 0; i < N_BEAMS; i++)
    {
        int re, im;
        // TODO: report saturation flags somewhere
        quant(accum[i].x, &re);
        quant(accum[i].y, &im);
        out[out_beam_stride * i] = make_char2(re, im);
    }
}
