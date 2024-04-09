/*******************************************************************************
 * Copyright (c) 2023-2024, National Research Foundation (SARAO)
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

extern "C++"  // PyCUDA wraps the whole file in extern "C"
{
#include <curand_kernel.h>
}

<%include file="/port.mako"/>
<%include file="/kernels/complex.mako"/>
<%include file="/kernels/quant.mako"/>

<%
n_beams = len(beam_pols)
batch_beams = min(32, n_beams)
%>

#define N_BEAMS ${n_beams}
// Number of spectra processed in each work group
#define BLOCK_SPECTRA ${block_spectra}
// Number of antennas whose data is loaded and processed at a time
#define BATCH_ANTENNAS 16
// Number of beams processed at a time
#define BATCH_BEAMS ${batch_beams}

/// Generate a random value in (-0.5, 0.5)
DEVICE_FN float dither(GLOBAL curandStateXORWOW_t *state)
{
    /* This magic value is chosen so that the largest possible return value
     * can be added to 127 and still produce 127.49999 rather than 127.5
     * (found experimentally). That ensures that exact integer values will not
     * be altered by dithering.
     */
    const float scale = 2.3282709e-10f;  // == 0xffff7f00p-64
    /* curand(state) returns a value in [0, 2**32). Casting it to int gives
     * a value in [-2**31, 2**31).
     */
    int x = int(curand(state));
    /* Add 1 to x if x is negative. This gives a distribution with zero mean
     * There is a tiny non-uniformity because 0 is twice as likely to appear as
     * other values in (-2**31, 2**31).
     */
    x -= x >> 31;
    return x * scale;
}

// Each thread computes all beams for one (channel, time)
KERNEL REQD_WORK_GROUP_SIZE(BLOCK_SPECTRA, 1, 1) void beamform(
    GLOBAL char2 * RESTRICT out,             // shape frame, beam, channel, time
    GLOBAL const char4 * RESTRICT in,        // shape frame, antenna, channel, time, pol
    GLOBAL const cplx * RESTRICT weights,    // shape antenna, beam, tightly packed
    GLOBAL const float * RESTRICT delays,    // shape antenna, beam, tightly packed
    GLOBAL curandStateXORWOW_t * RESTRICT rand_state,  // shape frame, channel, time (packed)
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
     * [BATCH_ANTENNAS][BATCH_BEAMS] but some parts of the code index
     * it as a linear array.
     */
    LOCAL_DECL cplx l_weights[BATCH_ANTENNAS * BATCH_BEAMS];
    const int beam_pols[N_BEAMS] = { ${ ", ".join(str(p) for p in beam_pols) } };
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
    rand_state += (frame * get_num_groups(1) + channel) * n_spectra + spectrum;

    /* It's critical that this loop is unrolled, so that b_batch_size is known at
     * compile time.
     */
#pragma unroll
    for (int b_batch = 0; b_batch < N_BEAMS; b_batch += BATCH_BEAMS)
    {
        const int b_batch_size = min(BATCH_BEAMS, N_BEAMS - b_batch);
        const char4 *batch_in = in;

        // Zero out the accumulators
        cplx accum[BATCH_BEAMS];
        for (int i = 0; i < BATCH_BEAMS; i++)
            accum[i] = make_float2(0.0f, 0.0f);

        for (int a_batch = 0; a_batch < n_ants; a_batch += BATCH_ANTENNAS)
        {
            int a_batch_size = min(BATCH_ANTENNAS, n_ants - a_batch);
            /* Precompute the weights for this batch.
             * The work items collectively iterate over the interval
             * [0, b_batch_size * a_batch_size).
             */
            for (int i = lid; i < b_batch_size * a_batch_size; i += BLOCK_SPECTRA)
            {
                int beam = b_batch + i % b_batch_size;
                int ant = a_batch + i / b_batch_size;
                // Address into the global memory
                int addr = ant * N_BEAMS + beam;
                cplx w = weights[addr];
                float cd = channel * delays[addr];
                cplx rot;
                sincospif(cd, &rot.y, &rot.x);
                l_weights[i] = cmul(w, rot);
            }
            BARRIER(); // Complete all weight calculations before using them

            if (valid)
            {
                for (int a = 0; a < a_batch_size; a++)
                {
                    char4 sample = *batch_in;
                    // Convert to float, and split the polarisations into an
                    // array.
                    cplx sample_pols[2] = {make_float2(sample.x, sample.y), make_float2(sample.z, sample.w)};
                    batch_in += in_antenna_stride;

                    /* It's critical that this loop gets unrolled, so that `pol` is
                     * known at compile time.
                     */
#pragma unroll
                    for (int i = 0; i < b_batch_size; i++)
                    {
                        int beam = b_batch + i;
                        int pol = beam_pols[beam];
                        accum[i] = cmad(l_weights[a * b_batch_size + i], sample_pols[pol], accum[i]);
                    }
                }
            }
            BARRIER();  // protect against next loop iteration overwriting l_weights
        }

        if (valid)
        {
            // Write accumulators to output
            for (int i = 0; i < b_batch_size; i++)
            {
                int re, im;
                int beam = i + b_batch;
                // TODO: report saturation flags somewhere
                quant_8bit(accum[i].x + dither(rand_state), &re);
                quant_8bit(accum[i].y + dither(rand_state), &im);
                out[out_beam_stride * beam] = make_char2(re, im);
            }
        }
    }
}
