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
#define BLOCK_SPECTRA ${block_spectra}
#define BLOCK_CHANNELS ${block_channels}
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
    return make_float2(a.x * b.x - a.y * b.y + c.x, a.x * b.y + a.y * b.x + c.y);
}

// out: shape frame, beam, channel, time
// in: shape frame, antenna, channel, time, pol
// weight: shape antenna, beam, tightly packed
// delays: shape antenna, beam, tightly packed
// Each thread computes all beams for one (channel, time)
KERNEL REQD_WORK_GROUP_SIZE(BLOCK_SPECTRA, BLOCK_CHANNELS, 1) void beamform(
    GLOBAL char2 * RESTRICT out,
    GLOBAL const char4 * RESTRICT in,
    GLOBAL const float2 * RESTRICT weights,
    GLOBAL const float * RESTRICT delays,
    int out_stride,
    int out_beam_stride,
    int out_frame_stride,
    int in_stride,
    int in_antenna_stride,
    int in_frame_stride,
    int n_ants,
    int n_channels,
    int n_times
)
{
    LOCAL_DECL float2 l_weights[BLOCK_CHANNELS][N_BEAMS];

    int l_time = get_local_id(0);
    int time = get_global_id(0);
    int l_channel = get_local_id(1);
    int channel = get_global_id(1);
    int frame = get_group_id(2);
    bool valid = (time < n_times && channel < n_channels);
    in += frame * in_frame_stride + channel * in_stride + time;
    out += frame * out_frame_stride + channel * out_stride + time;

    float2 accum[N_BEAMS];
    for (int i = 0; i < N_BEAMS; i++)
        accum[i] = make_float2(0.0f, 0.0f);

    for (int a = 0; a < n_ants; a++)
    {
        char4 sample = valid ? *in : make_char4(0, 0, 0, 0);
        float2 sample_pols[2] = {make_float2(sample.x, sample.y), make_float2(sample.z, sample.w)};
        in += in_antenna_stride;

        // Precompute the weights for all beams for this antenna
        for (int i = l_time; i < N_BEAMS; i += BLOCK_SPECTRA)
        {
            int addr = a * N_BEAMS + i;
            float2 w = weights[addr];
            float d = delays[addr] * channel;
            float2 rot;
            sincospif(d, &rot.y, &rot.x);
            w = cmul(w, rot);
            l_weights[l_channel][i] = w;
        }
        BARRIER();

% for i, pol in enumerate(beam_pols):
        {
            int i = ${i};
            int pol = ${pol};
            accum[i] = cmad(l_weights[l_channel][i], sample_pols[pol], accum[i]);
        }
% endfor
        BARRIER();  // protect against next loop iteration overwriting l_weights
    }

    if (!valid)
        return;
    for (int i = 0; i < N_BEAMS; i++)
    {
        int re, im;
        // TODO: report saturation flags somewhere
        quant(accum[i].x, &re);
        quant(accum[i].y, &im);
        out[out_beam_stride * i] = make_char2(re, im);
    }
}
