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
#define BLOCK_TIME ${block_time}
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

// out: shape beam, channel, time
// in: shape antenna, channel, time, pol
// weight: shape antenna, beam, tightly packed
// delays: shape antenna, beam, tightly packed
// Each thread computes all beams for one (channel, time)
KERNEL REQD_WORK_GROUP_SIZE(BLOCK_TIME, BLOCK_CHANNELS, 1) void beamform(
    GLOBAL char2 * RESTRICT out,
    GLOBAL const char4 * RESTRICT in,
    GLOBAL const float2 * RESTRICT weights,
    GLOBAL const float * RESTRICT delays,
    int out_stride,
    int out_beam_stride,
    int in_stride,
    int in_antenna_stride,
    int n_antennas,
    int n_channels,
    int n_times
)
{
    int time = get_global_id(0);
    int channel = get_global_id(1);
    if (time >= n_times || channel >= n_channels)
        return;  // out-of-bounds
    int in_addr = channel * in_stride + time;
    float2 accum[N_BEAMS];
    for (int i = 0; i < N_BEAMS; i++)
        accum[i] = make_float2(0.0f, 0.0f);

    for (int a = 0; a < n_antennas; a++)
    {
        char4 sample = in[in_addr];
        float2 sample_pols[2] = {make_float2(sample.x, sample.y), make_float2(sample.z, sample.w)};
        in_addr += in_antenna_stride;
% for i, pol in enumerate(beam_pols):
        {
            int i = ${i};
            int pol = ${pol};

            // TODO: w is time-invariant. Compute it once
            // for all work-items with the same channel.
            int addr = a * N_BEAMS + i;
            float2 w = weights[addr];
            float d = delays[addr] * channel;
            float2 rot;
            sincospif(d, &rot.y, &rot.x);
            w = cmul(w, rot);

            accum[i] = cmad(w, sample_pols[pol], accum[i]);
        }
% endfor
    }
    for (int i = 0; i < N_BEAMS; i++)
    {
        int re, im;
        // TODO: report saturation flags somewhere
        quant(accum[i].x, &re);
        quant(accum[i].y, &im);
        out[out_beam_stride * i + out_stride * channel + time] = make_char2(re, im);
    }
}
