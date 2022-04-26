/*******************************************************************************
 * Copyright (c) 2022, National Research Foundation (SARAO)
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
<%include file="unpack_10bit.mako"/>

/* Alignment requirements:
 * - DECIMATION must divide into TAPS
 * - GROUP_OUT_SIZE must be a multiple of COARSEN * WGS
 */

#define WGS ${wgs}
#define TAPS ${taps}
#define DECIMATION ${decimation}
#define GROUP_OUT_SIZE ${group_out_size}
#define GROUP_IN_SIZE (GROUP_OUT_SIZE * DECIMATION)
#define PADDED_SHARED_SIZE (GROUP_IN_SIZE + TAPS - DECIMATION)
#define COARSEN ${coarsen}

KERNEL REQD_WORK_GROUP_SIZE(WGS, 1, 1) void ddc(
    GLOBAL float2 * RESTRICT out,
    const GLOBAL uchar * RESTRICT in,
    const GLOBAL float * RESTRICT weights,
    int out_offset,
    int in_offset,
    int out_size,
    int in_size,
    float mix_scale,
    float mix_bias)
{
    int lid = get_local_id(0);
    int group = get_group_id(0);
    in_offset += group * GROUP_IN_SIZE;
    out_offset += group * GROUP_OUT_SIZE;
    out += out_offset;
    out_size -= out_offset;
    mix_bias += group * GROUP_IN_SIZE * mix_scale; // TODO: does this lose precision?

    // Load coefficients
    LOCAL_DECL float l_weights[TAPS];
    for (int i = lid; i < TAPS; i += WGS)
        l_weights[i] = weights[i];

    // Load, decode and mix input data
    LOCAL_DECL float2 samples[PADDED_SHARED_SIZE + PADDED_SHARED_SIZE / DECIMATION];
    for (int i = lid; i < PADDED_SHARED_SIZE; i += WGS)
    {
        int idx = in_offset + i;
        float orig = (idx < in_size) ? get_sample_10bit(in, in_offset + i) : 0.0f;
        float phase = i * mix_scale + mix_bias;
        float2 mix;
        sincospif(phase, &mix.y, &mix.x);
        samples[i + (unsigned) i / DECIMATION] = make_float2(mix.x * orig, mix.y * orig);
    }

    BARRIER();

    for (int i = COARSEN * lid; i < GROUP_OUT_SIZE; i += COARSEN * WGS)
    {
        float2 sums[COARSEN];
        for (int k = 0; k < COARSEN; k++)
            sums[k] = make_float2(0.0f, 0.0f);
        int start = i * (DECIMATION + 1);
        for (int j = 0; j < TAPS; j++)
        {
            float c = l_weights[j];
            for (int k = 0; k < COARSEN; k++)
            {
                float2 v = samples[start + k * (DECIMATION + 1) + j + j / DECIMATION];
                sums[k].x += c * v.x;
                sums[k].y += c * v.y;
            }
        }
        for (int k = 0; k < COARSEN; k++)
        {
            int idx = i + k;
            if (idx < out_size)
                out[idx] = sums[k];
        }
    }
}
