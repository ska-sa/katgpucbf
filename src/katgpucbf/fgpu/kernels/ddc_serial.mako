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
 */

#define WGS ${wgs}
#define TAPS ${taps}
#define DECIMATION ${decimation}
#define GROUP_OUT_SIZE ${group_out_size}
#define GROUP_IN_SIZE (GROUP_OUT_SIZE * DECIMATION)
#define PADDED_SHARED_SIZE (GROUP_IN_SIZE + TAPS - DECIMATION)

KERNEL REQD_WORK_GROUP_SIZE(WGS, 1, 1) void ddc(
    GLOBAL float2 * RESTRICT out,
    const GLOBAL uchar * RESTRICT in,
    const GLOBAL float * RESTRICT weights,
    int out_offset,
    int in_offset,
    float mix_scale,
    float mix_bias)
{
    int lid = get_local_id(0);
    int group = get_group_id(0);
    in_offset += group * GROUP_IN_SIZE;
    out_offset += group * GROUP_OUT_SIZE;
    mix_bias += group * GROUP_IN_SIZE * mix_scale; // TODO: does this lose precision?

    // Load coefficients
    LOCAL_DECL float l_weights[TAPS];
    for (int i = lid; i < TAPS; i += WGS)
        l_weights[i] = weights[i];

    // Load, decode and mix input data
    LOCAL_DECL float2 samples[PADDED_SHARED_SIZE];
    for (int i = lid; i < PADDED_SHARED_SIZE; i += WGS)
    {
        float orig = get_sample_10bit(in, in_offset + i);
        float phase = i * mix_scale + mix_bias;
        float2 mix;
        sincospif(phase, &mix.y, &mix.x);
        samples[i] = make_float2(mix.x * orig, mix.y * orig);
    }

    BARRIER();

    for (int i = lid; i < GROUP_OUT_SIZE; i += WGS)
    {
        float2 sum = make_float2(0.0f, 0.0f);
        int start = lid * DECIMATION;
#pragma unroll
        for (int j = 0; j < TAPS; j++)
        {
            float c = l_weights[j];
            float2 v = samples[start + j];
            sum.x += c * v.x;
            sum.y += c * v.y;
        }
        out[out_offset + i] = sum;
    }
}
