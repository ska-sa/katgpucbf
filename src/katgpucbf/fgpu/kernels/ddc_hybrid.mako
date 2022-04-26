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
<%namespace name="wg_reduce" file="/wg_reduce.mako"/>

/* Alignment requirements:
 * - DECIMATION must divide into TAPS
 * - SG_SIZE must divide into WGS
 * - SG_SIZE must divide into DECIMATION
 * - SG_SIZE should ideally be a power of 2
 * - COARSEN * (WGS / SG_SIZE) must divide into GROUP_OUT_SIZE
 */

#define WGS ${wgs}
#define TAPS ${taps}
#define DECIMATION ${decimation}
#define GROUP_OUT_SIZE ${group_out_size}
#define GROUP_IN_SIZE (GROUP_OUT_SIZE * DECIMATION)
#define LOAD_SIZE (GROUP_IN_SIZE + TAPS - DECIMATION)
#define COARSEN ${coarsen}
#define SG_SIZE ${sg_size}

#define PAD_ADDR(x) ((x) + (x) / (COARSEN * DECIMATION) * SG_SIZE)

DEVICE_FN unsigned int pad_addr(unsigned int addr)
{
    return PAD_ADDR(addr);
}

${wg_reduce.define_scratch('float', sg_size, 'scratch_t', allow_shuffle=True)}
${wg_reduce.define_function('float', sg_size, 'reduce', 'scratch_t', wg_reduce.op_plus, allow_shuffle=True, broadcast=False)}

typedef union
{
    struct
    {
        float2 samples[PAD_ADDR(LOAD_SIZE)];
        float weights[TAPS];
    };
    float2 out[COARSEN * (WGS / SG_SIZE)];
} local_t;

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
    LOCAL_DECL local_t local_data;

    int lid = get_local_id(0);
    int group = get_group_id(0);
    in_offset += group * GROUP_IN_SIZE;
    out_offset += group * GROUP_OUT_SIZE;
    out += out_offset;
    out_size -= out_offset;
    mix_bias += group * GROUP_IN_SIZE * mix_scale; // TODO: does this lose precision?

    // Load coefficients
    for (int i = lid; i < TAPS; i += WGS)
        local_data.weights[i] = weights[i];

    // Load, decode and mix input data
    for (int i = lid; i < LOAD_SIZE; i += WGS)
    {
        int idx = in_offset + i;
        float orig = (idx < in_size) ? get_sample_10bit(in, in_offset + i) : 0.0f;
        float phase = i * mix_scale + mix_bias;
        float2 mix;
        sincospif(phase, &mix.y, &mix.x);
        local_data.samples[pad_addr(i)] = make_float2(mix.x * orig, mix.y * orig);
    }

    int sgid = (unsigned int) lid / SG_SIZE;
    int sgpos = (unsigned int) lid % SG_SIZE;
    for (int outer = 0; outer < GROUP_OUT_SIZE; outer += COARSEN * (WGS / SG_SIZE))
    {
        BARRIER();

        // This subgroup will compute COARSEN consecutive outputs starting from
        // outer + sgid * COARSEN.
        float2 sums[COARSEN];
        for (int j = 0; j < COARSEN; j++)
            sums[j] = make_float2(0.0f, 0.0f);
        // Input sample index to start from for this work item
        int start = pad_addr((outer + COARSEN * sgid) * DECIMATION) + sgpos;
        for (int phase = 0; phase < DECIMATION; phase += SG_SIZE)
        {
            float2 r_samples[COARSEN];
            for (int j = 0; j < COARSEN - 1; j++)
                r_samples[j] = local_data.samples[start + pad_addr(phase + j * DECIMATION)];
            for (int row = 0; row < TAPS; row += DECIMATION)
            {
                int i = row + phase;
                float w = local_data.weights[i + sgpos];
                r_samples[COARSEN - 1] = local_data.samples[start + pad_addr(i + (COARSEN - 1) * DECIMATION)];
                for (int j = 0; j < COARSEN; j++)
                {
                    sums[j].x += w * r_samples[j].x;
                    sums[j].y += w * r_samples[j].y;
                }
                /* Shift down all the samples. The compiler should make
                 * these free by the magic of loop unrolling.
                 */
                for (int j = 0; j < COARSEN - 1; j++)
                    r_samples[j] = r_samples[j + 1];
            }
        }
        for (int j = 0; j < COARSEN; j++)
        {
            LOCAL_DECL scratch_t scratch;
            sums[j].x = reduce(sums[j].x, sgpos, &scratch);
            sums[j].y = reduce(sums[j].y, sgpos, &scratch);
        }

        BARRIER();

        /* TODO: write directly to global memory if it won't cause
         * uncoalesced accesses.
         */
        if (sgpos == 0)
        {
            for (int j = 0; j < COARSEN; j++)
                local_data.out[sgid * COARSEN + j] = sums[j]; // TODO: can bank conflicts be avoided?
        }

        BARRIER();

        if (COARSEN % SG_SIZE == 0)
        {
            // All work items will do the same amount of work
            for (int j = 0; j < COARSEN * (WGS / SG_SIZE); j += WGS)
                out[outer + j + lid] = local_data.out[j + lid];
        }
        else
        {
            for (int j = lid; j < COARSEN * (WGS / SG_SIZE); j += WGS)
                out[outer + j] = local_data.out[j];
        }
    }
}
