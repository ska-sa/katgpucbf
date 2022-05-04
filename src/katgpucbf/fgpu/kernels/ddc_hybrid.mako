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
 * - COARSEN * DECIMATION must be a multiple of 16
 */

#define WGS ${wgs}
#define TAPS ${taps}
#define DECIMATION ${decimation}
#define GROUP_OUT_SIZE (COARSEN * (WGS / SG_SIZE))
#define GROUP_IN_SIZE (GROUP_OUT_SIZE * DECIMATION)
#define LOAD_SIZE (GROUP_IN_SIZE + TAPS - DECIMATION)
#define COARSEN ${coarsen}
#define SG_SIZE ${sg_size}

#define REORDER_READ 0
#define REORDER_WRITE 0

// TODO: adapt to COARSEN and SG_SIZE
#define PAD_ADDR(x) ((x) + ((x) >> 5))
#define SAMPLE_ADDR(x) (((x) >> 2) + ((x) >> 4))  // * 10 / 32, but without overflow (must be a multiple of 16 though)
#define PADDED_LOAD_SIZE (PAD_ADDR(LOAD_SIZE - 1) + 1)

DEVICE_FN static unsigned int pad_addr(unsigned int addr)
{
    return PAD_ADDR(addr);
}

${wg_reduce.define_scratch('float', sg_size, 'scratch_t', allow_shuffle=True)}
${wg_reduce.define_function('float', sg_size, 'reduce', 'scratch_t', wg_reduce.op_plus, allow_shuffle=True, broadcast=False)}

DEVICE_FN static float2 cmul(float2 a, float2 b)
{
    return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

typedef union
{
    struct
    {
        union
        {
            float weights[TAPS];
#if REORDER_READ
            unsigned int raw_samples[WGS * 5];
#endif
        };
        float2 samples[PADDED_LOAD_SIZE];
    };
#if REORDER_WRITE
    float2 out[GROUP_OUT_SIZE];
#endif
} local_t;

KERNEL REQD_WORK_GROUP_SIZE(WGS, 1, 1) void ddc(
    GLOBAL float2 * RESTRICT out,
    const GLOBAL unsigned int * RESTRICT in,
    const GLOBAL float * RESTRICT weights,
    int out_offset,
    int in_offset,
    int out_size,
    int in_size,
    float mix_scale,
    float mix_bias,
    float2 mix_step)
{
    LOCAL_DECL local_t local_data;

    int lid = get_local_id(0);
    int group = get_group_id(0);
    in_offset += group * GROUP_IN_SIZE;
    out_offset += group * GROUP_OUT_SIZE;
    out += out_offset;
    out_size -= out_offset;
    mix_bias += group * GROUP_IN_SIZE * mix_scale; // TODO: does this lose precision?

    /* Load, decode and mix input data. Each work item decodes and mixes
     * 16 consecutive samples (20 bytes) at a time. These samples are first
     * loaded collaboratively to shared memory to improve the access
     * patterns.
     */
    // TODO: this doesn't handle non-round in_offset, which is necessary for coarse delay?
    // TODO: pad the array to avoid out-of-bounds accesses
    int load_addr = SAMPLE_ADDR(in_offset);
    int padded_lid = pad_addr(lid * 16);
#pragma unroll
    for (int i = 0; i < SAMPLE_ADDR(LOAD_SIZE); i += WGS * 5)
    {
        // TODO: could use memcpy_async for this
        // TODO: can this be restructured so that synchronisation is only
        // needed on the warp level?
        unsigned int raw[5];
#if REORDER_READ
        for (int j = 0; j < 5; j++)
        {
            if (i + j * WGS < SAMPLE_ADDR(LOAD_SIZE))
                raw[j] = in[load_addr + lid + i + j * WGS];
        }
        for (int j = 0; j < 5; j++)
        {
            if (i + j * WGS < SAMPLE_ADDR(LOAD_SIZE))
                local_data.raw_samples[lid + j * WGS] = raw[j];
        }
        BARRIER();

        for (int j = 0; j < 5; j++)
            raw[j] = local_data.raw_samples[j + 5 * lid];
#else
        for (int j = 0; j < 5; j++)
            raw[j] = in[load_addr + 5 * lid + i + j];
#endif
        // CUDA is little endian but the bits are packed in big endian
        for (int j = 0; j < 5; j++)
            raw[j] = __byte_perm(raw[j], raw[j], 0x0123);

        int i_samples = i / 5 * 16;
        int first_idx = i_samples + lid * 16;
        if (first_idx < LOAD_SIZE)
        {
            float first_phase = first_idx * mix_scale + mix_bias;
            float2 mix;
            sincospif(first_phase, &mix.y, &mix.x);
#pragma unroll
            for (int j = 0; j < 16; j++)
            {
                int word0 = j * 10 / 32;
                int shift = j * 10 % 32;
                int top;  // 10-bit value shifted to the top of a 32-bit word
                if (shift + 10 <= 32)
                    top = raw[word0] << shift;
                else
                    top = __funnelshift_l(raw[word0 + 1], raw[word0], shift);
                top >>= 22;  // trusts nvcc to sign extend - undefined in C++
                float orig = top;

                int idx_padded = pad_addr(i_samples + j) + padded_lid;
                if (idx_padded < PADDED_LOAD_SIZE)
                    local_data.samples[idx_padded] = make_float2(mix.x * orig, mix.y * orig);
                mix = cmul(mix, mix_step);
            }
        }

#if REORDER_READ
        BARRIER();
#endif
    }

    // Load coefficients
    for (int i = lid; i < TAPS; i += WGS)
        local_data.weights[i] = weights[i];

    int sgid = (unsigned int) lid / SG_SIZE;
    int sgpos = (unsigned int) lid % SG_SIZE;

    BARRIER();

    // This subgroup will compute COARSEN consecutive outputs starting from
    // sgid * COARSEN.
    float2 sums[COARSEN];
    for (int j = 0; j < COARSEN; j++)
        sums[j] = make_float2(0.0f, 0.0f);
    // Input sample index to start from for this work item
    int start = pad_addr((COARSEN * sgid) * DECIMATION) + sgpos;
#pragma unroll
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
#if !REORDER_WRITE
        if (sgpos == 0)
            out[sgid * COARSEN + j] = sums[j];
#endif
    }

#if REORDER_WRITE
    BARRIER();

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
            out[j + lid] = local_data.out[j + lid];
    }
    else
    {
        for (int j = lid; j < COARSEN * (WGS / SG_SIZE); j += WGS)
            out[j] = local_data.out[j];
    }
#endif
}
