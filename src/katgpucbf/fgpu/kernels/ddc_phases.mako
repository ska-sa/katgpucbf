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

/* TODO:
 * - specify alignment requirements
 * - test reading via shared memory (possibly with memcpy_async)
 * - test writing via shared memory
 * - test putting constants into shared memory
 * - look into __constant memory type
 * - Try cache hints e.g. __ldcg / __ldcs / __stcs
 * - Try packing multiple logical workgroups into one physical one e.g. to
 *   enable warp-synchronous operation without harming occupancy
 * - Check if PAD_TILE padding is optimal
 * - Check if it could be more official to load SG_SIZE strided values per
 *   tile, instead of contiguous

/* Alignment requirements: TODO complete this
 * - in_offset must be a multiple of SEGMENT_SAMPLES
 * - All the tuning parameters must be powers of 2 (probably not strictly required, but
 *   ensures that any parameter is either a multiple of or fraction of any other).
 *   TODO: don't think WGS needs to be
 * - WGS must be a multiple of SG_SIZE
 * - TAPS must be a multiple of DECIMATION
 * - SEGMENT_SAMPLES must be a multiple of SG_SIZE
 * - DECIMATION must be a multiple of SG_SIZE
 */

#define WGS ${wgs}
#define TAPS ${taps}
#define DECIMATION ${decimation}
#define COARSEN ${coarsen}
#define SG_SIZE ${sg_size}

// Bits per input sample. Defined for clarity, but can't easily be changed
// without breaking all the code
#define SAMPLE_BITS 10
// Number of contiguous 32-bit words to store in each segment
#define SEGMENT_WORDS (SAMPLE_BITS / 2)
// Number of samples in each segment
#define SEGMENT_SAMPLES (SEGMENT_WORDS * 32 / SAMPLE_BITS)
// Number of output samples
#define GROUP_OUT_SIZE (COARSEN * (WGS / SG_SIZE))
// Stride of input samples between workgroups
#define GROUP_IN_SIZE (GROUP_OUT_SIZE * DECIMATION)
// Number of input samples to load in this workgroup
#define LOAD_SIZE (GROUP_IN_SIZE + TAPS - DECIMATION)
// Load-size expressed in 32-bit words (rounding up)
#define LOAD_WORDS ((LOAD_SIZE - 1) * SAMPLE_BITS / 32 + 1)
// Number of segments to load per work-item
#define SEGMENTS ((LOAD_SIZE - 1) / (SEGMENT_SAMPLES * WGS) + 1)
/* Number of contiguous samples that take turns occupying a tile
 * (must divide both SEGMENT_SAMPLES and DECIMATION, and be a multiple
 * of SG_SIZE.
 */
#define TILE_SAMPLES (SEGMENT_SAMPLES < DECIMATION ? SEGMENT_SAMPLES : DECIMATION)
// Number of tiles to store in local memory, prior to padding
#define TILES (LOAD_SIZE / TILE_SAMPLES)
#define TILES_PER_SEGMENT (SEGMENT_SAMPLES / TILE_SAMPLES)
#define TILES_PER_DECIMATION (DECIMATION / TILE_SAMPLES)

#define PAD_TILE_SCALE (COARSEN * TILES_PER_DECIMATION)
#define PAD_TILE(idx) ((idx) + (idx) / PAD_TILE_SCALE)
#define PADDED_TILES (PAD_TILE(TILES - 1) + 1)

${wg_reduce.define_scratch('float', sg_size, 'scratch_t', allow_shuffle=True)}
${wg_reduce.define_function('float', sg_size, 'reduce', 'scratch_t', wg_reduce.op_plus, allow_shuffle=True, broadcast=False)}

DEVICE_FN static float2 cmul(float2 a, float2 b)
{
    return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

DEVICE_FN static unsigned int reverse_endian(unsigned int v)
{
    return __byte_perm(v, v, 0x0123);
}

DEVICE_FN static unsigned int pad_tile(unsigned int idx)
{
    return PAD_TILE(idx);
}

/* A segment consists of SEGMENT_SAMPLES contiguous samples, still in 10-bit
 * packing and so occupying SEGMENT_WORDS uint32's. Each uint32 is loaded from
 * memory as big-endian.
 */
struct segment
{
    unsigned int raw[SEGMENT_WORDS];
};

/* A tile represents a contiguous group of TILE_COLUMNS samples, of which
 * only SG_SIZE (contiguous) are loaded in memory at a time.
 */
struct tile
{
    float2 samples[SG_SIZE];
};

DEVICE_FN static int segment_get(const segment *seg, int idx)
{
    int word0 = idx * SAMPLE_BITS / 32;
    int shift = idx * SAMPLE_BITS % 32;
    int top;  // packed value shifted to the top of a 32-bit word
    if (shift + SAMPLE_BITS <= 32)
        top = seg->raw[word0] << shift;
    else
        top = __funnelshift_l(seg->raw[word0 + 1], seg->raw[word0], shift);
    return top >>= 32 - SAMPLE_BITS;  // trusts nvcc to sign extend - undefined in C++
}

DEVICE_FN static void load_segments(
    const GLOBAL unsigned int * RESTRICT in,
    segment segs[SEGMENTS],
    int lid)
{
    for (int i = 0; i < SEGMENTS; i++)
        for (int j = 0; j < SEGMENT_WORDS; j++)
        {
            int addr = i * WGS * SEGMENT_WORDS + lid * SEGMENT_WORDS + j;
            // TODO: Could also use this check to avoid need for padding `in`
            if (addr < LOAD_WORDS)
                segs[i].raw[j] = in[addr];
        }

    /* First schedule all the loads so that they can happen asynchronously,
     * and only then do endian swapping.
     */
    for (int i = 0; i < SEGMENTS; i++)
        for (int j = 0; j < SEGMENT_WORDS; j++)
            if (i * WGS * SEGMENT_WORDS + j < LOAD_WORDS)
                segs[i].raw[j] = reverse_endian(segs[i].raw[j]);
}

KERNEL REQD_WORK_GROUP_SIZE(WGS, 1, 1) void ddc(
    GLOBAL float2 * RESTRICT out,
    const GLOBAL unsigned int * RESTRICT in,
    const GLOBAL float * RESTRICT weights,
    int out_offset,
    int in_offset,  // in samples (TODO: make it words to simplify?)
    int out_size,
    int in_size,
    float mix_scale,
    float mix_bias,  // TODO: fold into mix_lookup?
    const GLOBAL float2 (* RESTRICT mix_lookup)[SEGMENT_SAMPLES])
{
    LOCAL_DECL tile tiles[PADDED_TILES];
    segment segs[SEGMENTS];
    float2 sums[COARSEN];

    unsigned int lid = get_local_id(0);
    unsigned int group = get_group_id(0);
    unsigned int sg_rank = lid % SG_SIZE;
    unsigned int sg_group = lid / SG_SIZE;
    in_offset += group * GROUP_IN_SIZE;
    in += in_offset / SEGMENT_SAMPLES * SEGMENT_WORDS;
    out += out_offset + group * GROUP_OUT_SIZE;
    // Adjust mix_bias to the first segment handled by the thread
    // (TODO: could incorporate GROUP_IN_SIZE in mix_scale)
    mix_bias += (group * GROUP_IN_SIZE + lid * SEGMENT_SAMPLES) * mix_scale;
    float2 mix_base;
    sincospif(mix_bias, &mix_base.y, &mix_base.x);

    load_segments(in, segs, lid);

    for (int i = 0; i < COARSEN; i++)
        sums[i] = make_float2(0.0f, 0.0f);

#pragma unroll
    for (int phase = 0; phase < TILE_SAMPLES; phase += SG_SIZE)
    {
        /* Decode and mix the samples for the phase */
#pragma unroll
        for (int i = 0; i < SEGMENTS; i++)
            for (int j = 0; j < TILES_PER_SEGMENT; j++)
            {
                // TODO: can optimise this calculation with some constant folding?
                unsigned int tile_id = i * WGS * TILES_PER_SEGMENT + lid * TILES_PER_SEGMENT + j;
                unsigned int padded_tile_id = pad_tile(tile_id);
                if (padded_tile_id < PADDED_TILES)
                {
#pragma unroll
                    for (int k = 0; k < SG_SIZE; k++)
                    {
                        int seg_idx = j * TILE_SAMPLES + phase + k;
                        float sample = segment_get(&segs[i], seg_idx);
                        float2 mixed = cmul(mix_base, mix_lookup[i][seg_idx]);
                        mixed.x *= sample;
                        mixed.y *= sample;
                        tiles[padded_tile_id].samples[k] = mixed;
                    }
                }
            }

        // tiles is written above and read below
        BARRIER();

        /* Apply the filter */
#pragma unroll
        for (int dphase = 0; dphase < DECIMATION; dphase += TILE_SAMPLES)
        {
            // Sample within the decimation group for the first work item in
            // the subgroup
            unsigned int total_phase = phase + dphase;
            float2 samples[COARSEN];
            // TODO: can constant fold some of this
            int tile_index_base = pad_tile(
                total_phase / TILE_SAMPLES + sg_group * (COARSEN * TILES_PER_DECIMATION)
            );
#pragma unroll
            for (int j = 0; j < COARSEN - 1; j++)
            {
                // Prime the pipeline
                samples[j] = tiles[tile_index_base + pad_tile(j * TILES_PER_DECIMATION)].samples[sg_rank];
            }
#pragma unroll
            for (int i = 0; i < TAPS / DECIMATION; i++)
            {
                int tap = i * DECIMATION + total_phase + sg_rank;
                float w = weights[tap];
                samples[COARSEN - 1] = tiles[tile_index_base + pad_tile((i + COARSEN - 1) * TILES_PER_DECIMATION)].samples[sg_rank];
                for (int j = 0; j < COARSEN; j++)
                {
                    sums[j].x += w * samples[j].x;
                    sums[j].y += w * samples[j].y;
                }
                /* Shift down all the samples. The compiler should make
                 * these free by the magic of loop unrolling.
                 */
                for (int j = 0; j < COARSEN - 1; j++)
                    samples[j] = samples[j + 1];
            }

            // tiles is read above and written by the next loop iteration
            // (TODO: could be eliminated on the final loop pass)
        }
        BARRIER();
    }

    // Reduce the result across work items that are contributing to the same sum.
    for (int j = 0; j < COARSEN; j++)
    {
        LOCAL_DECL scratch_t scratch;
        sums[j].x = reduce(sums[j].x, sg_rank, &scratch);
        sums[j].y = reduce(sums[j].y, sg_rank, &scratch);
        if (sg_rank == 0)
            out[sg_group * COARSEN + j] = sums[j];
    }
}
