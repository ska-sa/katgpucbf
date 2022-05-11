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
 * - test reading via shared memory with memcpy_async
 * - test writing via shared memory
 * - test putting constants into shared memory
 * - look into __constant memory type
 * - Try packing multiple logical workgroups into one physical one e.g. to
 *   enable warp-synchronous operation without harming occupancy
 * - Check if it could be more efficient to load SG_SIZE strided values per
 *   tile, instead of contiguous
 */

/* Alignment requirements:
 * - in_offset must be a multiple of SEGMENT_SAMPLES
 * - GROUP_IN_SIZE must be a multiple of SEGMENT_SAMPLES
 * - SEGMENT_SAMPLES * SAMPLE_BITS must be a multiple of 32
 * - WGS must be a multiple of SG_SIZE
 * - TAPS must be a multiple of DECIMATION
 * - SEGMENT_SAMPLES must be a multiple of SG_SIZE
 * - DECIMATION must be a multiple of SG_SIZE
 * - COARSEN should be odd for best performance
 * - SG_SIZE should be a power of two for best performance
 */

#define WGS ${wgs}
#define TAPS ${taps}
#define DECIMATION ${decimation}
#define COARSEN ${coarsen}
#define SG_SIZE ${sg_size}
#define SAMPLE_BITS ${sample_bits}
#define SEGMENT_SAMPLES ${segment_samples}

// Number of contiguous 32-bit words to store in each segment
#define SEGMENT_WORDS (SEGMENT_SAMPLES * SAMPLE_BITS / 32)
/* Number of contiguous 32-bit words to store in each segment. This must
 * correspond to a whole number of samples (and hence needs to be adjusted
 * if SAMPLE_BITS changes).
 */
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
 * of SG_SIZE). This implementation is pessimistic when DECIMATION is
 * neither a factor nor multiple of SEGMENT_SAMPLES, but that's not expected to
 * be a common case.
 */
#if DECIMATION % SEGMENT_SAMPLES == 0
# define TILE_SAMPLES SEGMENT_SAMPLES
#elif SEGMENT_SAMPLES % DECIMATION == 0
# define TILE_SAMPLES DECIMATION
#else
# define TILE_SAMPLES SG_SIZE
#endif
// Number of tiles to store in local memory
#define TILES (LOAD_SIZE / TILE_SAMPLES)
#define TILES_PER_SEGMENT (SEGMENT_SAMPLES / TILE_SAMPLES)
#define TILES_PER_DECIMATION (DECIMATION / TILE_SAMPLES)

${wg_reduce.define_scratch('float', sg_size, 'scratch_t', allow_shuffle=True)}
${wg_reduce.define_function('float', sg_size, 'reduce', 'scratch_t', wg_reduce.op_plus, allow_shuffle=True, broadcast=False)}

DEVICE_FN void sync()
{
    // When only a single warp is in use, we can use a cheaper barrier
#if defined(__CUDA_ARCH__)
    if (WGS == warpSize)
    {
        __syncwarp();
    }
    else
#endif
    {
        BARRIER();
    }
}

DEVICE_FN static float2 cmul(float2 a, float2 b)
{
    return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

DEVICE_FN static unsigned int reverse_endian(unsigned int v)
{
    return __byte_perm(v, v, 0x0123);
}

/* A segment consists of SEGMENT_SAMPLES contiguous samples, still in 10-bit
 * packing and so occupying SEGMENT_WORDS uint32's. Each uint32 is loaded from
 * memory as big-endian.
 */
struct segment
{
    unsigned int raw[SEGMENT_WORDS];
};

/* A tile represents a contiguous group of TILE_SAMPLES samples, of which
 * only SG_SIZE (contiguous) are loaded in memory at a time.
 */
struct tile
{
    float2 samples[SG_SIZE];
};

/* Retrieve a sample from a segment. */
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

/* Retrieve the data for all segments from global memory into registers. */
DEVICE_FN static void load_segments(
    const GLOBAL unsigned int * RESTRICT in,
    segment segs[SEGMENTS],
    int lid,
    int in_size_words)
{
    for (int i = 0; i < SEGMENTS; i++)
        for (int j = 0; j < SEGMENT_WORDS; j++)
        {
            int addr = i * WGS * SEGMENT_WORDS + lid * SEGMENT_WORDS + j;
            segs[i].raw[j] = (addr < in_size_words) ? in[addr] : 0;
        }

    /* First schedule all the loads so that they can happen asynchronously,
     * and only then do endian swapping.
     *
     * Note: this will do some unnecessary work (some of the values are
     * outside LOAD_SAMPLES) but it's cheaper to just do it than to check.
     */
    for (int i = 0; i < SEGMENTS; i++)
        for (int j = 0; j < SEGMENT_WORDS; j++)
            segs[i].raw[j] = reverse_endian(segs[i].raw[j]);
}

/* Decode and mix the samples for the phase. Specifically, each tile is
 * populated with the samples with positions in the range
 * [phase, phase + SG_SIZE).
 */
DEVICE_FN static void mix(
    const segment segs[SEGMENTS],
    LOCAL tile tiles[TILES],
    float2 mix_base,
    const GLOBAL float2 (* RESTRICT mix_lookup)[SEGMENT_SAMPLES],
    int phase,
    int lid)
{
#pragma unroll
    for (int i = 0; i < SEGMENTS; i++)
        for (int j = 0; j < TILES_PER_SEGMENT; j++)
        {
            int tile_id = i * WGS * TILES_PER_SEGMENT + lid * TILES_PER_SEGMENT + j;
            if (tile_id < TILES)
            {
#pragma unroll
                for (int k = 0; k < SG_SIZE; k++)
                {
                    int seg_idx = j * TILE_SAMPLES + phase + k;
                    float sample = segment_get(&segs[i], seg_idx);
                    float2 mixed = cmul(mix_base, mix_lookup[i][seg_idx]);
                    mixed.x *= sample;
                    mixed.y *= sample;
                    tiles[tile_id].samples[k] = mixed;
                }
            }
        }
}

/* Multiply samples in `tiles` with `weights` and accumulate in `sums`.
 */
DEVICE_FN static void filter(
    const GLOBAL float * RESTRICT weights,
    const LOCAL tile tiles[TILES],
    float2 sums[COARSEN],
    int phase, int sg_group, int sg_rank)
{
#pragma unroll
    for (int dphase = 0; dphase < DECIMATION; dphase += TILE_SAMPLES)
    {
        // Sample within the decimation group for the first work item in
        // the subgroup
        int total_phase = phase + dphase;
        // Holds a sliding window of samples which get multiplied by the
        // sample weights.
        float2 samples[COARSEN];
        int tile_index_base = total_phase / TILE_SAMPLES + sg_group * (COARSEN * TILES_PER_DECIMATION);
#pragma unroll
        for (int j = 0; j < COARSEN - 1; j++)
        {
            // Prime the pipeline
            samples[j] = tiles[tile_index_base + j * TILES_PER_DECIMATION].samples[sg_rank];
        }
#pragma unroll
        for (int i = 0; i < TAPS / DECIMATION; i++)
        {
            int tap = i * DECIMATION + total_phase + sg_rank;
            float w = weights[tap];
            samples[COARSEN - 1] = tiles[tile_index_base + (i + COARSEN - 1) * TILES_PER_DECIMATION].samples[sg_rank];
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
    }
}

KERNEL REQD_WORK_GROUP_SIZE(WGS, 1, 1) void ddc(
    GLOBAL float2 * RESTRICT out,
    const GLOBAL unsigned int * RESTRICT in,
    const GLOBAL float * RESTRICT weights,
    int out_offset,
    int in_offset_words,
    int out_size,
    int in_size_words,
    double mix_scale,  // Mixer frequency in cycles per sample
    double mix_bias,   // Mixer phase in cycles at the first sample
    const GLOBAL float2 (* RESTRICT mix_lookup)[SEGMENT_SAMPLES])
{
    LOCAL_DECL tile tiles[TILES];
    segment segs[SEGMENTS];
    float2 sums[COARSEN];

    /* Note: unsigned is important here as it allows the compiler to turn
     * div/mod into shift/mask when SG_SIZE is a power of 2.
     */
    unsigned int lid = get_local_id(0);
    unsigned int group = get_group_id(0);
    unsigned int sg_rank = lid % SG_SIZE;   // Position within subgroup
    unsigned int sg_group = lid / SG_SIZE;  // Subgroup number

    unsigned int in_offset_group = group * (GROUP_IN_SIZE * SEGMENT_WORDS / SEGMENT_SAMPLES);
    in_offset_words += in_offset_group;
    in += in_offset_words;
    in_size_words -= in_offset_group;
    /* Note: could also limit in_size_words to LOAD_WORDS to avoid loading
     * unwanted data. But that data will be needed by another workgroup and
     * it seems beneficial to load it into L2 cache.
     */

    out += out_offset + group * GROUP_OUT_SIZE;
    out_size -= group * GROUP_OUT_SIZE;

    // Complex mixer value at first sample mixed by this work item
    float2 mix_base;
    /* mix_bias needs to be computed at double precision because there are many
     * bits to the left of the decimal point. After we've gotten rid of those
     * we can go back to single precision.
     */
    mix_bias += (group * GROUP_IN_SIZE + lid * SEGMENT_SAMPLES) * mix_scale;
    mix_bias -= rint(mix_bias);
    sincospif(2 * (float) mix_bias, &mix_base.y, &mix_base.x);

    load_segments(in, segs, lid, in_size_words);

    for (int i = 0; i < COARSEN; i++)
        sums[i] = make_float2(0.0f, 0.0f);

    /* This loop determines which mixed samples are kept in `tiles`, namely
     * those in positions [phase, phase + SG_SIZE) within each tile.
     */
#pragma unroll
    for (int phase = 0; phase < TILE_SAMPLES; phase += SG_SIZE)
    {
        mix(segs, tiles, mix_base, mix_lookup, phase, lid);

        // tiles is written above and read below
        sync();

        filter(weights, tiles, sums, phase, sg_group, sg_rank);

        // tiles is read above and written by the next loop iteration
        // (could possibly be eliminated on the final loop pass)
        sync();
    }

    // Reduce the result across work items that are contributing to the same sum.
    for (int j = 0; j < COARSEN; j++)
    {
        LOCAL_DECL scratch_t scratch;
        sums[j].x = reduce(sums[j].x, sg_rank, &scratch);
        sums[j].y = reduce(sums[j].y, sg_rank, &scratch);
        if (sg_rank == 0)
        {
            int addr = sg_group * COARSEN + j;
            if (addr < out_size)
                out[addr] = sums[j];
        }
    }
}
