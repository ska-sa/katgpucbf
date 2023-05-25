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

/* Alignment requirements:
 * - WGS must be multiple of SAMPLE_WORD_BITS
 * - TAPS must be a multiple of SUBSAMPLING
 */

<%include file="/port.mako"/>

#define WGS ${wgs}
#define TAPS ${taps}
#define SUBSAMPLING ${subsampling}
#define SAMPLE_BITS ${sample_bits}
#define C ${unroll}
#define W ${taps // subsampling}

/// Raw storage type for sample data
typedef unsigned int sample_word;
/// Signed version of sample_word
typedef int ssample_word;
#define SAMPLE_WORD_BITS 32

/**
 *  Read a contiguous sequence of packed samples
 *
 * The implementation is optimised to work with loop unrolling, such that
 * buffer_bits should always be a compile-time constant and branches should
 * vanish.
 */
struct decoder
{
    const LOCAL sample_word *next_raw;
    sample_word buffer;
    int buffer_bits;
};

DEVICE_FN static unsigned int reverse_endian(unsigned int v)
{
    return __byte_perm(v, v, 0x0123);
}

/// Initialise a decoder.
DEVICE_FN static void decoder_init(decoder *dec, const LOCAL sample_word *base, unsigned int offset)
{
    // Compute how much to skip
    unsigned int bits = offset * SAMPLE_BITS;
    unsigned int full_words = bits / SAMPLE_WORD_BITS;
    dec->buffer = reverse_endian(base[full_words]);
    dec->buffer_bits = SAMPLE_WORD_BITS - bits % SAMPLE_WORD_BITS;
    dec->next_raw = base + (full_words + 1);
}

DEVICE_FN static int decoder_next(decoder *dec)
{
    // Desired value in the top SAMPLE_BITS bits
    sample_word shifted;
    if (dec->buffer_bits >= SAMPLE_BITS)
    {
        shifted = dec->buffer << (SAMPLE_WORD_BITS - dec->buffer_bits);
        dec->buffer_bits -= SAMPLE_BITS;
    }
    else
    {
        sample_word next = *dec->next_raw++;
        // CUDA is little-endian, but the packing uses big endian
        next = reverse_endian(next);
        shifted = __funnelshift_lc(next, dec->buffer, SAMPLE_WORD_BITS - dec->buffer_bits);
        dec->buffer = next;
        dec->buffer_bits += SAMPLE_WORD_BITS - SAMPLE_BITS;
    }
    // Rely on nvcc to do sign extension when right-shifting a negative
    // value (it's undefined behaviour in C).
    return ((ssample_word) shifted) >> (SAMPLE_WORD_BITS - SAMPLE_BITS);
}

DEVICE_FN static float2 cmul(float2 a, float2 b)
{
    return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

KERNEL REQD_WORK_GROUP_SIZE(WGS, 1, 1) void ddc(
    GLOBAL float2 * RESTRICT out,
    const GLOBAL sample_word * RESTRICT in,
    const GLOBAL float2 * RESTRICT weights,
    unsigned int out_size,
    unsigned int in_size_words,
    double mix_scale,  // Mixer frequency in cycles per sample
    double mix_bias,   // Mixer phase in cycles at the first sample
    const GLOBAL float2 * RESTRICT mix_lookup  // Mixer phase rotations
)
{
    const int group_in_size = TAPS + (WGS * C - 1) * SUBSAMPLING;
    const int group_in_words = (group_in_size * SAMPLE_BITS + SAMPLE_WORD_BITS - 1) / SAMPLE_WORD_BITS;
    LOCAL_DECL union
    {
        struct
        {
            sample_word l_in[group_in_words];
            float2 l_weights[TAPS];
            float2 l_mix_lookup[C];
        };
        float l_out[2][C * WGS];
    } local;

    unsigned int lid = get_local_id(0);
    /* Copy workgroup's sample data to local memory */
    unsigned int group_first_in_idx = get_group_id(0) * (WGS * C * SUBSAMPLING);
    // TODO: risk of integer overflow:
    unsigned int group_first_in_word = group_first_in_idx * SAMPLE_BITS / SAMPLE_WORD_BITS;
    for (int i = lid; i < group_in_words; i += WGS)
    {
        unsigned int idx = group_first_in_word + i;
        local.l_in[i] = (idx < in_size_words) ? in[group_first_in_word + i] : 0;
    }

    /* Copy weights and mix_lookup to local memory (TODO: bank conflicts?) */
    for (int i = lid; i < TAPS; i += WGS)
        local.l_weights[i] = weights[i];
    for (int i = lid; i < C; i += WGS)
        local.l_mix_lookup[i] = mix_lookup[i];

    BARRIER();

    float2 accum[C];
    decoder decoders[C + W - 1];
    float samples[C + W - 1];

    unsigned int first_in_idx = lid * (C * SUBSAMPLING);
    for (int i = 0; i < C + W - 1; i++)
        decoder_init(&decoders[i], local.l_in, first_in_idx + i * SUBSAMPLING);
    for (int i = 0; i < C; i++)
        accum[i] = make_float2(0.0f, 0.0f);

    for (int i = 0; i < SUBSAMPLING; i++)
    {
        for (int j = 0; j < C + W - 1; j++)
        {
            samples[j] = (float) decoder_next(&decoders[j]);
        }
        for (int j = 0; j < W; j++)
        {
            float2 w = local.l_weights[j * SUBSAMPLING + i];
            for (int k = 0; k < C; k++)
            {
                accum[k].x += samples[j + k] * w.x;
                accum[k].y += samples[j + k] * w.y;
            }
        }
    }

    mix_bias += (group_first_in_idx + first_in_idx) * mix_scale;
    mix_bias -= rint(mix_bias);
    float2 mix_base;
    sincospif(2 * (float) mix_bias, &mix_base.y, &mix_base.x);

    for (int i = 0; i < C; i++)
    {
        accum[i] = cmul(accum[i], mix_base);
        if (i > 0)
            accum[i] = cmul(accum[i], local.l_mix_lookup[i]);
    }

    BARRIER();

    for (int i = 0; i < C; i++)
    {
        unsigned int idx = lid * C + i;
        local.l_out[0][idx] = accum[i].x;
        local.l_out[1][idx] = accum[i].y;
    }

    BARRIER();

    unsigned int first_out_idx = get_group_id(0) * (WGS * C);
    for (int i = 0; i < C; i++)
    {
        unsigned int l_idx = lid + i * WGS;
        unsigned int idx = first_out_idx + l_idx;
        if (idx < out_size)
        {
            out[idx] = make_float2(local.l_out[0][l_idx], local.l_out[1][l_idx]);
        }
    }
}
