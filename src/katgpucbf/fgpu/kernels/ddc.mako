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
 * - C * SUBSAMPLING * INPUT_SAMPLE_BITS must be a multiple of SAMPLE_WORD_BITS
 */

<%include file="/port.mako"/>

#define WGS ${wgs}
#define TAPS ${taps}
#define SUBSAMPLING ${subsampling}
#define INPUT_SAMPLE_BITS ${input_sample_bits}
#define C ${unroll}
#define W ${(taps + subsampling - 1) // subsampling}

<%include file="sample.mako"/>

/**
 * Load the next sample value.
 *
 * This is intended to be called with a sequence of contiguous sample indices,
 * passing the same @a buffer each time. On the first call with a particular
 * buffer (or after a discontiguous change in @a idx), set @a start to false.
 *
 * @param in      Array with all the raw (but native-endian) sample words for the workgroup
 * @param buffer  Sample word that help the LSBs of the previous sample if any (updated on return)
 * @param idx     Index of the sample to retrieve, relative to @a in
 * @param start   If true, @a buffer is ignored
 */
DEVICE_FN static int decode(
    const LOCAL sample_word * RESTRICT in,
    sample_word *buffer,
    unsigned int idx,
    bool start)
{
    // Optimised for the case that idx is known at compile time
    unsigned int bit_idx = idx * INPUT_SAMPLE_BITS;
    unsigned int word = bit_idx / SAMPLE_WORD_BITS;
    unsigned int bit = bit_idx % SAMPLE_WORD_BITS;
    sample_word shifted;  // has desired value in the top INPUT_SAMPLE_BITS bits

    if (bit == 0 || start)
    {
        // The buffer isn't initialised yet, or still contains the
        // previous word
        *buffer = in[word];
    }

    if (bit + INPUT_SAMPLE_BITS <= SAMPLE_WORD_BITS)
    {
        // It's already in the buffer
        shifted = *buffer << bit;
    }
    else
    {
        // It's split across the buffer and the next word
        sample_word next = in[word + 1];
        shifted = __funnelshift_l(next, *buffer, bit);
        *buffer = next;
    }
    return extract_sample(shifted);
}

DEVICE_FN static float2 cmul(float2 a, float2 b)
{
    return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// Cooperatively copy n elements from in to out
DEVICE_FN static void copy_to_local_float(LOCAL float *out, const GLOBAL float * RESTRICT in, unsigned int n)
{
    // This implementation is optimised for 'items' being a compile-time constant,
    // and also for when it is a multiple of WGS.
    unsigned int lid = get_local_id(0);
    unsigned int full_rounds = n / WGS;
    unsigned int extra = n % WGS;
    for (unsigned int i = 0; i < full_rounds; i++)
    {
        unsigned int idx = i * WGS + lid;
        out[idx] = in[idx];
    }
    if (extra > 0 && lid < extra)
    {
        unsigned int idx = full_rounds * WGS + lid;
        out[idx] = in[idx];
    }
}

// Cooperatively copy n elements from in to out
DEVICE_FN static void copy_to_local_float2(LOCAL float2 *out, const GLOBAL float2 * RESTRICT in, unsigned int n)
{
    /* This relies on some behaviour that's not totally defined (aliasing
     * float2 as float[2] but which nvcc seems to handle sanely.
     *
     * This approach does perform marginally better than copying a float2
     * at a time, presumably due to bank conflicts.
     */
    copy_to_local_float((LOCAL float *) out, (const GLOBAL float *) in, n * 2);
}

KERNEL REQD_WORK_GROUP_SIZE(WGS, 1, 1)
void ddc(
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
    const int group_in_words =
        (group_in_size * INPUT_SAMPLE_BITS + SAMPLE_WORD_BITS - 1) / SAMPLE_WORD_BITS;
    const int load_rounds = (group_in_words + WGS - 1) / WGS;
    LOCAL_DECL union
    {
        struct
        {
            sample_word in[group_in_words];
            float2 weights[TAPS];
            float2 mix_lookup[C];
        };
        float out[2][C * WGS];  // Logically float2, but split to reduce bank conflicts
    } local;

    unsigned int lid = get_local_id(0);
    /* Copy workgroup's sample data to local memory */
    unsigned int group_first_in_word =
        get_group_id(0) * samples_to_words(WGS * C * SUBSAMPLING);
#pragma unroll
    for (int i = 0; i < load_rounds; i++)
    {
        unsigned int l_idx = i * WGS + lid;
        unsigned int idx = group_first_in_word + l_idx;
        int v = (idx < in_size_words) ? in[idx] : 0;
        if (l_idx < group_in_words)
        {
            // CUDA is little-endian, but the packing uses big endian
            local.in[l_idx] = betoh(v);
        }
    }

    /* Copy weights and mix_lookup to local memory */
    copy_to_local_float2(local.weights, weights, TAPS);
    copy_to_local_float2(local.mix_lookup, mix_lookup, C);

    BARRIER();

    float2 accum[C];
    sample_word buffer[C + W - 1];
    float samples[C + W - 1];

    for (int i = 0; i < C; i++)
        accum[i] = make_float2(0.0f, 0.0f);

    unsigned int first_in_word = lid * samples_to_words(C * SUBSAMPLING);
#pragma unroll
    for (int i = 0; i < SUBSAMPLING; i++)
    {
        const int w = (W - 1) * SUBSAMPLING + i < TAPS ? W : W - 1;
#pragma unroll
        for (int j = 0; j < C + w - 1; j++)
        {
            samples[j] = (float) decode(local.in + first_in_word, &buffer[j], j * SUBSAMPLING + i, i == 0);
        }
#pragma unroll
        for (int j = 0; j < w; j++)
        {
            float2 w = local.weights[j * SUBSAMPLING + i];
            for (int k = 0; k < C; k++)
            {
                accum[k].x += samples[j + k] * w.x;
                accum[k].y += samples[j + k] * w.y;
            }
        }
    }

    // Compute the mixer for the first sample output by this workitem
    mix_bias += get_global_id(0) * (C * SUBSAMPLING) * mix_scale;
    mix_bias -= rint(mix_bias);
    float2 mix_base;
    sincospif(2 * (float) mix_bias, &mix_base.y, &mix_base.x);

#pragma unroll
    for (int i = 0; i < C; i++)
    {
        accum[i] = cmul(accum[i], mix_base);
        if (i > 0)
            accum[i] = cmul(accum[i], local.mix_lookup[i]);
    }

    BARRIER(); // Only needed because local.out is in a union

#pragma unroll
    /* Copy the results to local memory to transpose it.
     * TODO: this can cause some bank conflicts if C is a multiple of a large
     * power of 2 - see if some padding could help.
     */
    for (int i = 0; i < C; i++)
    {
        unsigned int idx = lid * C + i;
        local.out[0][idx] = accum[i].x;
        local.out[1][idx] = accum[i].y;
    }

    BARRIER();

    // Copy the results from local memory to global memory
    unsigned int first_out_idx = get_group_id(0) * (WGS * C);
#pragma unroll
    for (int i = 0; i < C; i++)
    {
        unsigned int l_idx = lid + i * WGS;
        unsigned int idx = first_out_idx + l_idx;
        if (idx < out_size)
        {
            out[idx] = make_float2(local.out[0][l_idx], local.out[1][l_idx]);
        }
    }
}
