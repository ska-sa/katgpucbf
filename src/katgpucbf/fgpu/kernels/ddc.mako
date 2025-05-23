/*******************************************************************************
 * Copyright (c) 2023, 2025, National Research Foundation (SARAO)
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
<%include file="/kernels/complex.mako"/>

#define WGS ${wgs}
#define TAPS ${taps}
#define SUBSAMPLING ${subsampling}
#define INPUT_SAMPLE_BITS ${input_sample_bits}
#define C ${unroll}
#define W ${(taps + subsampling - 1) // subsampling}

/// Raw storage type for sample data
typedef unsigned int sample_word;
/// Signed version of sample_word
typedef int ssample_word;
#define SAMPLE_WORD_BITS 32

DEVICE_FN static unsigned int reverse_endian(unsigned int v)
{
    return __byte_perm(v, v, 0x0123);
}

/**
 * Load the next sample value.
 *
 * This is intended to be called with a sequence of contiguous sample indices,
 * passing the same @a buffer each time. On the first call with a particular
 * buffer (or after a discontiguous change in @a idx), set @a start to false.
 *
 * @param in      Array with all the raw (but native-endian) sample words for the workgroup
 * @param buffer  Sample word that holds the LSBs of the previous sample if any (updated on return)
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
    // Rely on nvcc to do sign extension when right-shifting a negative
    // value (it's undefined behaviour in C).
    return ((ssample_word) shifted) >> (SAMPLE_WORD_BITS - INPUT_SAMPLE_BITS);
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
DEVICE_FN static void copy_to_local_cplx(LOCAL cplx *out, const GLOBAL cplx * RESTRICT in, unsigned int n)
{
    /* This relies on some behaviour that's not totally defined (aliasing
     * float2 as float[2]) but which nvcc seems to handle sanely.
     *
     * This approach does perform marginally better than copying a float2
     * at a time, presumably due to bank conflicts.
     */
    copy_to_local_float((LOCAL float *) out, (const GLOBAL float *) in, n * 2);
}

KERNEL REQD_WORK_GROUP_SIZE(WGS, 1, 1)
void ddc(
    GLOBAL cplx * RESTRICT out,
    const GLOBAL sample_word * RESTRICT in,
    const GLOBAL cplx * RESTRICT weights,
    unsigned int out_stride, // stride between pols, unit: cplx
    unsigned int in_stride,  // stride between pols, unit: sample_word
    unsigned int out_size,
    unsigned int in_size_words,
    unsigned long mix_scale,  // Mixer frequency in cycles per SUBSAMPLING samples, fixed point
    unsigned long mix_bias    // Mixer phase in cycles at the first sample, fixed point
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
            cplx weights[TAPS];
        };
        float out[2][C * WGS];  // Logically cplx, but split to reduce bank conflicts
    } local;

    int pol = get_group_id(1);
    out += pol * out_stride;
    in += pol * in_stride;

    unsigned int lid = get_local_id(0);
    /* Copy workgroup's sample data to local memory */
    unsigned int group_first_in_word =
        get_group_id(0) * (WGS * C * SUBSAMPLING * INPUT_SAMPLE_BITS / SAMPLE_WORD_BITS);
#pragma unroll
    for (int i = 0; i < load_rounds; i++)
    {
        unsigned int l_idx = i * WGS + lid;
        unsigned int idx = group_first_in_word + l_idx;
        int v = (idx < in_size_words) ? in[idx] : 0;
        if (l_idx < group_in_words)
        {
            // CUDA is little-endian, but the packing uses big endian
            local.in[l_idx] = reverse_endian(v);
        }
    }

    /* Copy weights to local memory */
    copy_to_local_cplx(local.weights, weights, TAPS);

    BARRIER();

    cplx accum[C];
    sample_word buffer[C + W - 1];
    float samples[C + W - 1];

    for (int i = 0; i < C; i++)
        accum[i] = make_float2(0.0f, 0.0f);

    unsigned int first_in_word = lid * (C * SUBSAMPLING * INPUT_SAMPLE_BITS / SAMPLE_WORD_BITS);
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
            cplx w = local.weights[j * SUBSAMPLING + i];
            for (int k = 0; k < C; k++)
            {
                accum[k].x += samples[j + k] * w.x;
                accum[k].y += samples[j + k] * w.y;
            }
        }
    }

    unsigned long mix_cycles = get_global_id(0) * C * mix_scale + mix_bias;
#pragma unroll
    for (int i = 0; i < C; i++)
    {
        cplx mix;
        // Casting from unsigned long to long changes the range from [0, 2pi) to
        // [-pi, pi). The magic number is 2^64, used to convert fixed-point
        // representation to real.
        __sincosf(2 * (float) M_PI / 18446744073709551616.0f * (long) mix_cycles, &mix.y, &mix.x);
        accum[i] = cmul(accum[i], mix);
        mix_cycles += mix_scale;
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
