/*******************************************************************************
 * Copyright (c) 2020-2023, National Research Foundation (SARAO)
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

/* Before including, define INPUT_SAMPLE_BITS to the number of bits per digitiser
 * sample. At most 32-bit samples are supported.
 */

typedef unsigned int sample_word;  // Type used for accessing input global memory
typedef int ssample_word;  // Signed version of sample_word
#define SAMPLE_WORD_BITS 32

typedef int sample_t;  // Type returned by unpack_read

/* Return the word index in the chunk where the start of a sample will
 * be. The sample may be spread over two successive words but we just need to
 * know where the first one is.
 */
DEVICE_FN static unsigned int samples_to_words(unsigned int samples)
{
    /* We want to compute samples * INPUT_SAMPLE_BITS / SAMPLE_WORD_BITS, but
     * need to avoid overflow in the intermediate value, so we split
     * `samples` into a portion that's a multiple of SAMPLE_WORD_BITS and a
     * remainder.
     */
    return (samples / SAMPLE_WORD_BITS) * INPUT_SAMPLE_BITS
        + (samples % SAMPLE_WORD_BITS) * INPUT_SAMPLE_BITS / SAMPLE_WORD_BITS;
}

/* An "address" for a sample. It contains a pointer to the first word
 * of the sample and a shift value to pass to extract_bits.
 */
struct unpack_t
{
    const GLOBAL sample_word *ptr;
    int shift;
};

/* Initialise an unpack_t, given the pointer to the base of the array and
 * the sample index.
 */
DEVICE_FN static void unpack_init(unpack_t *unpack, const GLOBAL sample_word *in, unsigned int idx)
{
    int shift = (INPUT_SAMPLE_BITS * idx) % SAMPLE_WORD_BITS;
    unpack->shift = shift;
    unpack->ptr = in + samples_to_words(idx);
}

DEVICE_FN static unsigned int reverse_endian(unsigned int v)
{
    return __byte_perm(v, v, 0x0123);
}

// Dereference an unpack_t to get the sample value
DEVICE_FN static sample_t unpack_read(const unpack_t *unpack)
{
    sample_word hi;  // Contains desired bits in the MSBs
    if (SAMPLE_WORD_BITS % INPUT_SAMPLE_BITS == 0)
    {
        // It's always contained entirely in one word. No funnels shifts needed.
        hi = reverse_endian(unpack->ptr[0]) << unpack->shift;
    }
    else
    {
        // It's potentially split across words
        // Place the desired bits in the MSBs of hi
        hi = __funnelshift_l(
            reverse_endian(unpack->ptr[1]), reverse_endian(unpack->ptr[0]), unpack->shift
        );
    }

    /* This relies on the shift-right operation doing sign extension if the input
     * is negative. That's not defined by C but is the behaviour of CUDA.
     */
    return ((ssample_word) hi) >> (SAMPLE_WORD_BITS - INPUT_SAMPLE_BITS);
}

/* Increment an unpack_t by a given number of samples. The number of
 * samples must be a whole number of words (which in practice means
 * it should be a multiple of 32).
 */
DEVICE_FN static void unpack_advance(unpack_t *unpack, unsigned int dist)
{
    unpack->ptr += samples_to_words(dist);
}
