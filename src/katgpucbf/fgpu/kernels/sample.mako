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

typedef int sample_t;  // Type returned by extract_sample

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

/// Convert big endian value to native endian
DEVICE_FN static sample_word betoh(sample_word v)
{
    // CUDA is always little endian, so we just reverse the endianness
    return __byte_perm(v, v, 0x0123);
}

/**
 * Extract a sample from the MSBs of a sample_word.
 *
 * The implementation relies on nvcc's behaviour of doing right arithmetic
 * shift on negative values.
 */
DEVICE_FN static sample_t extract_sample(sample_word word)
{
    return ((ssample_word) word) >> (SAMPLE_WORD_BITS - INPUT_SAMPLE_BITS);
}
