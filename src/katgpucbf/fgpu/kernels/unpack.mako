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

/* Handles unpacking of packed integer samples for pfb_fir.mako.
 * Include sample.mako first.
 */

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

// Dereference an unpack_t to get the sample value
DEVICE_FN static sample_t unpack_read(const unpack_t *unpack)
{
    sample_word hi;  // Contains desired bits in the MSBs
    if (SAMPLE_WORD_BITS % INPUT_SAMPLE_BITS == 0)
    {
        // It's always contained entirely in one word. No funnel shifts needed.
        hi = betoh(unpack->ptr[0]) << unpack->shift;
    }
    else
    {
        // It's potentially split across words
        // Place the desired bits in the MSBs of hi
        hi = __funnelshift_l(betoh(unpack->ptr[1]), betoh(unpack->ptr[0]), unpack->shift);
    }
    return extract_sample(hi);
}

/* Increment an unpack_t by a given number of samples. The number of
 * samples must be a whole number of words (which in practice means
 * it should be a multiple of 32).
 */
DEVICE_FN static void unpack_advance(unpack_t *unpack, unsigned int dist)
{
    unpack->ptr += samples_to_words(dist);
}
