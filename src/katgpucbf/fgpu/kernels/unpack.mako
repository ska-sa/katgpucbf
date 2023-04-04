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
 * sample. Samples must fit within two contiguous bytes. Thus, supported values
 * are 2-10, 12 and 16. The supported values larger than 8 are guaranteed not
 * to be split across 3 bytes only because the first sample is byte-aligned, so
 * for example 10-bit samples are guaranteed to be 2-bit aligned.
 */

typedef int sample_t;  // Type returned by unpack_read

/* Return the byte index in the chunk where the start of a sample will
 * be. The sample may be spread over two successive bytes but we just need to
 * know where the first one is.
 */
DEVICE_FN unsigned int samples_to_bytes(unsigned int samples)
{
    /* We want to compute samples * INPUT_SAMPLE_BITS / 8, but need to avoid
     * overflow in the intermediate value, so we split INPUT_SAMPLE_BITS into
     * whole bytes and fractions of a byte. We also want to make it as
     * efficient as possible, which is why we have special cases
     * (INPUT_SAMPLE_BITS should be a constant expression, so the compiler will
     * make the switch statement vanish).
     */
    unsigned int addr = samples * (INPUT_SAMPLE_BITS / 8);
    switch (INPUT_SAMPLE_BITS % 8)
    {
    case 1:
        addr += samples >> 3;
        break;
    case 2:
        addr += samples >> 2;
        break;
    case 4:
        addr += samples >> 1;
        break;
    default:
        /* The Python code checks that the buffer has at most 2^29 samples,
         * to ensure that this will not overflow.
         */
        addr += samples * (INPUT_SAMPLE_BITS % 8) / 8;
        break;
    }
    return addr;
}

/* Extract INPUT_SAMPLE_BITS-wide bitfield from an integer, with sign extension.
 * The `shift` MSBs and `32 - INPUT_SAMPLE_BITS - shift` MSBs are removed.
 *
 * This relies on the shift-right operation doing sign extension if the input
 * is negative. That's not defined by C but is the behaviour of CUDA.
 */
DEVICE_FN sample_t extract_bits(int value, int shift)
{
    return (value << shift) >> (32 - INPUT_SAMPLE_BITS);
}

/* An "address" for a sample. It contains a pointer to the first byte
 * of the sample and a shift value to pass to extract_bits.
 */
struct unpack_t
{
    const GLOBAL unsigned char *ptr;
    int shift;
};

/* Initialise an unpack_t, given the pointer to the base of the array and
 * the sample index.
 */
DEVICE_FN void unpack_init(unpack_t *unpack, const GLOBAL unsigned char *in, unsigned int idx)
{
    int shift = (INPUT_SAMPLE_BITS % 8) * idx % 8;
    if (INPUT_SAMPLE_BITS == 2 || INPUT_SAMPLE_BITS == 4)
        shift += 24;  // To shift an 8-bit value to the top of a 32-bit value
    else
        shift += 16;  // To shift a 16-bit value to the top of a 32-bit value
    unpack->shift = shift;
    unpack->ptr = in + samples_to_bytes(idx);
}

// Dereference an unpack_t to get the sample value
DEVICE_FN sample_t unpack_read(const unpack_t *unpack)
{
    if (INPUT_SAMPLE_BITS == 8)
        return *(const GLOBAL char *) unpack->ptr;
    else
    {
        int raw;
        if (INPUT_SAMPLE_BITS == 2 || INPUT_SAMPLE_BITS == 4)
            raw = unpack->ptr[0];
        else
            raw = (unpack->ptr[0] << 8) + unpack->ptr[1];
        return extract_bits(raw, unpack->shift);
    }
}

/* Increment an unpack_t by a given number of samples. The number of
 * samples must be a whole number of bytes (which in practice means
 * it should be a multiple of 8).
 */
DEVICE_FN void unpack_advance(unpack_t *unpack, unsigned int dist)
{
    unpack->ptr += samples_to_bytes(dist);
}
