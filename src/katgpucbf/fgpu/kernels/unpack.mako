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

/* Before including, define DIG_SAMPLE_BITS to the number of bits per digitiser
 * sample. Samples must fit within two contiguous bytes. Thus, supported values
 * are 2-10, 12 and 16. The supported values larger than 8 are guaranteed not
 * to be split across 3 bytes only because the first sample is byte-aligned, so
 * for example 10-bit samples are guaranteed to be 2-bit aligned.
 */

/* Return the byte index in the chunk where the start of a sample will
 * be. The sample will be spread over two successive bytes but we just need to
 * know where the first one is.
 */
DEVICE_FN unsigned int samples_to_bytes(unsigned int samples)
{
    /* We want to compute samples * DIG_SAMPLE_BITS / 8, but need to avoid
     * overflow in the intermediate value, so we split DIG_SAMPLE_BITS into
     * whole bytes and fractions of a byte. We also want to make it as
     * efficient as possible, which is why we have special cases
     * (DIG_SAMPLE_BITS should be a constant expression, so the compiler will
     * make the switch statement vanish).
     */
    unsigned int addr = samples * (DIG_SAMPLE_BITS / 8);
    switch (DIG_SAMPLE_BITS % 8)
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
        addr += samples * (DIG_SAMPLE_BITS % 8) / 8;
        break;
    }
    return addr;
}

/* Extract DIG_SAMPLE_BITS-wide bitfield from an integer, with sign extension.
 * The `shift` MSBs and `32 - DIG_SAMPLE_BITS - shift` MSBs are removed.
 *
 * This relies on the shift-right operation doing sign extension if the input
 * is negative. That's not defined by C but is the behaviour of CUDA.
 */
DEVICE_FN int extract_bits(int value, int shift)
{
    return (value << shift) >> (32 - DIG_SAMPLE_BITS);
}

/* Get the bit sample at the given index from the chunk of samples, shake off
 * the unwanted surrounding pieces, and return as an integer.
 *
 * This version is appropriate when the samples may span two bytes. It is
 * intended for internal use only.
 */
DEVICE_FN int get_sample_2b(const GLOBAL uchar * RESTRICT in, unsigned int idx)
{
    // We were given the sample number. Get the byte index.
    unsigned int byte_idx = samples_to_bytes(idx);

    // We need two bytes to make sure we get all the bits of the sample.
    // TODO: probably better to make this `int32_t` instead of just `int`?
    // TODO: this may read past the end for 3/5/6/7-bit samples.
    int raw = (in[byte_idx] << 8) + in[byte_idx + 1];

    /* Number of bits to shift left to bring the MSB of the sample to the MSB
     * of the 16-bit value. Note that while the multiplication can overflow,
     * that's well-defined int C for unsigned integers (it wraps).
     */
    unsigned int shift_left = (DIG_SAMPLE_BITS % 8) * idx % 8;

    // int is 32-bit in CUDA, and we want to shift our 16-bit value to the top.
    return extract_bits(raw, shift_left + 16);
}

// Like get_sample_1b, but for cases where the sample is guaranteed to reside in 1 byte
DEVICE_FN int get_sample_1b(const GLOBAL uchar * RESTRICT in, unsigned int idx)
{
    // See comments in get_sample_2b for explanations
    unsigned int byte_idx = samples_to_bytes(idx);
    int raw = in[byte_idx];
    unsigned int shift_left = DIG_SAMPLE_BITS * idx % 8;
    return extract_bits(raw, shift_left + 24);
}

// Get a packed sample from the buffer
DEVICE_FN int get_sample(const GLOBAL uchar * RESTRICT in, unsigned int idx)
{
    if (DIG_SAMPLE_BITS == 2 || DIG_SAMPLE_BITS == 4)
        return get_sample_1b(in, idx);
    else
        return get_sample_2b(in, idx);
}
