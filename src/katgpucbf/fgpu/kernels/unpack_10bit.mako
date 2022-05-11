/*******************************************************************************
 * Copyright (c) 2020-2021, National Research Foundation (SARAO)
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

/* Return the byte index in the chunk where the start of your 10-bit sample will
 * be. The sample will be spread over two successive bytes but we just need to
 * know where the first one is.
 */
DEVICE_FN int samples_to_bytes(int samples)
{
    return samples + (samples >> 2);
}

/* Get the 10-bit sample at the given index from the chunk of samples, shake off
 * the unwanted surrounding pieces, and return as a float.
 */
DEVICE_FN float get_sample_10bit(const GLOBAL uchar * RESTRICT in, int idx)
{
    // We were given the sample number. Get the byte index.
    int byte_idx = samples_to_bytes(idx);

    // We need two bytes to make sure we get all 10 bits of the sample.
    // TODO: probably better to make this `int32_t` instead of just `int`?
    int raw = (in[byte_idx] << 8) + in[byte_idx + 1];

    //TODO replace the magic numbers 16 and 22 with expressions based on sizeof(int) * CHAR_BIT

    /* Bitwise AND with 3 (0b0000011) is equivalent to modulo 4. The position of
     * the MSB of the sample we want in the byte follows a pattern that repeats
     * every 4 bytes. (0, 2, 4, 6, 0, 2, 4, 6, etc)
     * The extra 16 is necessary because we use a 32-bit int value and the top
     * 16 bits are unused.
     */
    int shift_left = 2 * (idx & 3) + 16;

    /* Shift left to get rid of the stuff above, and right by 22 to get the
     * 10 bits that we are actually interested in. This relies on the
     * shift-right operation doing sign extension if the sample is negative.
     * This int gets promoted to a float in the return. Since we are going from
     * 10 bits to 24, we won't lose any precision, but if things change later
     * (e.g. a digitiser with more precision, or casting to __half) then this
     * might be a concern.
     */
    return (raw << shift_left) >> 22;
}
