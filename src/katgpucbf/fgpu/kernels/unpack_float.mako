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

/* Provide a similar interface to unpack.mako, but for input that is already
 * in float32.
 */

typedef float sample_t;  // type returned by unpack_read

/* An "address" for a sample. */
struct unpack_t
{
    const GLOBAL sample_t *ptr;
};

/* Initialise an unpack_t, given the pointer to the base of the array and
 * the sample index.
 */
DEVICE_FN void unpack_init(unpack_t *unpack, const GLOBAL sample_t *in, unsigned int idx)
{
    unpack->ptr = in + idx;
}

// Dereference an unpack_t to get the sample value
DEVICE_FN sample_t unpack_read(const unpack_t *unpack)
{
    return *unpack->ptr;
}

// Increment an unpack_t by a given number of samples.
DEVICE_FN void unpack_advance(unpack_t *unpack, unsigned int dist)
{
    unpack->ptr += dist;
}
