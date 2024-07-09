/*******************************************************************************
 * Copyright (c) 2024, National Research Foundation (SARAO)
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

extern "C++"  // PyCUDA wraps the whole file in extern "C"
{
#include <curand_kernel.h>
}

<%include file="/port.mako"/>

/// Generate a random value in (-0.5, 0.5)
DEVICE_FN float dither(curandStateXORWOW_t *state)
{
    /* This magic value is chosen so that the largest possible return value
     * can be added to 127 and still produce 127.49999 rather than 127.5
     * (found experimentally). That ensures that exact integer values will not
     * be altered by dithering.
     */
    const float scale = 2.3282709e-10f;  // == 0xffff7f00p-64
    /* curand(state) returns a value in [0, 2**32). Casting it to int gives
     * a value in [-2**31, 2**31).
     */
    int x = int(curand(state));
    /* Add 1 to x if x is negative. This gives a distribution with zero mean
     * There is a tiny non-uniformity because 0 is twice as likely to appear as
     * other values in (-2**31, 2**31).
     */
    x -= x >> 31;
    return x * scale;
}
