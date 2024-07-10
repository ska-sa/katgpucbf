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

<%include file="/port.mako"/>

extern "C++"  // PyCUDA wraps the whole file in extern "C"
{
#include <curand_kernel.h>
#include <type_traits>
#include <utility>
}

/* Holds the same random state as curandStateXORWOW_t, excluding the state for
 * the Box-Muller transform. It uses an array of uint2 rather than uint so
 * that global memory instructions use 64-bit rather than 32-bit access.
 *
 * If this structure is updated, be sure to update the constants in
 * curand_helpers.py.
 */
typedef struct
{
    uint2 raw[3];
} randState_t;

// Check that CUDA hasn't changed the internals of curandStateXORWOW_t
static_assert(offsetof(curandStateXORWOW_t, d) == 0);
static_assert(std::is_same_v<decltype(std::declval<curandStateXORWOW_t>().d), unsigned int>);
static_assert(offsetof(curandStateXORWOW_t, v) == 4);
static_assert(std::is_same_v<decltype(std::declval<curandStateXORWOW_t>().v), unsigned int[5]>);

DEVICE_FN static inline void rand_state_load(curandStateXORWOW_t * RESTRICT out, const GLOBAL randState_t * RESTRICT in)
{
    // Zero initialise, just to avoid any undefined behaviour. The compiler
    // should elide this.
    *out = curandStateXORWOW_t{};
    randState_t tmp = *in;  // Load into registers as uint2's first
    out->d = tmp.raw[0].x;
    out->v[0] = tmp.raw[0].y;
    out->v[1] = tmp.raw[1].x;
    out->v[2] = tmp.raw[1].y;
    out->v[3] = tmp.raw[2].x;
    out->v[4] = tmp.raw[2].y;
}

DEVICE_FN static inline void rand_state_save(GLOBAL randState_t * RESTRICT out, const curandStateXORWOW_t * RESTRICT in)
{
    randState_t tmp;  // Prepare in registers so that writes can be uint2 sized
    tmp.raw[0].x = in->d;
    tmp.raw[0].y = in->v[0];
    tmp.raw[1].x = in->v[1];
    tmp.raw[1].y = in->v[2];
    tmp.raw[2].x = in->v[3];
    tmp.raw[2].y = in->v[4];
    *out = tmp;
}
