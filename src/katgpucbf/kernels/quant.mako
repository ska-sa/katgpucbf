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

/* Quantise and convert to 8 bit, and return saturation flag.
 *
 * The result is clamped to [-qmax, qmax]. The saturation flag is set if the
 * original value is outside that interval.
 */
DEVICE_FN static inline bool quant(float value, int *out, int qmax)
{
    float clamped = fminf(fmaxf(value, -qmax), qmax);
    bool saturated = value != clamped;

    int q;
    // Convert to s8, round to nearest integer, and saturate
    // (saturation is redundant due to the test above, but automatic
    // for float-to-int conversions on CUDA hardware).
#ifdef __OPENCL_VERSION__
    q = convert_char_sat_rte(clamped);
#else
    asm("cvt.rni.sat.s8.f32 %0, %1;" : "=r" (q) : "f"(clamped));
#endif
    *out = q;
    return saturated;
}

// Quantise and convert to 8 bits, clamping to [-127, 127]
DEVICE_FN static inline bool quant_8bit(float value, int *out)
{
    return quant(value, out, 127);
}
