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

typedef float2 cplx;
typedef float4 cplx2;  // pair of complex values

// a * b
DEVICE_FN static inline cplx cmul(cplx a, cplx b)
{
    return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// a * conj(b)
DEVICE_FN static inline cplx cmulc(cplx a, cplx b)
{
    return make_float2(a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y);
}

// a + b
DEVICE_FN static inline cplx cadd(cplx a, cplx b)
{
    return make_float2(a.x + b.x, a.y + b.y);
}

// a - b
DEVICE_FN static inline cplx csub(cplx a, cplx b)
{
    return make_float2(a.x - b.x, a.y - b.y);
}

// a * b + c
DEVICE_FN static inline cplx cmad(cplx a, cplx b, cplx c)
{
    return make_float2(c.x + a.x * b.x - a.y * b.y, c.y + a.x * b.y + a.y * b.x);
}
