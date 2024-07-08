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
<%include file="curand_helpers.mako"/>

/// Initialise an array of randState_t with sequential sequence numbers
KERNEL void init_randState_t(
    randState_t *out,
    unsigned long long seed,
    unsigned long long sequence_first,
    unsigned long long sequence_step,
    unsigned long long offset,
    unsigned int n)
{
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n)
        return;
    curandStateXORWOW_t tmp;
    curand_init(seed, sequence_first + id * sequence_step, offset, &tmp);
    rand_state_save(out + id, &tmp);
}
