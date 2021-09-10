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

<%include file="/port.mako"/>
<%namespace name="transpose" file="/transpose_base.mako"/>

<%transpose:transpose_data_class class_name="scratch_t" type="char4" block="${block}" vtx="${vtx}" vty="${vty}"/>
<%transpose:transpose_coords_class class_name="transpose_coords" block="${block}" vtx="${vtx}" vty="${vty}"/>

DEVICE_FN float2 apply_delay(float2 in, float re, float im)
{
    return make_float2(in.x * re - in.y * im, in.y * re + in.x * im);
}

DEVICE_FN char quant(float value, float quant_gain)
{
    value *= quant_gain;
#ifdef __OPENCL_VERSION__
    return convert_char_sat_rte(value);
#else
    int out;
    // Convert to s8, round to nearest integer, and saturate
    asm("cvt.rni.sat.s8.f32 %0, %1;" : "=r" (out) : "f"(value));
    return out; // Returning an int as a char can lose values, but since we have
                // just made sure it's s8 in the previous line, it's guaranteed
                // not to.
#endif
}

/* Kernel that handles:
 * - discard of Nyquist frequency
 * - Fine delays
 * - Partial time/channel transposition
 * - 8-bit quantisation
 * - Interleaving of polarisations
 *
 * The kernel assumes that the actual delay (in fractions of a sample?) has been calculated, calculates the per-channel
 * phasor and applies it.
 *
 * Work group sizing:
 * - Each thread handles vtx channels from vty spectra.
 * - Thread-blocks are block x block x 1.
 * - A set of thread-blocks with the same z coordinate handles transposition of
 *   spectra_per_heap complete spectra.
 *
 * A note on stride length:
 * `out` is a multi-dimensional array of shape (heaps x channels x spectra_per_heap). If
 * it's contiguous then the strides will coincide with these dimensions, but
 * katsdpsigproc may have added some padding to satisfy alignment requirements.
 * At the moment, this isn't the case, but this code aims for robustness against
 * possible changes.
 */
KERNEL void postproc(
    GLOBAL char4 * RESTRICT out,              // Output memory.
    const GLOBAL float2 * RESTRICT in0,       // Complex input voltages (pol0)
    const GLOBAL float2 * RESTRICT in1,       // Complex input voltages (pol1)
    const GLOBAL float * RESTRICT fine_delay, // Fine delay, in fraction of a sample.
    const GLOBAL float * phase,               // Constant phase offset for fine delay [radians].
    int out_stride_z,                         // Output stride between heaps.
    int out_stride,                           // Output stride between channels within a heap.
    int in_stride,                            // Input stride between successive spectra.
    int spectra_per_heap,                     // Number of spectra per output heap.
    float delay_scale,                        // Scale factor for delay. 1/channels in magnitude.
    float quant_gain)                         // Scale factor for quantiser.
{
    LOCAL_DECL scratch_t scratch;
    transpose_coords coords;
    transpose_coords_init_simple(&coords);
    int z = get_group_id(2);

    // Load a block of data
    // The transpose happens per-accumulation.
    <%transpose:transpose_load coords="coords" block="${block}" vtx="${vtx}" vty="${vty}" args="r, c, lr, lc">
        // Which spectrum within the accumuation.
        int spectrum = z * spectra_per_heap + ${r};
        // Which channel within the spectrum.
        int addr = spectrum * in_stride + ${c};
        // Load the data. `float2` type handles both real and imag.
        float2 v0 = in0[addr];
        float2 v1 = in1[addr];

        // Apply fine delay.
        // TODO: load delays more efficiently (it's common across channels)
        float delay = fine_delay[spectrum];
        float ph = phase[spectrum];
        float re, im;
        /* Fine delay is in fractions of a sample. Gets multiplied by
         * delay_scale x ${c} to scale appropriately for the channel, and then
         * constant phase is added.
         */
        // Note: delay_scale incorporates the minus sign
        sincospif(delay * delay_scale * ${c} + ph, &im, &re);
        v0 = apply_delay(v0, re, im);
        v1 = apply_delay(v1, re, im);

        // Interleave polarisations. Quantise at the same time.
        char4 packed;
        packed.x = quant(v0.x, quant_gain);
        packed.y = quant(v0.y, quant_gain);
        packed.z = quant(v1.x, quant_gain);
        packed.w = quant(v1.y, quant_gain);
        scratch.arr[${lr}][${lc}] = packed;
    </%transpose:transpose_load>

    BARRIER();

    // Write it out
    <%transpose:transpose_store coords="coords" block="${block}" vtx="${vtx}" vty="${vty}" args="r, c, lr, lc">
        // Calculate the out address in one step instead of two as previously.
        int addr = z * out_stride_z + ${r} * out_stride + ${c};
        out[addr] = scratch.arr[${lr}][${lc}];
    </%transpose:transpose_store>
}
