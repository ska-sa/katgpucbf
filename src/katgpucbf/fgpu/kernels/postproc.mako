/*******************************************************************************
 * Copyright (c) 2020-2022, National Research Foundation (SARAO)
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

#define CHANNELS ${channels}

<%transpose:transpose_data_class class_name="scratch_t" type="char4" block="${block}" vtx="${vtx}" vty="${vty}"/>
<%transpose:transpose_coords_class class_name="transpose_coords" block="${block}" vtx="${vtx}" vty="${vty}"/>

DEVICE_FN float2 apply_delay(float2 in, float re, float im)
{
    return make_float2(in.x * re - in.y * im, in.y * re + in.x * im);
}

DEVICE_FN float2 cmul(float2 a, float2 b)
{
    return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

DEVICE_FN char quant(float value)
{
    int out;
#ifdef __OPENCL_VERSION__
    out = convert_char_sat_rte(value);
#else
    // Convert to s8, round to nearest integer, and saturate
    asm("cvt.rni.sat.s8.f32 %0, %1;" : "=r" (out) : "f"(value));
#endif
    // Clamp -128 to -127 to give symmetrical output
    out = (out == -128) ? -127 : out;
    return out; // Returning an int as a char can lose values, but since we have
                // just made sure it's s8 in the previous line, it's guaranteed
                // not to.
}

/* Kernel that handles:
 * - Computation of real-to-complex Fourier transform from a complex-to-complex transform
 * - Fine delays
 * - Partial time/channel transposition
 * - 8-bit quantisation
 * - Interleaving of polarisations
 *
 * The kernel assumes that the actual delay (in fractions of a sample?) has
 * been calculated, calculates the per-channel phasor and applies it.
 *
 * Work group sizing:
 * - Each thread handles 2 * vtx channels from vty spectra. It handles vtx
 *   contiguous channels plus their mirror image (i and channels - i).
 *   Channel 0 and channel channels/2 are special cases because they're their
 *   own mirror images, so vtx * block * blocks_x must be at least
 *   channels/2 + 1.
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
    GLOBAL char4 * RESTRICT out,              // Output memory
    const GLOBAL float2 * RESTRICT in0,       // Complex input voltages (pol0)
    const GLOBAL float2 * RESTRICT in1,       // Complex input voltages (pol1)
    const GLOBAL float2 * RESTRICT fine_delay, // Fine delay, in fraction of a sample (per pol)
    const GLOBAL float2 * RESTRICT phase,     // Constant phase offset for fine delay (per pol) [radians]
    // Pre-quantisation scale factor per channel (.xy for pol0, .zw for pol1)
    const GLOBAL float4 * RESTRICT gains,
    int out_stride_z,                         // Output stride between heaps
    int out_stride,                           // Output stride between channels within a heap
    int in_stride,                            // Input stride between successive spectra
    int spectra_per_heap)                     // Number of spectra per output heap
{
    LOCAL_DECL scratch_t scratch[2];
    transpose_coords coords;
    transpose_coords_init_simple(&coords);
    int z = get_group_id(2);
    const int channels = ${channels};
    const float delay_scale = -1.0f / channels;

    // Load a block of data
    // The transpose happens per-accumulation.
    <%transpose:transpose_load coords="coords" block="${block}" vtx="${vtx}" vty="${vty}" args="r, c, lr, lc">
    {
        int ch[2];
        ch[0] = ${c};
        if (ch[0] * 2 <= channels)  // Note: <= not <. We need to process channels/2 + 1 times
        {
            // Compute the mirror channel, making channel 0 its own mirror
            ch[1] = ch[0] ? channels - ch[0] : 0;
            // Which spectrum within the accumulation.
            int spectrum = z * spectra_per_heap + ${r};
            // Which channel within the spectrum.
            int addr[2];
            for (int i = 0; i < 2; i++)
                addr[i] = spectrum * in_stride + ch[i];
            // Load the data. `float2` type handles both real and imag.
            // p relates to ch[0], q to ch[1]
            float2 p[2], q[2];
            p[0] = in0[addr[0]];
            q[0] = in0[addr[1]];
            p[1] = in1[addr[0]];
            q[1] = in1[addr[1]];

            /* Clean up C2C transform to form an R2C transform.
             * The axes of v are polarisation and channel (index into ch)
             */
            float2 v[2][2];
            // TODO: can probably use __sincos for better efficiency
            float2 rot;
            sincospif(delay_scale * ch[0], &rot.y, &rot.x);
            for (int pol = 0; pol < 2; pol++)
            {
                float2 a = make_float2(p[pol].x + q[pol].x, p[pol].y - q[pol].y);  // p + conj(q)
                float2 b = make_float2(p[pol].y + q[pol].y, q[pol].x - p[pol].x);  // (p - conj(q)) / j
                float2 r = cmul(b, rot);
                v[pol][0] = make_float2(0.5 * (a.x + r.x), 0.5 * (a.y + r.y));
                v[pol][1] = make_float2(0.5 * (a.x - r.x), -0.5 * (a.y - r.y));
            }

            // Apply fine delay and gain
            // TODO: fine_delay is common across channels and gain is common across
            // spectra, so could possibly be loaded more efficiently.
            float2 delay = fine_delay[spectrum];
            float2 ph = phase[spectrum];
#pragma unroll
            for (int i = 0; i < 2; i++)
            {
                float4 g = gains[ch[i]];
                float re[2], im[2];
                /* Fine delay is in fractions of a sample. Gets multiplied by
                 * delay_scale x channel to scale appropriately for the channel, and then
                 * constant phase is added.
                 */
                // Note: delay_scale incorporates the minus sign
                sincospif(delay.x * delay_scale * ch[i] + ph.x, &im[0], &re[0]);
                sincospif(delay.y * delay_scale * ch[i] + ph.y, &im[1], &re[1]);
                for (int pol = 0; pol < 2; pol++)
                    v[pol][i] = apply_delay(v[pol][i], re[pol], im[pol]);
                v[0][i] = cmul(make_float2(g.x, g.y), v[0][i]);
                v[1][i] = cmul(make_float2(g.z, g.w), v[1][i]);

                // Interleave polarisations. Quantise at the same time.
                char4 packed;
                packed.x = quant(v[0][i].x);
                packed.y = quant(v[0][i].y);
                packed.z = quant(v[1][i].x);
                packed.w = quant(v[1][i].y);
                scratch[i].arr[${lr}][${lc}] = packed;
            }
        }
    }
    </%transpose:transpose_load>

    BARRIER();

    // Write it out
    <%transpose:transpose_store coords="coords" block="${block}" vtx="${vtx}" vty="${vty}" args="r, c, lr, lc">
    {
        int ch[2];
        ch[0] = ${r};
        if (ch[0] * 2 <= channels)  // Note: <= not <. We need to process channels/2 + 1 times
        {
            ch[1] = ch[0] ? channels - ch[0] : 0;
            for (int i = 0; i < 2; i++)
            {
                // Avoid writing channel 0 twice, because the calculation for
                // the mirrored value doesn't give the right result.
                if (i == 0 || ch[0] != 0)
                {
                    // Calculate the out address in one step instead of two as previously.
                    int addr = z * out_stride_z + ch[i] * out_stride + ${c};
                    out[addr] = scratch[i].arr[${lr}][${lc}];
                }
            }
        }
    }
    </%transpose:transpose_store>
}
