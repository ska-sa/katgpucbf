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
<%!
    import math
%>

#define CHANNELS ${channels}
#define n ${unzip_factor}
#define m ${channels // unzip_factor}

<%transpose:transpose_data_class class_name="scratch_t" type="char4" block="${block}" vtx="${vtx}" vty="${vty}"/>
<%transpose:transpose_coords_class class_name="transpose_coords" block="${block}" vtx="${vtx}" vty="${vty}"/>

DEVICE_FN float2 cmul(float2 a, float2 b)
{
    return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

DEVICE_FN float2 cadd(float2 a, float2 b)
{
    return make_float2(a.x + b.x, a.y + b.y);
}

DEVICE_FN float2 csub(float2 a, float2 b)
{
    return make_float2(a.x - b.x, a.y - b.y);
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

/* The mini_fftN functions are FFTs for small fixed sizes. */

DEVICE_FN void mini_fft1(const float2 in[1], float2 out[1])
{
    out[0] = in[0];
}

DEVICE_FN void mini_fft2(const float2 in[2], float2 out[2])
{
    out[0] = cadd(in[0], in[1]);
    out[1] = csub(in[0], in[1]);
}

DEVICE_FN void mini_fft4(const float2 in[4], float2 out[4])
{
    float2 apc = cadd(in[0], in[2]);  // a + c (where inputs are a, b, c, d)
    float2 amc = csub(in[0], in[2]);  // a - c
    float2 bpd = cadd(in[1], in[3]);  // b + d
    float2 jdmjb = make_float2(in[1].y - in[3].y, in[3].x - in[1].x);  // jd - jb
    out[0] = cadd(apc, bpd);
    out[1] = cadd(amc, jdmjb);
    out[2] = csub(apc, bpd);
    out[3] = csub(amc, jdmjb);
}

#define mini_fft mini_fft${unzip_factor}

/* Fetch channel s from each input spectrum */
DEVICE_FN void load_data(const GLOBAL float2 *base, int s, float2 out[n])
{
    for (int r = 0; r < n; r++)
        out[r] = base[r * m + s];
}

/* Kernel that handles:
 * - Computation of a real-to-complex Fourier transform from complex-to-complex
 *   transforms
 * - Fine delays
 * - Partial time/channel transposition
 * - 8-bit quantisation
 * - Interleaving of polarisations
 *
 * The kernel assumes that the actual delay (in fractions of a sample?) has
 * been calculated, calculates the per-channel phasor and applies it.
 *
 * See fgpu.design.rst for more detailed explanations of what this kernel is
 * doing (particularly with regards to FFTs), as well as the meanings of many
 * single-letter variables.
 *
 * Work group sizing:
 * - Each thread handles 2 * vtx * unzip_factor channels from vty spectra.
 *   Channel 0 and channel channels/2 are special cases because they're their
 *   own mirror images, so vtx * unzip_factor * block * blocks_x must be at
 *   least channels/2 + 1.
 * - Thread-blocks are block x block x 1.
 * - A set of thread-blocks with the same z coordinate handles transposition of
 *   spectra_per_heap complete spectra.
 *
 * A note on stride length:
 * `out` is a multi-dimensional array of shape (heaps x channels x spectra_per_heap). If
 * it's contiguous then the strides will coincide with these dimensions, but
 * katsdpsigproc may have added some padding to satisfy alignment requirements.
 * At the moment, this isn't the case, but this code aims for robustness against
 * possible changes. The input array is guaranteed to be tightly packed and so no
 * in_stride is used.
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
    int spectra_per_heap)                     // Number of spectra per output heap
{
    LOCAL_DECL scratch_t scratch[n][2];
    transpose_coords coords;
    transpose_coords_init_simple(&coords);
    int z = get_group_id(2);
    const float delay_scale = -1.0f / CHANNELS;

    // Load a block of data
    // The transpose happens per-accumulation.
    <%transpose:transpose_load coords="coords" block="${block}" vtx="${vtx}" vty="${vty}" args="r, c, lr, lc">
    {
        int s[2];  // s[0] is s in the design doc; s[1] is -s, modulo m
        s[0] = ${c};
        if (s[0] * 2 <= m)  // Note: <= not <. We need to process m/2 + 1 times
        {
            // Compute the mirror channel (with & to wrap it)
            s[1] = (-s[0]) & (m - 1);
            // Which spectrum within the accumulation.
            int spectrum = z * spectra_per_heap + ${r};
            // Which channel within the spectrum.
            int base_addr = spectrum * CHANNELS;
            float2 Zr[2][2][n];  // Raw data; axes are pol, +/- s, sub-spectrum
            float2 Z[2][2][n];   // Final complex Fourier transform
            for (int i = 0; i < 2; i++)
            {
                load_data(in0 + base_addr, s[i], Zr[0][i]);
                load_data(in1 + base_addr, s[i], Zr[1][i]);
            }
            // Conjugate the negative channel
            for (int pol = 0; pol < 2; pol++)
                for (int r = 0; r < n; r++)
                    Zr[pol][1][r].y = -Zr[pol][1][r].y;

            // Apply twiddle factors for Cooley-Tukey
            for (int r = 1; r < n; r++)  // Skip 0 because it's a no-op
            {
                float angle = s[0] * r * (-2.0f / CHANNELS);
                float2 t;
                sincospif(angle, &t.y, &t.x);
                for (int pol = 0; pol < 2; pol++)
                    for (int i = 0; i < 2; i++)
                        Zr[pol][i][r] = cmul(Zr[pol][i][r], t);
            }

            // Final pass of C2C transform
            for (int pol = 0; pol < 2; pol++)
                for (int i = 0; i < 2; i++)
                    mini_fft(Zr[pol][i], Z[pol][i]);

            float2 delay = fine_delay[spectrum];
            float2 ph = phase[spectrum];
            /* Clean up C2C transform to form an R2C transform.
             * The axes of v are pol, +/- s, sub-spectrum
             */
            for (int p = 0; p < n; p++)
            {
                float2 X[2][2];
                float2 rot;
                int k[2];
                k[0] = p * m + s[0];
                k[1] = (-k[0]) & (CHANNELS - 1);
                sincospif(delay_scale * k[0], &rot.y, &rot.x);
                for (int pol = 0; pol < 2; pol++)
                {
                    float2 z = Z[pol][0][p];
                    float2 zn = Z[pol][1][p];
                    float2 u = cadd(z, zn);
                    float2 v = make_float2(z.y - zn.y, zn.x - z.x);  // (z - zn) / j
                    float2 r = cmul(v, rot);
                    X[pol][0] = make_float2(0.5 * (u.x + r.x), 0.5 * (u.y + r.y));
                    X[pol][1] = make_float2(0.5 * (u.x - r.x), -0.5 * (u.y - r.y));
                }

                // Apply fine delay and gain
                // TODO: fine_delay is common across channels and gain is common across
                // spectra, so could possibly be loaded more efficiently.
                for (int i = 0; i < 2; i++)
                {
                    float4 g = gains[k[i]];
                    float2 delay_rot[2];
                    /* Fine delay is in fractions of a sample. Gets multiplied by
                     * delay_scale x channel to scale appropriately for the channel, and then
                     * constant phase is added.
                     */
                    // Note: delay_scale incorporates the minus sign
                    float channel_scale = delay_scale * k[i];
                    sincospif(delay.x * channel_scale + ph.x, &delay_rot[0].y, &delay_rot[0].x);
                    sincospif(delay.y * channel_scale + ph.y, &delay_rot[1].y, &delay_rot[1].x);
                    for (int pol = 0; pol < 2; pol++)
                        X[pol][i] = cmul(X[pol][i], delay_rot[pol]);
                    X[0][i] = cmul(make_float2(g.x, g.y), X[0][i]);
                    X[1][i] = cmul(make_float2(g.z, g.w), X[1][i]);

                    // Interleave polarisations. Quantise at the same time.
                    char4 packed;
                    packed.x = quant(X[0][i].x);
                    packed.y = quant(X[0][i].y);
                    packed.z = quant(X[1][i].x);
                    packed.w = quant(X[1][i].y);
                    scratch[p][i].arr[${lr}][${lc}] = packed;
                }
            }
        }
    }
    </%transpose:transpose_load>

    BARRIER();

    // Write it out
    <%transpose:transpose_store coords="coords" block="${block}" vtx="${vtx}" vty="${vty}" args="r, c, lr, lc">
    {
        int s = ${r};
        if (s * 2 <= m)  // Note: <= not <. We need to process fft_channels/2 + 1 times
        {
            char4 *base = out + z * out_stride_z + ${c};
            for (int p = 0; p < n; p++)
            {
                int k = p * m + s;
                base[k * out_stride] = scratch[p][0].arr[${lr}][${lc}];
                if (k != 0)
                    base[(CHANNELS - k) * out_stride] = scratch[p][1].arr[${lr}][${lc}];
            }
        }
    }
    </%transpose:transpose_store>
}
