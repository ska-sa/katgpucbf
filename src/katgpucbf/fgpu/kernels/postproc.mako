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
#define UF ${unzip_factor}
#define FFT_CHANNELS ${channels // unzip_factor}

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

DEVICE_FN float2 cmul_add(float2 a, float2 b, float2 c)  // a * b + c
{
    return make_float2(c.x + a.x * b.x - a.y * b.y, c.y + a.x * b.y + a.y * b.x);
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
    float2 apc = cadd(in[0], in[2]);  // a + c
    float2 amc = csub(in[0], in[2]);  // a - c
    float2 bpd = cadd(in[1], in[3]);  // b + d
    float2 jdmjb = make_float2(in[1].y - in[3].y, in[3].x - in[1].x);  // jd - jb
    out[0] = cadd(apc, bpd);
    out[1] = cadd(amc, jdmjb);
    out[2] = csub(apc, bpd);
    out[3] = csub(amc, jdmjb);
}

// TODO: generalise to larger values of unzip_factor
#define mini_fft mini_fft${unzip_factor}

DEVICE_FN void load_data(const GLOBAL float2 *base, int ch, float2 out[UF])
{
    for (int i = 0; i < UF; i++)
    {
        out[i] = base[ch & (CHANNELS - 1)];
        ch += FFT_CHANNELS;
    }
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
    int spectra_per_heap)                     // Number of spectra per output heap
{
    LOCAL_DECL scratch_t scratch[UF][2];
    transpose_coords coords;
    transpose_coords_init_simple(&coords);
    int z = get_group_id(2);
    const float delay_scale = -1.0f / CHANNELS;

    // Load a block of data
    // The transpose happens per-accumulation.
    <%transpose:transpose_load coords="coords" block="${block}" vtx="${vtx}" vty="${vty}" args="r, c, lr, lc">
    {
        // Which spectrum within the accumulation.
        int spectrum = z * spectra_per_heap + ${r};
        int ch[2];
        ch[0] = ${c};
        if (ch[0] * 2 <= FFT_CHANNELS)  // Note: <= not <. We need to process fft_channels/2 + 1 times
        {
            // Compute the mirror channel
            ch[1] = FFT_CHANNELS - ch[0];
            // Which channel within the spectrum.
            int base_addr = spectrum * CHANNELS;
            float2 p[2][2][UF];  // Raw data; axes are pol, ch, sub-spectrum
            float2 q[2][2][UF];  // Fourier transform of p
            for (int i = 0; i < 2; i++)
            {
                load_data(in0 + base_addr, ch[i], p[0][i]);
                load_data(in1 + base_addr, ch[i], p[1][i]);
            }

            // Apply twiddle factors for Cooley-Tukey
            for (int j = 1; j < UF; j++)
            {
                for (int i = 0; i < 2; i++)
                {
                    float angle = ch[i] * j * (-2.0f / CHANNELS);
                    float2 t;
                    sincospif(angle, &t.y, &t.x);
                    for (int pol = 0; pol < 2; pol++)
                        p[pol][i][j] = cmul(p[pol][i][j], t);
                }
            }

            // Final pass of C2C transform
            for (int pol = 0; pol < 2; pol++)
                for (int i = 0; i < 2; i++)
                    mini_fft(p[pol][i], q[pol][i]);

            float2 delay = fine_delay[spectrum];
            float2 ph = phase[spectrum];
            /* Clean up C2C transform to form an R2C transform.
             * The axes of v are polarisation, ch, sub-spectrum
             */
            for (int j = 0; j < UF; j++)
            {
                float2 v[2][2];
                float2 rot;
                int chj[2];
                chj[0] = ch[0] + j * FFT_CHANNELS;
                chj[1] = CHANNELS - chj[0];
                sincospif(delay_scale * chj[0], &rot.y, &rot.x);
                for (int pol = 0; pol < 2; pol++)
                {
                    float2 z = q[pol][0][j];
                    float2 zp = q[pol][1][UF - 1 - j];
                    float2 a = make_float2(z.x + zp.x, z.y - zp.y);  // z + conj(z')
                    float2 b = make_float2(z.y + zp.y, zp.x - z.x);  // (z - conj(z')) / j
                    float2 r = cmul(b, rot);
                    v[pol][0] = make_float2(0.5 * (a.x + r.x), 0.5 * (a.y + r.y));
                    v[pol][1] = make_float2(0.5 * (a.x - r.x), -0.5 * (a.y - r.y));
                }

                // Apply fine delay and gain
                // TODO: fine_delay is common across channels and gain is common across
                // spectra, so could possibly be loaded more efficiently.
#pragma unroll
                for (int i = 0; i < 2; i++)
                {
                    float4 g = gains[chj[i]];
                    float re[2], im[2];
                    /* Fine delay is in fractions of a sample. Gets multiplied by
                     * delay_scale x channel to scale appropriately for the channel, and then
                     * constant phase is added.
                     */
                    // Note: delay_scale incorporates the minus sign
                    sincospif(delay.x * delay_scale * chj[i] + ph.x, &im[0], &re[0]);
                    sincospif(delay.y * delay_scale * chj[i] + ph.y, &im[1], &re[1]);
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
                    scratch[j][i].arr[${lr}][${lc}] = packed;
                }
            }
        }
    }
    </%transpose:transpose_load>

    BARRIER();

    // Write it out
    <%transpose:transpose_store coords="coords" block="${block}" vtx="${vtx}" vty="${vty}" args="r, c, lr, lc">
    {
        int ch = ${r};
        char4 *base = out + z * out_stride_z + ${c};
        if (ch * 2 <= FFT_CHANNELS)  // Note: <= not <. We need to process fft_channels/2 + 1 times
        {
            for (int j = 0; j < UF; j++)
            {
                base[ch] = scratch[j][0].arr[${lr}][${lc}];
                if (ch != 0)
                    base[CHANNELS - ch] = scratch[j][1].arr[${lr}][${lc}];
                ch += FFT_CHANNELS;
            }
        }
    }
    </%transpose:transpose_store>
}
