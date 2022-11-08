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

// Quantise, and return saturation flag
DEVICE_FN bool quant(float value, signed char *out)
{
    bool saturated = false;
    if (value < -127.0f)
    {
        value = -127.0f;
        saturated = true;
    }
    else if (value > 127.0f)
    {
        value = 127.0f;
        saturated = true;
    }

    int q;
    // Convert to s8, round to nearest integer, and saturate
    // (saturation is redundant due to the test above, but automatic
    // for float-to-int conversions on CUDA hardware).
#ifdef __OPENCL_VERSION__
    q = convert_char_sat_rte(value);
#else
    asm("cvt.rni.sat.s8.f32 %0, %1;" : "=r" (q) : "f"(value));
#endif
    // Returning an int as a char can lose values, but since we have
    // just made sure it's s8 in the previous line, it's guaranteed
    // not to.
    *out = q;
    return saturated;
}

// Quantise, and return saturation flag
DEVICE_FN bool quant_complex(float2 value, signed char *out_x, signed char *out_y)
{
    // Note: | not || to avoid short-circuiting
    return quant(value.x, out_x) | quant(value.y, out_y);
}

/* Turn a float2 into an array of 2 floats for easier indexing */
DEVICE_FN void float2_to_array(float2 v, float out[2])
{
    out[0] = v.x;
    out[1] = v.y;
}

/* Turn a float4 into an array of 2 float2 for easier indexing */
DEVICE_FN void float4_to_float2_array(float4 v, float2 out[2])
{
    out[0] = make_float2(v.x, v.y);
    out[1] = make_float2(v.z, v.w);
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

/* Apply twiddle factors for Cooley-Tukey (in place) */
DEVICE_FN void twiddle(int s, float2 Zr[2][2][n])
{
    for (int r = 1; r < n; r++)  // Skip 0 because it's a no-op
    {
        float angle = (-2.0f / CHANNELS) * r * s;
        float2 t;
        sincospif(angle, &t.y, &t.x);
        for (int pol = 0; pol < 2; pol++)
            for (int i = 0; i < 2; i++)
                Zr[pol][i][r] = cmul(Zr[pol][i][r], t);
    }
}

/* Compute the real-to-complex transform from a complex-to-complex transform
 * of the same raw data.
 *
 * @param      z    Complex-to-complex transform of some channel k
 * @param      zn   Conjugation of channel -k
 * @param      rot  exp(-pi * j * k / CHANNELS)
 * @param[out] out  Real-to-complex transform at channels k and CHANNELS - k
 */
DEVICE_FN void fix_r2c(float2 z, float2 zn, float2 rot, float2 out[2])
{
    float2 u = cadd(z, zn);
    float2 v = make_float2(z.y - zn.y, zn.x - z.x);  // (z - zn) / j
    float2 r = cmul(v, rot);
    out[0] = make_float2(0.5 * (u.x + r.x), 0.5 * (u.y + r.y));
    out[1] = make_float2(0.5 * (u.x - r.x), -0.5 * (u.y - r.y));
}

/* Load/compute the complex-to-complex Fourier transform of the input.
 *
 * @param      s_  First channel to compute. The channels have the form
 *                 +/-(pm + s) where 0 <= p < n.
 * @param      in  Pointers to raw input for this spectrum, indexed by
 *                 polarisation
 * @param[out] Z   Result, indexed by pol, +/- s, p. The negative channels are
 *                 conjugated.
 */
DEVICE_FN void finish_c2c(int s_, const GLOBAL float2 *in[2], float2 Z[2][2][n])
{
    // Compute the mirror channel (with & to wrap it)
    int s[2] = {s_, (-s_) & (m - 1)};
    float2 Zr[2][2][n];  // Raw data; axes are pol, +/- s, sub-spectrum
    for (int i = 0; i < 2; i++)
        for (int pol = 0; pol < 2; pol++)
            load_data(in[pol], s[i], Zr[pol][i]);

    // Conjugate the negative channel
    for (int pol = 0; pol < 2; pol++)
        for (int r = 0; r < n; r++)
            Zr[pol][1][r].y = -Zr[pol][1][r].y;

    twiddle(s[0], Zr);

    // Final pass of C2C transform
    for (int pol = 0; pol < 2; pol++)
        for (int i = 0; i < 2; i++)
            mini_fft(Zr[pol][i], Z[pol][i]);
}

/* Load/compute the real-to-complex Fourier transform of the input.
 *
 * @param      s   First channel to compute. The channels have the form
 *                 pm + s and CHANNELS - (pm + s) where 0 <= p < n.
 * @param      in  Pointers to raw input for this spectrum, indexed by
 *                 polarisation
 * @param[out] X   Result, indexed by pol, +/- s, p
 */
DEVICE_FN void finish_fft(int s, const GLOBAL float2 *in[2], float2 X[2][2][n])
{
    float2 Z[2][2][n];
    finish_c2c(s, in, Z);

    // Clean up C2C transform to form an R2C transform.
    for (int p = 0; p < n; p++)
    {
        float2 rot;
        int k = p * m + s;
        sincospif(k * (-1.0f / CHANNELS), &rot.y, &rot.x);
        for (int pol = 0; pol < 2; pol++)
        {
            float2 Xt[2];  // temporary since X[pol, :, p] is not contiguous
            fix_r2c(Z[pol][0][p], Z[pol][1][p], rot, Xt);
            for (int i = 0; i < 2; i++)
                X[pol][i][p] = Xt[i];
        }
    }
}

/* Apply delay model.
 *
 * @param          k      Channel
 * @param          gain   Gain for the channel
 * @param          phase  Total phase rotation in radians
 * @param[in,out]  X      Channelised voltage, indexed by pol
 */
DEVICE_FN float2 apply_delay_gain(int k, float2 gain, float phase, float2 X)
{
    float2 rot;
    sincospif(phase, &rot.y, &rot.x);
    return cmul(cmul(X, rot), gain);
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
KERNEL REQD_WORK_GROUP_SIZE(${block}, ${block}, 1) void postproc(
    GLOBAL char4 * RESTRICT out,              // Output memory
    GLOBAL unsigned int (* RESTRICT out_saturated)[2], // Output saturation count, per heap and pol
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
    LOCAL_DECL float2 l_gains[n][2][2][${block}];  // indexed by p, pol, +/- s
    transpose_coords coords;
    transpose_coords_init_simple(&coords);
    int z = get_group_id(2);
    unsigned int saturated[2] = {0, 0};

    // Load a block of data
    // The transpose happens per-accumulation.
    <%transpose:transpose_load coords="coords" block="${block}" vtx="${vtx}" vty="${vty}" args="r, c, lr, lc">
    {
        int s = ${c};
        /* Load all the necessary gains for the subtile into local memory. This
         * depends on some implementation details of transpose_load
         * (specifically that ${c} is coords.lx plus a constant).
         */
        if (s * 2 <= m)
        {
            for (int p = coords.ly; p < n; p += ${block})
            {
                int k = p * m + s;
                for (int i = 0; i < 2; i++)
                {
                    float2 g[2];
                    // & to avoid overflow in edge cases
                    int ch = (i ? -k : k) & (CHANNELS - 1);
                    float4_to_float2_array(gains[ch], g);
                    for (int pol = 0; pol < 2; pol++)
                        l_gains[p][pol][i][coords.lx] = g[pol];
                }
            }
        }

        BARRIER();

        if (s * 2 <= m)  // Note: <= not <. We need to process m/2 + 1 times
        {
            // Which spectrum within the accumulation.
            int spectrum = z * spectra_per_heap + ${r};
            int base_addr = spectrum * CHANNELS;
            const GLOBAL float2 *in[2] = {in0 + base_addr, in1 + base_addr};
            float2 X[2][2][n];   // Final real-to-complex Fourier transform
            finish_fft(s, in, X);

            float delay[2], ph[2];
            float2_to_array(fine_delay[spectrum], delay);
            float2_to_array(phase[spectrum], ph);
            for (int p = 0; p < n; p++)
            {
                /* Apply fine delay, phase and gain.
                 * Fine delay is in fractions of a sample. Gets multiplied by
                 * delay_scale x channel to scale appropriately for the
                 * channel, and then constant phase is added.
                 * TODO: fine_delay and phase are common across channels, so
                 * could possibly be loaded more efficiently.
                 */
                int k[2];
                k[0] = p * m + s;
                k[1] = CHANNELS - k[0];
                for (int i = 0; i < 2; i++)
                {
                    /* Skip processing duplicates. This is mainly needed to
                     * avoid double-counting saturation.
                     */
                    if (i == 1 && (s == 0 || s * 2 == m))
                        continue;
                    const float delay_scale = -1.0f / CHANNELS;
                    float channel_scale = delay_scale * k[i];
                    float2 out[2];
                    for (int pol = 0; pol < 2; pol++)
                    {
                        float2 g = l_gains[p][pol][i][coords.lx];
                        float phase = delay[pol] * channel_scale + ph[pol];
                        out[pol] = apply_delay_gain(k[i], g, phase, X[pol][i][p]);
                    }

                    // Interleave polarisations. Quantise at the same time.
                    char4 packed;
                    saturated[0] += quant_complex(out[0], &packed.x, &packed.y);
                    saturated[1] += quant_complex(out[1], &packed.z, &packed.w);
                    scratch[p][i].arr[${lr}][${lc}] = packed;
                }
            }
        }

        BARRIER(); // ensure l_gains has been consumed before it gets updated
    }
    </%transpose:transpose_load>

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
                if (s != 0 && s * 2 != m)  // Skip the duplicates (calculation was skipped)
                    base[(CHANNELS - k) * out_stride] = scratch[p][1].arr[${lr}][${lc}];
            }
        }
    }
    </%transpose:transpose_store>

    // TODO: could reduce within the workgroup and do only one atomic update
    // per workgroup. It doesn't seem to have much impact though.
    for (int i = 0; i < 2; i++)
        atomicAdd(&out_saturated[z][i], saturated[i]);
}
