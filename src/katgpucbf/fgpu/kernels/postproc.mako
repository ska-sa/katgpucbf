/*******************************************************************************
 * Copyright (c) 2020-2024, National Research Foundation (SARAO)
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
<%include file="/kernels/complex.mako"/>
<%include file="/kernels/quant.mako"/>

% if dither:
<%include file="/kernels/curand_helpers.mako"/>
<%include file="/kernels/dither.mako"/>
#define DITHER 1
% else:
#define DITHER 0
% endif

#ifndef M_PIf
#define M_PIf 3.14159265358979323846f
#endif

#define CHANNELS ${channels}
#define OUT_LOW ${out_low}
#define OUT_CHANNELS ${out_high - out_low}
#define n ${unzip_factor}
#define m ${channels // unzip_factor}
#define OUT_BITS ${out_bits}

#if OUT_BITS == 4
// quantised complex number
typedef unsigned char qcomplex;
// quantised Jones vector (pair of complex numbers)
typedef uchar2 qjones;
#define QMAX 7.0f
#else
typedef char2 qcomplex;
typedef char4 qjones;
#define QMAX 127.0f
#endif

<%transpose:transpose_data_class class_name="scratch_t" type="qjones" block="${block}" vtx="${vtx}" vty="${vty}"/>
<%transpose:transpose_coords_class class_name="transpose_coords" block="${block}" vtx="${vtx}" vty="${vty}"/>

#if OUT_BITS == 4
// Pack two 4-bit values into a char
DEVICE_FN unsigned char pack_4bit(int2 q)
{
    return (q.x << 4) | (q.y & 0xF);
}
#endif // OUT_BITS == 4

// Quantise a Jones vector, and update number of complex values that saturated
DEVICE_FN qjones quant_jones(
#if DITHER
    curandStateXORWOW_t *rand_state,
#endif
    const cplx value[2],
    unsigned int saturated[2])
{
    int2 q[2];  // Quantised but not packed values
    for (int i = 0; i < 2; i++)
    {
        float re = value[i].x;
        float im = value[i].y;
#if DITHER
        re += dither(rand_state);
        im += dither(rand_state);
#endif
        // Note: | not || to avoid short-circuiting
        saturated[i] += quant(re, &q[i].x, QMAX) | quant(im, &q[i].y, QMAX);
    }
#if OUT_BITS == 4
    return make_uchar2(pack_4bit(q[0]), pack_4bit(q[1]));
#else
    return make_char4(q[0].x, q[0].y, q[1].x, q[1].y);
#endif
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

/* The mini_fftN functions are FFTs for small fixed sizes. Use sign=-1 for
 * a forward transform, sign=1 for a reverse transform (the normalisation
 * is not changed though).
 */

DEVICE_FN void mini_fft1(const cplx in[1], cplx out[1], int sign)
{
    out[0] = in[0];
}

DEVICE_FN void mini_fft2(const cplx in[2], cplx out[2], int sign)
{
    out[0] = cadd(in[0], in[1]);
    out[1] = csub(in[0], in[1]);
}

DEVICE_FN void mini_fft4(const cplx in[4], cplx out[4], int sign)
{
    cplx apc = cadd(in[0], in[2]);  // a + c (where inputs are a, b, c, d)
    cplx amc = csub(in[0], in[2]);  // a - c
    cplx bpd = cadd(in[1], in[3]);  // b + d
    cplx jdmjb = make_float2(sign * (in[3].y - in[1].y), sign * (in[1].x - in[3].x));  // (j*sign)(b - d)
    out[0] = cadd(apc, bpd);
    out[1] = cadd(amc, jdmjb);
    out[2] = csub(apc, bpd);
    out[3] = csub(amc, jdmjb);
}

#define mini_fft mini_fft${unzip_factor}

/* Fetch channel s from each input spectrum */
DEVICE_FN void load_data(const GLOBAL cplx *base, int s, cplx out[n])
{
    for (int r = 0; r < n; r++)
        out[r] = base[r * m + s];
}

/* Apply twiddle factors for Cooley-Tukey (in place) */
DEVICE_FN void twiddle(int s, cplx Zr[2][2][n])
{
    for (int r = 1; r < n; r++)  // Skip 0 because it's a no-op
    {
        float angle = (-2.0f * M_PIf / CHANNELS) * r * s;
        cplx t;
        __sincosf(angle, &t.y, &t.x);
        for (int pol = 0; pol < 2; pol++)
        {
            Zr[pol][0][r] = cmul(Zr[pol][0][r], t);
            Zr[pol][1][r] = cmulc(Zr[pol][1][r], t);
        }
    }
}

/* Compute the real-to-complex transform from a complex-to-complex transform
 * of the same raw data.
 *
 * @param      z    Complex-to-complex transform of some channel k
 * @param      zn   Complex-to-complex transform of channel -k
 * @param      rot  exp(-pi * j * k / CHANNELS)
 * @param[out] out  Real-to-complex transform at channels k and CHANNELS - k
 */
DEVICE_FN void fix_r2c(cplx z, cplx zn, cplx rot, cplx out[2])
{
    cplx u = make_float2(z.x + zn.x, z.y - zn.y);  // z + conj(zn)
    cplx v = make_float2(z.y + zn.y, zn.x - z.x);  // (z - conj(zn)) / j
    cplx r = cmul(v, rot);
    out[0] = make_float2(0.5 * (u.x + r.x), 0.5 * (u.y + r.y));
    out[1] = make_float2(0.5 * (u.x - r.x), -0.5 * (u.y - r.y));
}

/* Load/compute the complex-to-complex Fourier transform of the input.
 *
 * @param      s_  First channel to compute. The channels have the form
 *                 +/-(pm + s) where 0 <= p < n.
 * @param      in  Pointer to raw input for this spectrum
 * @param      in_stride  Stride between polarisations in @a in
 * @param[out] Z   Result, indexed by pol, +/- s, p.
 */
DEVICE_FN void finish_c2c(int s_, const GLOBAL cplx * RESTRICT in, unsigned int in_stride, cplx Z[2][2][n])
{
    // Compute the mirror channel (with & to wrap it)
    int s[2] = {s_, (-s_) & (m - 1)};
    cplx Zr[2][2][n];  // Raw data; axes are pol, +/- s, sub-spectrum
    for (int i = 0; i < 2; i++)
        for (int pol = 0; pol < 2; pol++)
            load_data(in + pol * in_stride, s[i], Zr[pol][i]);

    twiddle(s[0], Zr);

    // Final pass of C2C transform
    for (int pol = 0; pol < 2; pol++)
    {
        mini_fft(Zr[pol][0], Z[pol][0], -1);
        mini_fft(Zr[pol][1], Z[pol][1], 1);  // sign flipped because we're dealing with negative frequencies
    }
}

/* Load/compute the real-to-complex Fourier transform of the input.
 *
 * @param      s   First channel to compute. The channels have the form
 *                 pm + s and CHANNELS - (pm + s) where 0 <= p < n.
 * @param      in  Pointers to raw input for this spectrum
 * @param      in_stride  Stride between polarisations in @a in
 * @param[out] X   Result, indexed by pol, +/- s, p
 */
DEVICE_FN void finish_fft(int s, const GLOBAL cplx * RESTRICT in, unsigned int in_stride, cplx X[2][2][n])
{
    cplx Z[2][2][n];
    finish_c2c(s, in, in_stride, Z);

    // Clean up C2C transform to form an R2C transform.
    for (int p = 0; p < n; p++)
    {
        cplx rot;
        int k = p * m + s;
        __sincosf(k * (-M_PIf / CHANNELS), &rot.y, &rot.x);
        for (int pol = 0; pol < 2; pol++)
        {
            cplx Xt[2];  // temporary since X[pol, :, p] is not contiguous
            fix_r2c(Z[pol][0][p], Z[pol][1][p], rot, Xt);
            for (int i = 0; i < 2; i++)
                X[pol][i][p] = Xt[i];
        }
    }
}

/* Apply delay model.
 *
 * @param          gain   Gain for the channel
 * @param          phase  Total phase rotation in radians
 * @param[in,out]  X      Channelised voltage, indexed by pol
 */
DEVICE_FN cplx apply_delay_gain(cplx gain, float phase, cplx X)
{
    cplx rot;
    __sincosf(phase, &rot.y, &rot.x);
    return cmul(cmul(X, rot), gain);
}

/* Compute index in the output array.
 *
 * @param          k      Channel number, in the range [0, CHANNELS)
 */
DEVICE_FN int wrap_index(int k)
{
% if complex_pfb:
    return ((k + CHANNELS / 2) & (CHANNELS - 1)) - OUT_LOW;
% else:
    return k - OUT_LOW;
% endif
}

/* Check whether output array index (returned by wrap_index) is a real output channel */
DEVICE_FN bool valid_channel(int ch)
{
% if out_low > 0 or out_high < channels:
    return ch >= 0 && ch < OUT_CHANNELS;
% else:
    return true;
% endif
}

/* Turn a channel index into a delay scale factor
 *
 * @param          k      Channel number, in the range [0, CHANNELS)
 */
DEVICE_FN int delay_channel(int k)
{
% if complex_pfb:
    return (k >= CHANNELS / 2) ? k - CHANNELS : k;
% else:
    return k - CHANNELS / 2;
% endif
}

/* Kernel that handles:
 * - Computation of a real-to-complex Fourier transform from complex-to-complex
 *   transforms
 * - Fine delays
 * - Partial time/channel transposition
 * - 8-bit quantisation
 * - Interleaving of polarisations
 *
 * The kernel assumes that the actual delay has been calculated, calculates the
 * per-channel phasor and applies it. The fine delay is expressed in terms of
 * the phase slope, and must be scaled to the change in phase across the output
 * band.
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
 * possible changes.
 */
KERNEL REQD_WORK_GROUP_SIZE(${block}, ${block}, 1) void postproc(
    GLOBAL qjones * RESTRICT out,              // Output memory
    GLOBAL unsigned int (* RESTRICT out_saturated)[2], // Output saturation count, per heap and pol
    const GLOBAL cplx * RESTRICT in,          // Complex input voltages
    const GLOBAL float2 * RESTRICT fine_delay, // Fine delay, in fraction of a sample (per pol)
    const GLOBAL float2 * RESTRICT phase,     // Constant phase offset for fine delay (per pol) [radians]
    // Pre-quantisation scale factor per channel (.xy for pol0, .zw for pol1)
    const GLOBAL cplx2 * RESTRICT gains,
#if DITHER
    GLOBAL randState_t * RESTRICT rand_states, // Random states, indexed by linearised thread ID
#endif
    int out_stride_z,                         // Output stride between heaps
    int out_stride,                           // Output stride between channels within a heap
    int in_stride,                            // Input stride between polarisations
    int spectra_per_heap,                     // Number of spectra per output heap
    int heaps)                                // Number of output heaps
{
    LOCAL_DECL scratch_t scratch[n][2];
    LOCAL_DECL cplx l_gains[n][2][2][${block}];  // indexed by p, pol, +/- s
    transpose_coords coords;
    transpose_coords_init_simple(&coords);

#if DITHER
    /* Pick a unique randState_t to use for this workitem. We start by
     * computing a workgroup index, then use that to compute a workitem
     * index (using the fact that coords.lx/ly are the local id).
     */
    unsigned int linear_group_id = get_group_id(1) * ${groups_x} + get_group_id(0);
    unsigned int linear_id =
        linear_group_id * (${block * block}) + coords.ly * ${block} + coords.lx;
    rand_states += linear_id;

    curandStateXORWOW_t rand_state;
    rand_state_load(&rand_state, rand_states);
#endif

    for (int z = 0; z < heaps; z++)
    {
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
                    int k[2];
                    k[0] = p * m + s;
                    // & to avoid overflow in edge cases.
                    k[1] = (CHANNELS - k[0]) & (CHANNELS - 1);
                    for (int i = 0; i < 2; i++)
                    {
                        cplx g[2];
                        int ch = wrap_index(k[i]);
                        if (valid_channel(ch))
                        {
                            float4_to_float2_array(gains[ch], g);
                        }
                        else
                        {
                            g[0] = make_float2(0.0f, 0.0f);
                            g[1] = g[0];
                        }
                        // TODO: avoid taking up space for unused gains
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
                cplx X[2][2][n];   // Final Fourier transform
% if complex_pfb:
                finish_c2c(s, in + base_addr, in_stride, X);
% else:
                finish_fft(s, in + base_addr, in_stride, X);
% endif

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
                        // Skip processing channels that are not in the output
                        if (!valid_channel(wrap_index(k[i])))
                            continue;
                        const float delay_scale = -M_PIf / CHANNELS;
                        float channel_scale = delay_scale * delay_channel(k[i]);
                        cplx out[2];
                        for (int pol = 0; pol < 2; pol++)
                        {
                            cplx g = l_gains[p][pol][i][coords.lx];
                            float phase = delay[pol] * channel_scale + ph[pol];
                            out[pol] = apply_delay_gain(g, phase, X[pol][i][p]);
                        }

                        // Interleave polarisations. Quantise at the same time.
                        scratch[p][i].arr[${lr}][${lc}] = quant_jones(
#if DITHER
                            &rand_state,
#endif
                            out,
                            saturated);
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
                qjones *base = out + z * out_stride_z + ${c};
                for (int p = 0; p < n; p++)
                {
                    int k = p * m + s;
                    int ch = wrap_index(k);
                    if (valid_channel(ch))
                        base[ch * out_stride] = scratch[p][0].arr[${lr}][${lc}];
                    if (s != 0 && s * 2 != m)  // Skip the duplicates (calculation was skipped)
                    {
                        ch = wrap_index(CHANNELS - k);
                        if (valid_channel(ch))
                            base[ch * out_stride] = scratch[p][1].arr[${lr}][${lc}];
                    }
                }
            }
        }
        </%transpose:transpose_store>

        // TODO: could reduce within the workgroup and do only one atomic update
        // per workgroup. It doesn't seem to have much impact though.
        for (int i = 0; i < 2; i++)
            atomicAdd(&out_saturated[z][i], saturated[i]);

        BARRIER();  // before the next z loop iteration
    }

#if DITHER
    rand_state_save(rand_states, &rand_state);
#endif
}
