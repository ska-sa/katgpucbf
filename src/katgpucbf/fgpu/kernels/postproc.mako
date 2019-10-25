<%include file="/port.mako"/>
<%namespace name="transpose" file="/transpose_base.mako"/>

<%transpose:transpose_data_class class_name="scratch_t" type="char4" block="${block}" vtx="${vtx}" vty="${vty}"/>
<%transpose:transpose_coords_class class_name="transpose_coords" block="${block}" vtx="${vtx}" vty="${vty}"/>

DEVICE_FN float2 apply_delay(float2 in, float re, float im)
{
    return make_float2(in.x * re - in.y * im, in.y * re + in.x * im);
}

DEVICE_FN char quant(float value, float quant_scale)
{
    value *= quant_scale;
#ifdef __OPENCL_VERSION__
    return convert_char_sat_rte(value);
#else
    int out;
    // Convert to s8, round to nearest integer, and saturate
    asm("cvt.rni.sat.s8.f32 %0, %1;" : "=r" (out) : "f"(value));
    return out;
#endif
}

/* Kernel that handles:
 * - discard of Nyquist frequency
 * - Fine delays
 * - Partial time/channel transposition
 * - 8-bit quantisation
 * - Interleaving of polarisations
 */
KERNEL void postproc(
    GLOBAL char4 * RESTRICT out,
    const GLOBAL float2 * RESTRICT in0,
    const GLOBAL float2 * RESTRICT in1,
    const GLOBAL float * RESTRICT fine_delay,
    int out_stride_z, int out_stride,
    int in_stride,
    int acc_len,
    float delay_scale,
    float quant_scale)
{
    LOCAL_DECL scratch_t scratch;
    transpose_coords coords;
    transpose_coords_init_simple(&coords);
    int z = get_group_id(2);

    // Load a block of data
    <%transpose:transpose_load coords="coords" block="${block}" vtx="${vtx}" vty="${vty}" args="r, c, lr, lc">
        int spectrum = z * acc_len + ${r};
        int addr = spectrum * in_stride + ${c};
        float2 v0 = in0[addr];
        float2 v1 = in1[addr];
        // TODO: load delays more efficiently (it's common across channels)
        float delay = fine_delay[spectrum];
        float re, im;
        // TODO: Check sign convention!
        sincospif(delay * delay_scale * ${c}, &im, &re);
        v0 = apply_delay(v0, re, im);
        v1 = apply_delay(v1, re, im);
        char4 packed;
        packed.x = quant(v0.x, quant_scale);
        packed.y = quant(v0.y, quant_scale);
        packed.z = quant(v1.x, quant_scale);
        packed.w = quant(v1.y, quant_scale);
        scratch.arr[${lr}][${lc}] = packed;
    </%transpose:transpose_load>

    BARRIER();

    // Write it out
    <%transpose:transpose_store coords="coords" block="${block}" vtx="${vtx}" vty="${vty}" args="r, c, lr, lc">
        int addr = z * out_stride_z + ${r} * out_stride + ${c};
        out[addr] = scratch.arr[${lr}][${lc}];
    </%transpose:transpose_store>
}
