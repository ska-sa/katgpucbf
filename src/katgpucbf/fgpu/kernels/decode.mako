<%include file="/port.mako"/>

/* As-simple-as-possible implementation of converting 10-bit to 16-bit. */
KERNEL REQD_WORK_GROUP_SIZE(${wgs}, 1, 1) void decode_10bit(
    GLOBAL short * RESTRICT out,
    const GLOBAL uchar * RESTRICT in)
{
    int sample_idx = get_global_id(0);
    int byte_idx = sample_idx + (sample_idx >> 2);
    int shift = (sample_idx & 3) * 2;
    unsigned short raw = (in[byte_idx] << 8) + in[byte_idx + 1];
    // Shift so that the relevant 10 bits are at the highest bits
    raw <<= shift;
    // Depend on >> doing sign extension
    short value = ((short) raw) >> 6;
    out[sample_idx] = value;
}
