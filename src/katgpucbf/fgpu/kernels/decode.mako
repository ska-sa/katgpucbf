<%include file="/port.mako"/>

#define WGS ${wgs}

typedef union
{
    uint words[WGS * 5];
    uchar bytes[WGS * 20];
} scratch_in_t;

/* Optimised kernel for decoding 10 bit digitiser samples.
 *
 * The optimisation is mostly about regularising the memory accesses.
 * The resulting kernel is memory-bound.
 *
 * Each work-item is responsible for 16 samples. There are three phases:
 *
 * 1. Load the data. Each work-item loads 5 words of data and transfers it
 *    to local memory.
 * 2. Each work-item computes the values of 16 contiguous samples and writes
 *    the results to local memory. The local memory is padded to avoid bank
 *    conflicts.
 * 3. The local results are transferred to global memory.
 */
KERNEL REQD_WORK_GROUP_SIZE(WGS, 1, 1) void decode_10bit(
    GLOBAL short * RESTRICT out,
    const GLOBAL uint * RESTRICT in)
{
    LOCAL_DECL scratch_in_t scratch_in;
    LOCAL_DECL short scratch_out[WGS][18];   // 18 not 16 to avoid bank conflicts

    // Adjust to point at the region of data handled by this workgroup
    int group = get_group_id(0);
    in += group * (WGS * 5);
    out += group * (WGS * 16);

    int lid = get_local_id(0);
    for (int i = 0; i < 5; i++)
    {
        int addr = lid + i * WGS;
        scratch_in.words[addr] = in[addr];
    }
    BARRIER();

#pragma unroll
    for (int i = 0; i < 16; i++)
    {
        int sample_idx = lid * 16 + i;
        int byte_idx = sample_idx + (sample_idx >> 2);
        int shift = (sample_idx & 3) * 2;
        unsigned short raw = (scratch_in.bytes[byte_idx] << 8) + scratch_in.bytes[byte_idx + 1];
        // Shift so that the relevant 10 bits are at the highest bits
        raw <<= shift;
        // Depend on >> doing sign extension
        short value = ((short) raw) >> 6;
        scratch_out[lid][i] = value;
    }
    BARRIER();

    for (int i = 0; i < 16; i++)
    {
        int addr = lid + i * WGS;
        out[addr] = scratch_out[addr >> 4][addr & 15];
    }
}
