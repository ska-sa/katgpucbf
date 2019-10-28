<%include file="/port.mako"/>

#define WGS ${wgs}
#define TAPS ${taps}

DEVICE_FN int samples_to_bytes(int samples)
{
    return samples + (samples >> 2);
}

DEVICE_FN float get_sample_10bit(const GLOBAL uchar * RESTRICT in, int idx)
{
    int byte_idx = samples_to_bytes(idx);
    int raw = (in[byte_idx] << 8) + in[byte_idx + 1];
    // Relies on >> doing sign extension
    int shl = 2 * (idx & 3) + 16;
    return (raw << shl) >> 22;
}

/* Each work-item is responsible for a run of input values with stride `step`.
 *
 * This approach becomes very register-heavy as the number of taps increases.
 * A better approach may be to have the work group cooperatively load a
 * rectangle of data into local memory, transpose, and work from there. While
 * local memory is smaller than the register file, multiple threads will read
 * the same value.
 */
KERNEL REQD_WORK_GROUP_SIZE(WGS, 1, 1) void pfb_fir(
    GLOBAL float * RESTRICT out,
    const GLOBAL uchar * RESTRICT in,
    const GLOBAL float * RESTRICT weights,
    int n, int step, int stepy,
    int in_offset, int out_offset)
{
    int group_x = get_group_id(0);
    int group_y = get_group_id(1);
    int lid = get_local_id(0);
    int offset = group_y * stepy + group_x * WGS + lid;
    out += offset + out_offset;
    // can't skip individual samples with pointer arithmetic, so track in_offset
    in_offset += offset;
    n -= offset;

    float samples[TAPS];
    for (int i = 0; i < TAPS - 1; i++)
    {
        samples[i] = get_sample_10bit(in, in_offset);
        in_offset += step;
    }

    float rweights[TAPS];
    for (int i = 0; i < TAPS; i++)
        rweights[i] = weights[i];

    int rows = stepy / step;
    // Unrolling by factor of TAPS makes the sample index known at compile time.
#pragma unroll ${taps}
    for (int i = 0; i < rows; i++)
    {
        int idx = i * step;
        if (idx >= n)
            break;
        samples[(i + TAPS - 1) % TAPS] = get_sample_10bit(in, in_offset + idx);
        float sum = 0.0f;
        for (int j = 0; j < TAPS; j++)
            sum += rweights[j] * samples[(i + j) % TAPS];
        out[idx] = sum;
    }
}
