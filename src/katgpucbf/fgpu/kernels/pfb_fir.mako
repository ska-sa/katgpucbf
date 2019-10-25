<%include file="/port.mako"/>

#define WGS ${wgs}
#define TAPS ${taps}

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
    const GLOBAL short * RESTRICT in,
    const GLOBAL float * RESTRICT weights,
    int n, int step, int stepy)
{
    int group_x = get_group_id(0);
    int group_y = get_group_id(1);
    int lid = get_local_id(0);
    int offset = group_y * stepy + group_x * WGS + lid;
    out += offset;
    in += offset;
    n -= offset;

    float samples[TAPS];
    for (int i = 0; i < TAPS - 1; i++)
    {
        samples[i] = *in;
        in += step;
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
        samples[(i + TAPS - 1) % TAPS] = in[idx];
        float sum = 0.0f;
        for (int j = 0; j < TAPS; j++)
            sum += rweights[j] * samples[(i + j) % TAPS];
        out[idx] = sum;
    }
}
