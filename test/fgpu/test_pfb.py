import numpy as np
from katsdpsigproc import accel

from katfgpu import pfb


def pfb_fir_host(data, step, weights):
    grid = data.reshape(-1, step).astype(np.float32)
    out = np.apply_along_axis(np.convolve, 0, grid, v=weights[::-1], mode='valid')
    return out.reshape(-1)


def test_pfb_fir():
    ctx = accel.create_some_context(interactive=False)
    queue = ctx.create_command_queue()
    samples = 128 * 1024 * 1024
    step = 8192
    weights = np.array([3, 17, -4, 7], np.float32)
    taps = len(weights)
    h_in = np.random.randint(-512, 512, samples + step * (taps - 1), np.int16)
    expected = pfb_fir_host(h_in, step, weights)

    template = pfb.PFBFIRTemplate(ctx, 4)
    fn = template.instantiate(queue, samples, 8192)
    fn.ensure_all_bound()
    fn.buffer('in').set(queue, h_in)
    fn.buffer('weights').set(queue, weights)
    fn()
    h_out = fn.buffer('out').get(queue)
    np.testing.assert_array_equal(h_out, expected)


def bench_pfb_fir():
    ctx = accel.create_some_context(interactive=False)
    queue = ctx.create_command_queue()
    samples = 128 * 1024 * 1024
    step = 8192
    taps = 4
    h_in = np.random.randint(-512, 512, samples + step * (taps - 1), np.int16)

    template = pfb.PFBFIRTemplate(ctx, 4)
    fn = template.instantiate(queue, samples, 8192)
    fn.ensure_all_bound()
    fn.buffer('in').set(queue, h_in)
    for i in range(100):
        fn()


if __name__ == '__main__':
    bench_pfb_fir()
