import numpy as np
from katsdpsigproc import accel

from katfgpu import pfb


def decode_10bit_host(data):
    bits = np.unpackbits(data).reshape(-1, 10)
    # Replicate the high (sign) bit
    extra = np.tile(bits[:, 0:1], (1, 6))
    combined = np.hstack([extra, bits])
    packed = np.packbits(combined)
    return packed.view('>i2').astype('i2')


def pfb_fir_host(data, channels, weights):
    decoded = decode_10bit_host(data)
    grid = decoded.reshape(-1, 2 * channels).astype(np.float32)
    out = np.apply_along_axis(np.convolve, 0, grid, v=weights[::-1], mode='valid')
    return out


def test_pfb_fir(repeat=1):
    ctx = accel.create_some_context(interactive=False)
    queue = ctx.create_command_queue()

    weights = np.array([3, 17, -4, 7], np.float32)
    taps = len(weights)
    spectra = 3123
    channels = 4096
    samples = 2 * channels * (spectra + taps - 1)
    h_in = np.random.randint(0, 256, samples * 10 // 8, np.uint8)
    expected = pfb_fir_host(h_in, channels, weights)

    template = pfb.PFBFIRTemplate(ctx, 4)
    fn = template.instantiate(queue, samples, spectra, channels)
    fn.ensure_all_bound()
    fn.buffer('in').set(queue, h_in)
    fn.buffer('weights').set(queue, weights)
    for i in range(repeat):
        # Split into two parts to test the offsetting
        fn.in_offset = 0
        fn.out_offset = 0
        fn.spectra = 1003
        fn()
        fn.in_offset = fn.spectra * 2 * channels
        fn.out_offset = fn.spectra
        fn.spectra = spectra - fn.spectra
        fn()
    h_out = fn.buffer('out').get(queue)
    np.testing.assert_array_equal(h_out, expected)


def test_fft():
    ctx = accel.create_some_context(interactive=False)
    queue = ctx.create_command_queue()
    spectra = 37
    channels = 256
    h_data = np.random.uniform(-5, 5, (spectra, 2 * channels)).astype(np.float32)
    expected = np.fft.rfft(h_data, axis=-1)

    fn = pfb.FFT(queue, spectra, channels)
    fn.ensure_all_bound()
    fn.buffer('in').set(queue, h_data)
    fn()
    h_out = fn.buffer('out').get(queue)
    np.testing.assert_allclose(h_out, expected, rtol=1e-4)


if __name__ == '__main__':
    test_pfb_fir(repeat=100)
