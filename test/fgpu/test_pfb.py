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


def pfb_fir_host(data, channels, taps):
    decoded = decode_10bit_host(data)
    # Hann window - should be equivalent to scipy.signal.windows.hann, but we can avoid
    # depending on scipy.
    window_size = 2 * channels * taps
    window = 0.5 - 0.5 * np.cos(2 * np.pi / (window_size - 1) * np.arange(window_size))
    window = window.astype(np.float32)
    step = 2 * channels
    out = np.empty((len(decoded) // step - taps + 1, step), np.float32)
    for i in range(0, len(out)):
        windowed = decoded[i * step : i * step + window_size] * window
        out[i] = np.sum(windowed.reshape(-1, step), axis=0)
    return out


def test_pfb_fir(repeat=1):
    ctx = accel.create_some_context(interactive=False)
    queue = ctx.create_command_queue()

    taps = 16
    spectra = 3123
    channels = 4096
    samples = 2 * channels * (spectra + taps - 1)
    h_in = np.random.randint(0, 256, samples * 10 // 8, np.uint8)
    expected = pfb_fir_host(h_in, channels, taps)

    template = pfb.PFBFIRTemplate(ctx, taps)
    fn = template.instantiate(queue, samples, spectra, channels)
    fn.ensure_all_bound()
    fn.buffer('in').set(queue, h_in)
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
    np.testing.assert_allclose(h_out, expected, rtol=1e-5, atol=1e-3)


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
