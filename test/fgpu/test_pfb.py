import numpy as np
from katsdpsigproc import accel

from katgpucbf.fgpu import pfb


def decode_10bit_host(data):
    bits = np.unpackbits(data).reshape(-1, 10)
    # Replicate the high (sign) bit
    extra = np.tile(bits[:, 0:1], (1, 6))
    combined = np.hstack([extra, bits])
    packed = np.packbits(combined)
    return packed.view(">i2").astype("i2")


def pfb_fir_host(data, channels, weights):
    step = 2 * channels
    assert len(weights) % step == 0
    taps = len(weights) // step
    decoded = decode_10bit_host(data)
    window_size = 2 * channels * taps
    out = np.empty((len(decoded) // step - taps + 1, step), np.float32)
    for i in range(0, len(out)):
        windowed = decoded[i * step : i * step + window_size] * weights
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
    weights = np.random.uniform(-1.0, 1.0, (2 * channels * taps,)).astype(np.float32)
    expected = pfb_fir_host(h_in, channels, weights)

    template = pfb.PFBFIRTemplate(ctx, taps)
    fn = template.instantiate(queue, samples, spectra, channels)
    fn.ensure_all_bound()
    fn.buffer("in").set(queue, h_in)
    fn.buffer("weights").set(queue, weights)
    for _ in range(repeat):
        # Split into two parts to test the offsetting
        fn.in_offset = 0
        fn.out_offset = 0
        fn.spectra = 1003
        fn()
        fn.in_offset = fn.spectra * 2 * channels
        fn.out_offset = fn.spectra
        fn.spectra = spectra - fn.spectra
        fn()
    h_out = fn.buffer("out").get(queue)
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
    fn.buffer("in").set(queue, h_data)
    fn()
    h_out = fn.buffer("out").get(queue)
    np.testing.assert_allclose(h_out, expected, rtol=1e-4)


if __name__ == "__main__":
    test_pfb_fir(repeat=100)
