import numpy as np
from katsdpsigproc import accel

from katgpucbf.fgpu import postproc


def postproc_host_pol(data, spectra, acc_len, channels, fine_delay, fringe_phase, quant_scale):
    # Throw out last channel (Nyquist frequency)
    data = data[:, :channels]
    # Compute delay phases
    channel_idx = np.arange(channels, dtype=np.float32)[np.newaxis, :]
    m2ipi = np.complex64(-2j * np.pi)
    phase = np.exp(m2ipi * fine_delay[:, np.newaxis] * channel_idx / (2 * channels) + 1j * fringe_phase[:, np.newaxis])
    assert phase.dtype == np.complex64
    corrected = data * phase.astype(np.complex64)
    # Split complex into real, imaginary
    corrected = corrected.view(np.float32).reshape(spectra, channels, 2)
    # Convert to integer
    corrected = np.rint(corrected * quant_scale)
    # Cast to integer with saturation
    corrected = np.minimum(np.maximum(corrected, -128), 127)
    corrected = corrected.astype(np.int8)
    # Partial transpose
    reshaped = corrected.reshape(-1, acc_len, channels, 2)
    return reshaped.transpose(0, 2, 1, 3)


def postproc_host(in0, in1, channels, acc_len, spectra, fine_delay, fringe_phase, quant_scale):
    out0 = postproc_host_pol(in0, channels, acc_len, spectra, fine_delay, fringe_phase, quant_scale)
    out1 = postproc_host_pol(in1, channels, acc_len, spectra, fine_delay, fringe_phase, quant_scale)
    return np.stack([out0, out1], axis=3)


def test_postproc(repeat=1):
    ctx = accel.create_some_context(interactive=False)
    queue = ctx.create_command_queue()
    channels = 4096
    acc_len = 256
    spectra = 512
    quant_scale = 0.1
    # TODO: make properly complex
    h_in0 = np.random.uniform(-512, 512, (spectra, channels + 1)).astype(np.complex64)
    h_in1 = np.random.uniform(-512, 512, (spectra, channels + 1)).astype(np.complex64)
    h_fine_delay = np.random.uniform(0.0, 2.0, (spectra,)).astype(np.float32)
    h_phase = np.random.uniform(0.0, np.pi / 2, (spectra,)).astype(np.float32)
    expected = postproc_host(h_in0, h_in1, spectra, acc_len, channels, h_fine_delay, h_phase, quant_scale)

    template = postproc.PostprocTemplate(ctx)
    fn = template.instantiate(queue, spectra, acc_len, channels)
    fn.ensure_all_bound()
    fn.buffer("in0").set(queue, h_in0)
    fn.buffer("in1").set(queue, h_in1)
    fn.buffer("fine_delay").set(queue, h_fine_delay)
    fn.buffer("phase").set(queue, h_phase / np.pi)
    fn.quant_scale = quant_scale
    for _ in range(repeat):
        fn()
    h_out = fn.buffer("out").get(queue)

    np.testing.assert_allclose(h_out, expected, atol=1)


if __name__ == "__main__":
    test_postproc(repeat=100)
