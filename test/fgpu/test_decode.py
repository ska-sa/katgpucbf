import numpy as np
from katsdpsigproc import accel

from katfgpu import decode


def decode_10bit_host(data):
    buffer = 0
    buffer_bits = 0
    out = []
    for b in data:
        buffer = (buffer << 8) | int(b)
        buffer_bits += 8
        while buffer_bits >= 10:
            buffer_bits -= 10
            value = (buffer >> buffer_bits) & 1023
            # Convert to signed
            if value & 512:
                value -= 1024
            out.append(value)
    return np.array(out, dtype=np.int16)


def test_decode():
    ctx = accel.create_some_context(interactive=False)
    queue = ctx.create_command_queue()
    samples = 128 * 1024 + 8
    h_in = np.random.randint(0, 256, samples * 10 // 8, np.uint8)
    expected = decode_10bit_host(h_in)

    template = decode.Decode10BitTemplate(ctx)
    fn = template.instantiate(queue, samples)
    fn.ensure_all_bound()
    fn.buffer('in').set(queue, h_in)
    fn()
    h_out = fn.buffer('out').get(queue)
    np.testing.assert_array_equal(h_out, expected)


def bench_decode():
    ctx = accel.create_some_context(interactive=False)
    queue = ctx.create_command_queue()
    samples = 128 * 1024 * 1024
    h_in = np.random.randint(0, 256, samples * 10 // 8, np.uint8)

    template = decode.Decode10BitTemplate(ctx)
    fn = template.instantiate(queue, samples)
    fn.ensure_all_bound()
    fn.buffer('in').set(queue, h_in)
    for i in range(1000):
        fn()


if __name__ == '__main__':
    bench_decode()
