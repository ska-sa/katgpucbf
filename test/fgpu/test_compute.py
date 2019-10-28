import numpy as np
from katsdpsigproc import accel

from katfgpu import compute


def test_compute():
    # TODO: this isn't a proper test, just a smoke test
    ctx = accel.create_some_context(interactive=False)
    queue = ctx.create_command_queue()

    weights = np.array([3, 17, -4, 7], np.float32)
    taps = len(weights)

    template = compute.ComputeTemplate(ctx, taps)
    fn = template.instantiate(queue, 100000000, 1280, 256, 32768)
    fn.ensure_all_bound()
    fn()
