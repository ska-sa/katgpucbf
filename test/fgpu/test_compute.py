"""Smoke test for Compute class."""
from katsdpsigproc import accel

from katgpucbf.fgpu import compute


def test_compute():
    """Test creation and running of :class:`Compute`.

    .. todo:: This isn't a proper test, just a smoke test.
    """
    ctx = accel.create_some_context(interactive=False)
    queue = ctx.create_command_queue()

    template = compute.ComputeTemplate(ctx, 4)
    fn = template.instantiate(queue, 100000000, 1280, 256, 32768)
    fn.ensure_all_bound()
    fn()
