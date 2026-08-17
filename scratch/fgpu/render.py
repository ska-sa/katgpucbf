from katsdpsigproc import accel

ctx = accel.create_some_context()

resource_dir = "../../src/katgpucbf/fgpu"
render = accel.render(
    ctx,
    "kernels/pfb_fir.mako",
    {
        "wgs": 128,
        "taps": 16,
        "channels": 4096,
        "input_sample_bits": 2,
        "unzip_factor": 1,
        "complex_input": False,
        "n_pols": 2,
    },
    extra_dirs=[str(resource_dir)],
)

with open("/tmp/render.ocl", "w") as f:
    f.write(render)
