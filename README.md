# Karoo Array Telescope GPU-Accelerated Correlator/Beamformer (katgpucbf)

This repository implements both F- and X-engines for a GPU-based correlator,
developed for the MeerKAT Extension Radio Telescope by the South African Radio
Astronomy Observatory (SARAO). A B-engine for beamforming is planned, but not
yet included.

Detailed documentation can be found in the [`doc/`](doc/) folder, and can be
built with Sphinx. Included is also a [guide for development](doc/dev-guide.rst).

## Requirements
Apart from the Python dependencies, the following are requirements for running
katgpucbf:

1. Python 3.8.
2. Nvidia GPU of compute capability 7.5.
3. CUDA version 11.4.
4. Ubuntu 20.04
5. Mellanox OFED Drivers v5.3.1 for ibverbs functionality.

More recent versions of anything listed above may work, but this is untested
at time of writing.

**Note**: The F-engine should work on any CUDA or Open-CL GPU, with some minor
adaptation. The X-engine relies on Nvidia's Tensor Cores and is not portable.
The Tensor Core X-engine is adapted from
[ASTRON's implementation](https://git.astron.nl/RD/tensor-core-correlator).
