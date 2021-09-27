# Karoo Array Telescope GPU-Accellerated Correlator/Beamformer (katgpucbf)

This repository implements both F- and X-engines for a GPU-based correlator,
developed for MeerKAT Extension by the South African Radio Astronomy Observatory
(SARAO).

Detailed documentation can be found in the `doc/` folder, and can be built with
Sphinx. See also the `Developing` section below.

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

## Developing
Katgpucbf is under active development and may change drastically from one
release to the next. A `dev-setup.sh` script is provided to get you going. This
script will set up all the required Python packages, and build the module's
documentation for reference.

Unit testing for the module is done using `pytest`.
