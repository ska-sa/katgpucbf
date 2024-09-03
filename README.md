# Karoo Array Telescope GPU-Accelerated Correlator/Beamformer (katgpucbf)

This repository implements F-, X-, and B-engines for a GPU-based correlator,
developed for the extended MeerKAT Radio Telescope by the South African Radio
Astronomy Observatory (SARAO).

Detailed documentation can be found on
[readthedocs](https://katgpucbf.readthedocs.io), including a [guide for
development](https://katgpucbf.readthedocs.io/en/latest/dev-guide.html).
The documentation sources are in the [`doc/`](doc/) folder, and can be
built with Sphinx.

## Requirements
The following requirements are what we consider to be the minimum for making use
of this module. Listed are the tested / supported software versions and hardware
generations. Other platforms may well work, but are not tested.

### Hardware
* An NVIDIA GPU. For the XB-engine, it needs to be of compute capability 7.2
  or greater and have tensor cores.
* For best performance, an NVIDIA NIC with OFED Drivers for ibverbs
  functionality.

### Software
* Ubuntu 24.04
* Python 3.12
* CUDA 12.5
