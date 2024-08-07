# Karoo Array Telescope GPU-Accelerated Correlator/Beamformer (katgpucbf)

This repository implements both F- and X-engines for a GPU-based correlator,
developed for the MeerKAT Extension Radio Telescope by the South African Radio
Astronomy Observatory (SARAO). A B-engine for beamforming is planned, but not
yet included.

Detailed documentation can be found on
[readthedocs](https://katgpucbf.readthedocs.io), including a [guide for
development](https://katgpucbf.readthedocs.io/en/latest/dev-guide.html).
The documentation sources are in the [`doc/`](doc/) folder, and can be
built with Sphinx.

## Requirements
The following requirements are what we consider to be the minimum for making use
of this module. Listed are the tested / supported software versions and hardware
generations. Other platforms may well work, but have not been tested.

### Hardware
* An Nvidia GPU. For the XB-engine, it needs to be of compute capability 7.2 or greater
  and have tensor cores.
* Mellanox OFED Drivers v5.3.1 for ibverbs functionality. (Any v5+ should work.)
  For best performance, ibverbs is recommended.

### Software
* Ubuntu 22.04
* Python 3.10.
* CUDA version 12.0.1. (Most early development was done using 10.1, which may
  still work but has not been tested for some time.)
