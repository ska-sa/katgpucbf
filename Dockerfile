# Use the nvidia development image as a base. This gives access to all
# nvidia and cuda runtime and development tools. pycuda needs nvcc, so
# the development tools are necessary.
# CUDA 11.3.1 is used rather than 11.4.0 because scikit-cuda seems not to work
# with the latter.
FROM nvidia/cuda:11.3.1-devel-ubuntu20.04 as build

# Install system packages:
# - git is needed for setuptools_scm
# - autoconf, automake, pkg-config, and lib* are needed for spead2
# DEBIAN_FRONTEND=noninteractive prevents apt-get from asking configuration
# questions.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    python-is-python3 \
    git \
    autoconf \
    automake \
    pkg-config \
    libboost-dev \
    libboost-program-options-dev \
    libboost-system-dev \
    libibverbs-dev \
    librdmacm-dev \
    libpcap-dev \
    libcap-dev

# Create a virtual environment
RUN python -m venv /venv
# Activate it
ENV PATH=/venv/bin:$PATH
# Install up-to-date versions of installation tools, for the benefits of
# packages not using PEP 517/518.
RUN pip install pip==21.1.3 setuptools==57.1.0 wheel==0.36.2
# Install packages needed to bootstrap spead2
RUN pip install jinja2==3.0.1 pycparser==2.20

# Install spead2 (C++ bindings)
WORKDIR /tmp/spead2
COPY 3rdparty/spead2/ .
RUN ./bootstrap.sh && \
    mkdir build && \
    cd build && \
    ../configure && \
    make -j && \
    make install

# Install requirements, copying only requirements.txt so that changes to other
# files do not bust the build cache.
WORKDIR /tmp/katgpucbf
COPY requirements.txt .
RUN pip install -r requirements.txt

# Install the package itself. Using --no-deps ensures that if there are
# requirements that aren't pinned in requirements.txt, the subsequent
# pip check will complain.
COPY . .
RUN pip install --no-deps . && pip check

# Build simulation utilities.
# We use make clean to ensure that an existing build from the build context
# won't accidentally get used instead.
RUN cd scratch/fgpu && make clean && make -j dsim
RUN cd scratch/xbgpu && make clean && make -j fsim

#######################################################################

# The above builds everything. Now install it into a lighter-weight runtime
# image, without all the stuff needed for the build itself.
FROM nvidia/cuda:11.3.1-devel-ubuntu20.04
LABEL maintainer="MeerKAT CBF team <cbf@ska.ac.za>"

# curl is needed for running under katsdpcontroller
# numactl allows CPU and memory affinity to be controlled.
# libboost-program-options is for spead2's C++ command-line tools - not
# strictly needed but useful for debugging.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python-is-python3 \
    curl \
    numactl \
    libboost-program-options1.71.0 \
    libboost-system1.71.0 \
    libibverbs1 \
    librdmacm1 \
    libpcap0.8 \
    libcap2

COPY --from=build /venv /venv
COPY --from=build /tmp/katgpucbf/scratch/fgpu/dsim /usr/local/bin
COPY --from=build /tmp/katgpucbf/scratch/xbgpu/fsim /usr/local/bin
ENV PATH=/venv/bin:$PATH
