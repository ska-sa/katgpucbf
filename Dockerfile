# syntax = docker/dockerfile:1

################################################################################
# Copyright (c) 2021-2022, National Research Foundation (SARAO)
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy
# of the License at
#
#   https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

# Use the nvidia development image as a base. This gives access to all
# nvidia and cuda runtime and development tools. pycuda needs nvcc, so
# the development tools are necessary.

FROM nvidia/cuda:11.7.1-devel-ubuntu22.04 as base

# This "base" layer is modified to better support running with Vulkan. That's
# needed by both build-base (used by Jenkins to run unit tests) and the final
# image. Additionally, for the Vulkan drivers to work one needs
# libvulkan1, libegl1 and libxext6, but that's done in later layers together
# with other packages.
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
COPY docker/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json
# See also https://github.com/NVIDIA/nvidia-container-toolkit/issues/16
COPY docker/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json

FROM base as build-base

# Install system packages:
# - git is needed for setuptools_scm
# - wget is used to download spead2
# - pkg-config and lib* are needed for spead2
# DEBIAN_FRONTEND=noninteractive prevents apt-get from asking configuration
# questions.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    python-is-python3 \
    git \
    pkg-config \
    libboost-dev \
    libboost-program-options-dev \
    libboost-system-dev \
    libibverbs-dev \
    librdmacm-dev \
    libpcap-dev \
    libcap-dev \
    libdivide-dev \
    libvulkan-dev libxext6 libegl1 \
    wget

# Create a virtual environment
RUN python -m venv /venv
# Activate it
ENV PATH=/venv/bin:$PATH
# Install up-to-date versions of installation tools, for the benefits of
# packages not using PEP 517/518.
RUN pip install pip==21.3.1 setuptools==58.3.0 wheel==0.36.2

# Install spead2 C++ bindings. We use requirements.txt just to get the
# version, so that when we want to update we only have to do it in one place.
WORKDIR /tmp/katgpucbf
COPY requirements.txt .
WORKDIR /tmp
RUN SPEAD2_VERSION=$(grep ^spead2== katgpucbf/requirements.txt | cut -d= -f3) && \
    wget "https://github.com/ska-sa/spead2/releases/download/v$SPEAD2_VERSION/spead2-$SPEAD2_VERSION.tar.gz" && \
    tar -zxf "spead2-$SPEAD2_VERSION.tar.gz" && \
    cd "spead2-$SPEAD2_VERSION" && \
    mkdir build && \
    cd build && \
    ../configure && \
    make -j && \
    make install

# Build pycuda with a workaround to pin the numpy version. We then inject the
# patched version into pip's wheel cache, at the place it would look for the
# original pycuda wheel. The patch has been merged upstream, but not yet (Oct
# 2022) released. Once a new release is in use in requirements.txt, this will
# become useless as pip will look in a different place for the cached wheel,
# but it won't break anything.
#
# A side benefit is that later stages that install pycuda will be quick as
# they'll use the wheel.
RUN pip wheel --no-deps "pycuda @ git+https://github.com/bmerry/pycuda@fcb925e3c697e326d70969aeb8851df86466db5f" && \
    wheel_path="$HOME/.cache/pip/wheels/95/16/07/8e2ba8228e568968e7f19a2cdfa8a3c60bf47e385507a6a4e6/" && \
    mkdir -p "$wheel_path" && \
    mv pycuda-*.whl "$wheel_path"

#######################################################################

# Image used by Jenkins
FROM build-base as jenkins

# docker so that Jenkins can build a Docker image
# All the TeX stuff for building the documentation
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    docker.io \
    latexmk \
    pdf2svg \
    tex-gyre \
    texlive-latex-recommended \
    texlive-latex-extra

#######################################################################

# The above image is independent of the contents of this package (except
# for requirements.txt), and is used to form the image for Jenkins
# testing. We now install the package in a new build stage.

FROM build-base as build-py

# Install requirements (already copied to build-base image).
WORKDIR /tmp/katgpucbf
RUN pip install -r requirements.txt

# Install the package itself. Using --no-deps ensures that if there are
# requirements that aren't pinned in requirements.txt, the subsequent
# pip check will complain.
COPY . .
RUN pip install --no-deps . && pip check

#######################################################################

# Separate stage to build the C++ tools. This is in a separate build stage
# so that changes to either the C++ code or the Python code don't invalidate
# the build cache for the other.

FROM build-base as build-cxx

# Build utilities.
# We use make clean to ensure that an existing build from the build context
# won't accidentally get used instead.
WORKDIR /tmp/tools
COPY src/tools .
RUN make clean && make -j fsim

RUN wget https://raw.githubusercontent.com/ska-sa/katsdpdockerbase/master/docker-base-runtime/schedrr.c && \
    gcc -Wall -O2 -o schedrr schedrr.c

#######################################################################

# The above builds everything. Now install it into a lighter-weight runtime
# image, without all the stuff needed for the build itself.
FROM base
LABEL maintainer="MeerKAT CBF team <cbf@ska.ac.za>"

# curl is needed for running under katsdpcontroller
# numactl allows CPU and memory affinity to be controlled.
# libboost-program-options is for spead2's C++ command-line tools - not
# strictly needed but useful for debugging.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    python3 \
    python3-pip \
    python3-venv \
    python-is-python3 \
    curl \
    numactl \
    libboost-program-options1.74.0 \
    libboost-system1.74.0 \
    libibverbs1 \
    librdmacm1 \
    ibverbs-providers \
    libpcap0.8 \
    libcap2 \
    libcap2-bin \
    libvulkan1 libxext6 libegl1

COPY --from=build-py /venv /venv
COPY --from=build-cxx /tmp/tools/fsim /usr/local/bin
COPY --from=build-cxx /tmp/tools/schedrr /usr/local/bin
RUN setcap cap_sys_nice+ep /usr/local/bin/schedrr
ENV PATH=/venv/bin:$PATH
