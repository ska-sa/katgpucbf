################################################################################
# Copyright (c) 2020-2021, National Research Foundation (SARAO)
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

import configparser
import os
import subprocess
from glob import glob

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


class BuildExt(build_ext):
    """Customise extension builder to bootstrap the spead2 subdirectory."""

    def run(self):
        if not os.path.exists("3rdparty/spead2/bootstrap.sh"):
            raise RuntimeError(
                "3rdparty/spead2/bootstrap.sh is missing. This probably indicates that "
                "the submodule was not checked out. Try 'git submodule update --init --recursive'."
            )
        self.mkpath(self.build_temp)
        # Generate the configure script and some source files
        subprocess.check_call(["./bootstrap.sh"], cwd="3rdparty/spead2")
        # Run the configure script
        subprocess.check_call(os.path.abspath("3rdparty/spead2/configure"), cwd=self.build_temp)
        # The configure script writes a python-build.cfg with information about
        # the necessary libraries and flags.
        config = configparser.ConfigParser()
        config.read(os.path.join(self.build_temp, "python-build.cfg"))
        for extension in self.extensions:
            extension.extra_compile_args.extend(config["compiler"]["CFLAGS"].split())
            extension.extra_link_args.extend(config["compiler"]["LIBS"].split())
            extension.include_dirs.insert(0, os.path.join(self.build_temp, "include"))
        super().run()


spead2_sources = (
    glob("3rdparty/spead2/src/common_*.cpp")
    + glob("3rdparty/spead2/src/recv_*.cpp")
    + glob("3rdparty/spead2/src/send_*.cpp")
)
# The spead2 bootstrap process generates some of the source files. These might
# or might not exist at the time the globs above are run, so we need to add them
# to the list if they are missing.
spead2_gen_sources = [
    "3rdparty/spead2/src/common_loader_ibv.cpp",
    "3rdparty/spead2/src/common_loader_rdmacm.cpp",
    "3rdparty/spead2/src/common_loader_mlx5dv.cpp",
]
for source in spead2_gen_sources:
    if source not in spead2_sources:
        spead2_sources.append(source)
spead2_headers = glob("3rdparty/spead2/include/spead2/*.h")

ext_modules = []
for mod in ["fgpu", "xbgpu"]:
    src_dir = f"src/katgpucbf/{mod}/_kat{mod}"
    ext_modules.append(
        Pybind11Extension(
            f"{mod}._kat{mod}",
            sources=spead2_sources + glob(os.path.join(src_dir, "*.cpp")),
            depends=spead2_headers + glob(os.path.join(src_dir, "*.h")),
            cxx_std=17,
            include_dirs=[src_dir, "3rdparty/spead2/include"],
            extra_compile_args=["-O3"],
            # Trick pybind11 into believing each of these modules is
            # ABI-incompatible with the Python spead2 module (and each other)
            # so that it doesn't try to make them interoperable, which seems to
            # cause test failures (the exact reason for the failures hasn't
            # been investigated).
            define_macros=[("PYBIND11_INTERNALS_KIND", f'"katgpucbf_{mod}_local"')],
        )
    )

# The metadata is all in setup.cfg and pyproject.toml. We only need to configure the
# extensions here.
setup(
    ext_package="katgpucbf",  # Put extensions inside the katgpucbf package
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExt},  # Override how extensions are built
)
