"""Script for installing the katxgpu package."""

import setuptools
from pybind11.setup_helpers import Pybind11Extension, build_ext

# These three imports are used in the BuildExt class.
import configparser
import subprocess
import os
from glob import glob


class BuildExt(build_ext):
    """Class to manage the building of the SPEAD2 C++ submodule."""

    def run(self):
        """Run the spead2 compilation."""
        self.mkpath(self.build_temp)
        subprocess.check_call(["./bootstrap.sh"], cwd="3rdparty/spead2")
        subprocess.check_call(os.path.abspath("3rdparty/spead2/configure"), cwd=self.build_temp)
        config = configparser.ConfigParser()
        config.read(os.path.join(self.build_temp, "python-build.cfg"))
        for extension in self.extensions:
            extension.extra_compile_args.extend(config["compiler"]["CFLAGS"].split())
            extension.extra_link_args.extend(config["compiler"]["LIBS"].split())
            extension.include_dirs.insert(0, os.path.join(self.build_temp, "include"))
        super().run()

    def build_extensions(self):
        """
        Stop GCC complaining about -Wstrict-prototypes in C++ code.

        This function has been copied from katfgpu's setup.py file. The exact need for it is not understood.
        """
        try:
            self.compiler.compiler_so.remove("-Wstrict-prototypes")
        except ValueError:
            pass
        super().build_extensions()


# Points to the pybind11 C++ sources to build. Includes paths to spead2 header and source files required for building
ext_modules = [
    Pybind11Extension(
        "_katxgpu",
        sources=(
            glob("3rdparty/spead2/src/common_*.cpp")
            + glob("3rdparty/spead2/src/recv_*.cpp")
            + glob("3rdparty/spead2/src/send_*.cpp")
            + glob("src/*.cpp")
        ),
        depends=["3rdparty/spead2/include/spead2/*.h"],  # Header files
        include_dirs=["src", "3rdparty/spead2/include"],
        extra_compile_args=["-std=c++17", "-g3", "-O3", "-fvisibility=hidden"],
    ),
]

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="katxgpu",
    version="0.0.1",
    author="Gareth Callanan",
    author_email="gcallanan@ska.ac.za",
    description="GPU-accelerated X-Engine for the MeerKAT-Extension correlator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ska-sa/katxgpu",
    packages=setuptools.find_packages(),
    package_data={"": ["kernels/*.mako"]},
    include_package_data=True,
    classifiers=[
        # "License :: OSI Approved :: GNU General Public License v2 (GPLv2)", # TBD before taking this repo public
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
    python_requires=">=3.6",
    # The following three lines are needed to install the pybind11 C++ modules:
    # 1. ext_package ensures that the pybind modules fall under the katxgpu module when importing
    # 2. ext_modules lists the pybind modules to install
    # 3. build_ext installs ensures the SPEAD2 submodule is part of the C++ compilation.
    ext_package="katxgpu",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExt},
)
