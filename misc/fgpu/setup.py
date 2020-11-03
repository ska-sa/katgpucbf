#!/usr/bin/env python3

import configparser
from glob import glob
import os
import subprocess

from setuptools import setup, find_packages, Extension
from distutils.command.build_ext import build_ext     # type: ignore  # typeshed doesn't capture it

import pybind11


class BuildExt(build_ext):
    def run(self):
        self.mkpath(self.build_temp)
        subprocess.check_call(['./bootstrap.sh'], cwd='3rdparty/spead2')
        subprocess.check_call(os.path.abspath('3rdparty/spead2/configure'), cwd=self.build_temp)
        config = configparser.ConfigParser()
        config.read(os.path.join(self.build_temp, 'python-build.cfg'))
        for extension in self.extensions:
            extension.extra_compile_args.extend(config['compiler']['CFLAGS'].split())
            extension.extra_link_args.extend(config['compiler']['LIBS'].split())
            extension.include_dirs.insert(0, os.path.join(self.build_temp, 'include'))
        super().run()

    def build_extensions(self):
        # Stop GCC complaining about -Wstrict-prototypes in C++ code
        try:
            self.compiler.compiler_so.remove('-Wstrict-prototypes')
        except ValueError:
            pass
        super().build_extensions()


sources = (glob('3rdparty/spead2/src/common_*.cpp') +
           glob('3rdparty/spead2/src/recv_*.cpp') +
           glob('3rdparty/spead2/src/send_*.cpp') +
           glob('src/*.cpp'))
# Generated files: might be missing from sources
gen_sources = [
    '3rdparty/spead2/src/common_loader_ibv.cpp',
    '3rdparty/spead2/src/common_loader_rdmacm.cpp',
    '3rdparty/spead2/src/common_loader_mlx5dv.cpp'
]
for source in gen_sources:
    if source not in sources:
        sources.append(source)
headers = glob('3rdparty/spead2/include/spead2/*.h')

extensions = [
    Extension(
        '_katfgpu',
        sources=sources,
        depends=headers,
        language='c++',
        include_dirs=['src', '3rdparty/spead2/include', pybind11.get_include()],
        extra_compile_args=['-std=c++17', '-g3', '-O3', '-fvisibility=hidden']
    )
]

setup(
    author='Bruce Merry',
    author_email='bmerry@ska.ac.za',
    name='katfgpu',
    version='0.1.dev0',
    description='GPU-accelerated F-engine for MeerKAT',
    ext_package='katfgpu',
    ext_modules=extensions,
    cmdclass={'build_ext': BuildExt},
    python_requires='>=3.5',
    install_requires=[
        'katsdpsigproc[CUDA]',
        'katsdpservices',
        'katsdptelstate',
        'numpy',
        'scikit-cuda',
        'typing_extensions'
    ],
    entry_points={
        'console_scripts': ['fgpu = katfgpu.main:main']
    },
    packages=find_packages(),
    package_data={'': ['kernels/*.mako']},
    include_package_data=True
)
