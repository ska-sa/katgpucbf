"""Script for installing the katxgpu package."""

import setuptools

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
)
