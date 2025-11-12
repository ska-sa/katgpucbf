# noqa
#  Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import unittest.mock
import warnings
from importlib.metadata import distribution

# Allow docs to be built even if pycuda is not installed
sys.modules["katsdpsigproc.cuda"] = unittest.mock.Mock()

sys.path.insert(0, os.path.abspath("."))

# -- Project information -----------------------------------------------------

project = "katgpucbf"
copyright = "2021-2025, National Research Foundation (SARAO)"

# Get the information from the installed package, no need to maintain it in
# multiple places.
dist = distribution(project)

# PEP 621 specifies that the author name and email in pyproject.toml are
# combined into the Author-email metadata field, in the format, so we need to
# split them again. This isn't a fully general solution, but good enough since
# we don't expect the field to contain anything too weird.
# TODO: Sphinx doesn't recognise the "maintainer" field, which we don't use
# right now but may in future. Should we concatenate them here?
author = dist.metadata["Author-email"].split("<")[0].strip()

# Sphinx provides for "release" and "version" to be different. See here:
# https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-release
# I think for our purposes, it's better if the two are the same, at least for
# now, so that you know immediately if you're reading documentation from a non-
# tagged commit.
release = dist.metadata["Version"]
version = release

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.graphviz",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinxcontrib.tikz",
    "sphinxcontrib.bibtex",
    "sphinx-mathjax-offline",
]


# From numpy code: these warnings are harmless
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

intersphinx_mapping = {
    "aiokatcp": ("https://aiokatcp.readthedocs.io/en/latest/", None),
    "black": ("https://black.readthedocs.io/en/stable/", None),
    "coverage": ("https://coverage.readthedocs.io/en/latest/", None),
    "katcbf-vlbi-resample": ("https://katcbf-vlbi-resample.readthedocs.io/en/latest/", None),
    "katpoint": ("https://katpoint.readthedocs.io/en/latest", None),
    "katsdpsigproc": ("https://katsdpsigproc.readthedocs.io/en/latest", None),
    "katsdptelstate": ("https://katsdptelstate.readthedocs.io/en/latest", None),
    "mypy": ("https://mypy.readthedocs.io/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pip": ("https://pip.pypa.io/en/stable/", None),
    "pytest": ("https://docs.pytest.org/en/stable/", None),
    "python": ("https://docs.python.org/3", None),
    "setuptools": ("https://setuptools.pypa.io/en/stable/", None),
    "spead2": ("https://spead2.readthedocs.io/en/latest", None),
    "sphinx-apidoc": ("https://www.sphinx-doc.org/en/master/", None),
}


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


# Bibtex files to look for references.
bibtex_bibfiles = ["references.bib"]


# Autodoc settings, need these here to get the proper signatures from the
# imported C++ functions and classes.
autodoc_docstring_signature = True
autoclass_content = "both"
# TODO: autodoc_default_options

todo_include_todos = True

# Adds \usetikzlibrary{...} to the latex preamble. We need "chains" and
# "fit" for rendering flowcharts.
tikz_tikzlibraries = "chains,fit"

# Force MathJax to render as SVG rather than CHTML, to work around
# https://github.com/mathjax/MathJax/issues/2701
mathjax3_config = {"loader": {"load": ["input/tex", "output/svg"]}, "startup": {"output": "svg"}}
