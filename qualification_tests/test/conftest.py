################################################################################
# Copyright (c) 2026, National Research Foundation (SARAO)
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
# See the for the specific language governing permissions and
# limitations under the License.
################################################################################

"""Fixtures and options for qualification testing of the CBF."""

import itertools
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest
import spead2
import spead2.recv  # noqa: F401

from qualification.numpy import build_numpy_function

pytest_plugins = ["pytester"]


def pytest_addoption(parser) -> None:
    """Register new command-line options."""
    parser.addini(
        "array_dir",
        help="Directory in which to save failed array comparisons",
        type="paths",
        default=[Path(tempfile.gettempdir())],
    )


@pytest.fixture(scope="session")
def _array_compare_counter() -> Iterator[int]:
    """Counter used to give unique filenames to array dumps."""
    return itertools.count(0)


@pytest.fixture(autouse=True)
def _array_compare(
    monkeypatch: pytest.MonkeyPatch, pytestconfig: pytest.Config, _array_compare_counter: Iterator[int]
) -> None:
    """Patch numpy.testing to save failed array comparisons if enabled."""
    paths = pytestconfig.getini("array_dir")
    if not paths:
        return  # Not enabled
    path = paths[0]
    path.mkdir(parents=True, exist_ok=True)
    build_err_msg = build_numpy_function(path, _array_compare_counter)

    # We have to patch in the private module since that's where it gets called.
    monkeypatch.setattr("numpy.testing._private.utils.build_err_msg", build_err_msg)
