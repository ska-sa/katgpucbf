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
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

"""Numpy utilities for qualification tests."""

import functools
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pytest


def _unwrap_pytest_approx(a: np.ndarray) -> np.ndarray:
    """Unwrap an array that has possibly been wrapped in :func:`pytest.approx`."""
    # pytest doesn't explicitly expose this class, so we have to infer it
    approx_cls = type(pytest.approx(np.array([1])))
    if a.shape == () and isinstance(a[()], approx_cls):
        return a[()].expected
    return a


def build_numpy_function(path: Path, _array_compare_counter: Iterator[int]) -> Callable[[np.ndarray], str]:
    """Build a function that saves numpy arrays for the qualification tests."""
    orig_build_err_msg = np.testing.build_err_msg

    @functools.wraps(orig_build_err_msg)
    def build_err_msg(arrays, *args, **kwargs) -> str:
        # Original only requires Iterable, but we need to iterate multiple
        # times.
        arrays = list(arrays)
        msg = orig_build_err_msg(arrays, *args, **kwargs)

        # If any of the arrays are wrapped in pytest.approx, strip that off
        # to avoid pickling the arrays (which could cause issues when loading
        # them later).
        arrays = [_unwrap_pytest_approx(array) for array in arrays]
        counter = next(_array_compare_counter)
        filename = path / f"arrays-{counter:06}.npz"
        # This is not perfect, because names can be passed positionally, but
        # the various call sites in numpy don't seem to do that.
        names = kwargs.get("names", ["ACTUAL", "DESIRED"])
        named_arrays = dict(zip(names, arrays, strict=True))
        np.savez(filename, **named_arrays)
        return msg + f"\n\nArrays written to {filename}"

    return build_err_msg
