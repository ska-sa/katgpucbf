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
"""Numpy dump plugin unit tests."""

from pathlib import Path

import numpy as np
import pytest
from _pytest.python_api import ApproxBase


@pytest.fixture(autouse=True)
def setup_pytester(pytester: pytest.Pytester) -> None:
    """Set up pytester with conftest, ini, and test files."""
    pytester.makeconftest(
        """
        pytest_plugins = [
            "katgpucbf.pytest_plugins.reporter_plugin",
            "katgpucbf.pytest_plugins.numpy_dump",
        ]
        """
    )
    pytester.makeini(
        """
        [pytest]
        addopts = --report-log=report.json
        raw_data = false
        array_dir = arrays
        """
    )
    pytester.copy_example("demo.py")


def _extract_array_path_from_output(outlines: list[str], pytester_path: Path) -> Path | None:
    """Extract the array path from pytest output lines that contain 'Arrays written to'."""
    for line in outlines:
        if "Arrays written to" in line:
            # Extract the path after "Arrays written to "
            prefix = "Arrays written to "
            idx = line.find(prefix)
            if idx != -1:
                array_path_str = line[idx + len(prefix) :].strip()
                # The path might be relative to pytester.path or absolute
                array_path = Path(array_path_str)
                if not array_path.is_absolute():  # TODO: uuhhh...
                    array_path = pytester_path / array_path
                return array_path
    return None


def test_failed_np_assertion_dumps_arrays(pytester: pytest.Pytester) -> None:
    """Test that the numpy fail test is reported correctly."""
    result = pytester.runpytest("demo.py::test_numpy_fails")
    result.assert_outcomes(failed=1)
    assert any("Arrays written to" in line for line in result.outlines), (
        "Arrays written to should be present in the report"
    )

    # Extract the path from the output
    array_path = _extract_array_path_from_output(result.outlines, pytester.path)

    assert array_path is not None, "Could not find array path in output"
    assert array_path.exists(), f"Array file should exist at {array_path}"

    # Open and verify the file
    with np.load(array_path) as data:
        assert "ACTUAL" in data, "Array file should contain ACTUAL array"
        assert "DESIRED" in data, "Array file should contain DESIRED array"
        assert np.all(data["ACTUAL"] == np.array([1, 2, 3]))
        assert np.all(data["DESIRED"] == np.array([4, 5, 6]))


def test_failed_np_assertion_dumps_arrays_with_scalar_comparison(pytester: pytest.Pytester) -> None:
    """Test that the numpy fail test is reported correctly."""
    result = pytester.runpytest("demo.py::test_numpy_fails_with_scalar_comparison")
    result.assert_outcomes(failed=1)
    assert any("Arrays written to" in line for line in result.outlines), (
        "Arrays written to should be present in the report"
    )

    # Extract the path from the output
    array_path = _extract_array_path_from_output(result.outlines, pytester.path)

    assert array_path is not None, "Could not find array path in output"
    assert array_path.exists(), f"Array file should exist at {array_path}"

    # Open and verify the file
    with np.load(array_path) as data:
        assert "ACTUAL" in data, "Array file should contain ACTUAL array"
        assert "DESIRED" in data, "Array file should contain DESIRED array"
        assert np.all(data["ACTUAL"] == np.array([1, 1, 1]))
        assert np.all(data["DESIRED"] == np.array([2]))


def test_failed_np_assertion_dumps_arrays_and_unwraps_approx(pytester: pytest.Pytester) -> None:
    """Test that the numpy fail approx test is reported correctly."""
    result = pytester.runpytest("demo.py::test_numpy_with_approx_fails")
    result.assert_outcomes(failed=1)

    # Extract the path from the output
    array_path = _extract_array_path_from_output(result.outlines, pytester.path)

    assert array_path is not None, "Could not find array path in output"
    assert array_path.exists(), f"Array file should exist at {array_path}"

    # Open and verify the file
    with np.load(array_path) as data:
        assert "ACTUAL" in data, "Array file should contain ACTUAL array"
        assert "DESIRED" in data, "Array file should contain DESIRED array"
        assert np.all(data["ACTUAL"] == np.array([1, 2, 3]))
        assert np.all(data["DESIRED"] == np.array([4, 5, 6]))
        assert not isinstance(data["DESIRED"], ApproxBase)


def test_failed_with_no_array_path_set(pytester: pytest.Pytester) -> None:
    """Test that no numpy dump happens when the array path is not set."""
    pytester.makeini(
        """
        [pytest]
        addopts = --report-log=report.json
        raw_data = false
        """
    )
    result = pytester.runpytest("demo.py::test_numpy_fails")
    result.assert_outcomes(failed=1)
    assert not any("Arrays written to" in line for line in result.outlines), (
        "Arrays written to should not be present in the report"
    )
