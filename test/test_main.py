################################################################################
# Copyright (c) 2020-2025, National Research Foundation (SARAO)
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

"""Tests for :mod:`katcbfgpu.main`."""

import pytest

from katgpucbf.main import comma_split, parse_dither
from katgpucbf.utils import DitherType


class TestCommaSplit:
    """Test :func:`.comma_split`."""

    def test_basic(self) -> None:
        """Test normal usage, without optional features."""
        assert comma_split(int)("3,5") == [3, 5]
        assert comma_split(int)("3") == [3]
        assert comma_split(int)("") == []

    def test_bad_value(self) -> None:
        """Test with a value that isn't valid for the element type."""
        with pytest.raises(ValueError, match="invalid literal for int"):
            assert comma_split(int)("3,hello")

    def test_fixed_count(self) -> None:
        """Test with a value for `count`."""
        splitter = comma_split(int, 2)
        assert splitter("3,5") == [3, 5]
        with pytest.raises(ValueError, match="Expected 2 comma-separated fields, received 3"):
            splitter("3,5,7")
        with pytest.raises(ValueError, match="Expected 2 comma-separated fields, received 1"):
            splitter("3")

    def test_allow_single(self) -> None:
        """Test with `allow_single`."""
        splitter = comma_split(int, 2, allow_single=True)
        assert splitter("3,5") == [3, 5]
        assert splitter("3") == [3, 3]
        with pytest.raises(ValueError, match="Expected 2 comma-separated fields, received 3"):
            splitter("3,5,7")


class TestParseDither:
    """Test :func:`.parse_dither`."""

    @pytest.mark.parametrize(
        "input, output",
        [("none", DitherType.NONE), ("uniform", DitherType.UNIFORM)],
    )
    def test_success(self, input: str, output: DitherType) -> None:
        """Test with valid inputs."""
        assert parse_dither(input) == output

    @pytest.mark.parametrize("input", ["", "false", "UnIFoRM", "NONE", "default"])
    def test_invalid(self, input: str) -> None:
        """Test with invalid inputs."""
        with pytest.raises(
            ValueError,
            match=rf"Invalid dither value {input} \(valid values are \['none', 'uniform'\]\)",
        ):
            parse_dither(input)
