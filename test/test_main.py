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

import argparse

import pytest

from katgpucbf.main import _multi_add_argument, comma_split, parse_dither
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
        with pytest.raises(argparse.ArgumentTypeError, match="invalid int value: 'hello'"):
            assert comma_split(int)("3,hello")

    def test_fixed_count(self) -> None:
        """Test with a value for `count`."""
        splitter = comma_split(int, 2)
        assert splitter("3,5") == [3, 5]
        with pytest.raises(argparse.ArgumentTypeError, match="Expected 2 comma-separated fields, received 3"):
            splitter("3,5,7")
        with pytest.raises(argparse.ArgumentTypeError, match="Expected 2 comma-separated fields, received 1"):
            splitter("3")

    def test_allow_single(self) -> None:
        """Test with `allow_single`."""
        splitter = comma_split(int, 2, allow_single=True)
        assert splitter("3,5") == [3, 5]
        assert splitter("3") == [3, 3]
        with pytest.raises(argparse.ArgumentTypeError, match="Expected 2 comma-separated fields, received 3"):
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
            argparse.ArgumentTypeError,
            match=rf"invalid dither value: {input} \(valid values are 'none', 'uniform'\)",
        ):
            parse_dither(input)


class TestMultiAddArguments:
    """Test the :func:`._multi_add_arguments` helper."""

    @staticmethod
    def _make_parser(multi: bool) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        _multi_add_argument(
            multi,
            parser,
            "--foo",
            default=0,
            type=int,
            metavar="THING{dots}",
            help="Name{s} of the foo{s} [%(default)s]",
        )
        # Variant with no default or explicit type
        _multi_add_argument(
            multi,
            parser,
            "--bar",
            help="Name{s} of the bar{s}",
        )
        return parser

    @pytest.fixture
    def singular_parser(self) -> argparse.ArgumentParser:
        """Parser whose arguments are singular."""
        return self._make_parser(False)

    @pytest.fixture
    def multi_parser(self) -> argparse.ArgumentParser:
        """Parser whose arguments are multi."""
        return self._make_parser(True)

    def test_parse_singular(self, singular_parser: argparse.ArgumentParser) -> None:
        """Test parsing singular arguments."""
        args = singular_parser.parse_args(["--foo=4"])
        assert args.foo == 4
        args = singular_parser.parse_args([])
        assert args.foo == 0
        assert args.bar is None

    def test_parse_multi(self, multi_parser: argparse.ArgumentParser) -> None:
        """Test parsing multi-arguments."""
        args = multi_parser.parse_args(["--foo=4"])
        assert args.foo == [4]
        args = multi_parser.parse_args(["--foo=4,5"])
        assert args.foo == [4, 5]
        args = multi_parser.parse_args([])
        assert args.foo == [0]
        assert args.bar is None

    def test_help_singular(self, singular_parser: argparse.ArgumentParser) -> None:
        """Test the help string for a singular argument."""
        help_text = singular_parser.format_help()
        assert "--foo THING " in help_text
        assert "Name of the foo [0]\n" in help_text

    def test_help_multi(self, multi_parser: argparse.ArgumentParser) -> None:
        """Test the help string for a multi-argument."""
        help_text = multi_parser.format_help()
        assert "--foo THING,... " in help_text
        assert "Name(s) of the foo(s) [0]\n" in help_text
