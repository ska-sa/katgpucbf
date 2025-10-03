################################################################################
# Copyright (c) 2025, National Research Foundation (SARAO)
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

"""Unit tests for argument parsing."""

import argparse
from typing import NoReturn

import pytest

from katgpucbf.vgpu.main import parse_args

REQUIRED_ARGS = ["1.1.1.1", "2.2.2.2", "3.3.3.3"]


class CustomArgumentParser(argparse.ArgumentParser):
    """Override ArgumentParser behaviour to make it more suitable for tests.

    All arguments become optional (but note that main.parse_args is not
    guaranteed to handle this correctly). Errors become exceptions.
    """

    def add_argument(self, *args, **kwargs):  # noqa: D102
        kwargs.pop("required", None)
        return super().add_argument(*args, **kwargs)

    def error(self, message: str) -> NoReturn:  # noqa: D102
        raise RuntimeError(message)


class TestParsePols:
    """Test error handling in the parsing of polarisation arguments."""

    @pytest.fixture(autouse=True)
    def patch_argument_parser(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Replace :class:`argparse.ArgumentParser` with our custom subclass."""
        monkeypatch.setattr("katgpucbf.vgpu.main._ARGUMENT_PARSER", CustomArgumentParser)

    def test_ok_no_prefixes(self) -> None:
        """Test success case when --recv-pol has no ± prefixes."""
        args = parse_args(["--recv-pol=x,y", "--send-pol=y,x"] + REQUIRED_ARGS)
        assert args.recv_pols == ["x", "y"]
        assert args.send_pols == ["y", "x"]

    def test_ok_prefixes(self) -> None:
        """Test success case when --recv-pol has ± prefixes."""
        args = parse_args(["--recv-pol=+x,-y", "--send-pol=y,x"] + REQUIRED_ARGS)
        assert args.recv_pols == ["+x", "-y"]
        assert args.send_pols == ["y", "x"]

    @pytest.mark.parametrize("pol", ["z", "X", "l", "-", "xx"])
    def test_recv_invalid_pol(self, pol: str) -> None:
        """Test error when --recv-pol has a component that is not a recognised polarisation."""
        with pytest.raises(RuntimeError, match=f"'{pol}' is not a valid --recv-pol value"):
            parse_args([f"--recv-pol=x,{pol}", "--send-pol=y,x"] + REQUIRED_ARGS)

    @pytest.mark.parametrize("pol", ["z", "X", "l", "-", "-x"])
    def test_send_invalid_letter(self, pol: str) -> None:
        """Test error when --send-pol has letter that is not a recognised polarisation."""
        with pytest.raises(RuntimeError, match=f"'{pol}' is not a valid --send-pol value"):
            parse_args(["--recv-pol=x,y", f"--send-pol={pol},x"] + REQUIRED_ARGS)

    @pytest.mark.parametrize("arg", ["x,x", "+x,-x", "x,L", "R,y", "R,R"])
    def test_recv_non_orthogonal(self, arg: str) -> None:
        """Test error when --recv-pol is given a non-orthogonal basis."""
        with pytest.raises(RuntimeError, match="--recv-pol is not an orthogonal polarisation basis"):
            parse_args([f"--recv-pol={arg}", "--send-pol=x,y"] + REQUIRED_ARGS)

    def test_recv_bad_count(self) -> None:
        """Test error when --recv-pol is given the wrong number of elements."""
        with pytest.raises(RuntimeError, match="argument --recv-pols: Expected 2 comma-separated fields, received 3"):
            parse_args(["--recv-pol=x,y,L", "--send-pol=x,y"] + REQUIRED_ARGS)

    def test_send_bad_count(self) -> None:
        """Test error when --recv-pol is given the wrong number of elements."""
        with pytest.raises(RuntimeError, match="argument --send-pols: Expected 2 comma-separated fields, received 1"):
            parse_args(["--recv-pol=x,y", "--send-pol=x"] + REQUIRED_ARGS)
