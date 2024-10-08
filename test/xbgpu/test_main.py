################################################################################
# Copyright (c) 2023-2024, National Research Foundation (SARAO)
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

import pytest
from katsdptelstate.endpoint import Endpoint

from katgpucbf.utils import DitherType
from katgpucbf.xbgpu.main import parse_beam, parse_corrprod
from katgpucbf.xbgpu.output import BOutput, XOutput


class TestParseBeam:
    """Test :func:`.parse_beam`."""

    def test_maximal(self) -> None:
        """Test with all valid arguments."""
        assert parse_beam("name=beam1,dst=239.1.2.3:7148,pol=1,dither=none") == BOutput(
            name="beam1",
            dst=Endpoint("239.1.2.3", 7148),
            pol=1,
            dither=DitherType.NONE,
        )

    def test_minimal(self) -> None:
        """Test with only required arguments."""
        assert parse_beam("name=beam1,dst=239.1.2.3:7148,pol=1") == BOutput(
            name="beam1",
            dst=Endpoint("239.1.2.3", 7148),
            pol=1,
            dither=DitherType.DEFAULT,
        )

    @pytest.mark.parametrize(
        "missing,value",
        [
            ("dst", "name=foo,pol=0"),
            ("name", "dst=239.1.2.3:7148,pol=1"),
            ("pol", "name=foo,dst=239.1.2.3:7148"),
        ],
    )
    def test_missing_key(self, missing: str, value: str) -> None:
        """Test without one of the required keys."""
        with pytest.raises(ValueError, match=f"--beam: {missing} is missing"):
            parse_beam(value)

    def test_duplicate_key(self) -> None:
        """Test with a key specified twice."""
        with pytest.raises(ValueError, match="--beam: name already specified"):
            parse_beam("name=foo,name=bar,dst=239.1.2.3:7148,pol=0")

    def test_invalid_key(self) -> None:
        """Test with an unknown key/value pair."""
        with pytest.raises(ValueError, match="--beam: unknown key fizz"):
            parse_beam("fizz=buzz,name=foo,dst=239.1.2.3:7148,pol=0")

    def test_bad_pol(self) -> None:
        """Test with a polarisation value that isn't 0 or 1."""
        with pytest.raises(ValueError, match="--beam: pol must be either 0 or 1"):
            parse_beam("name=foo,dst=239.1.2.3:7148,pol=2")


class TestParseCorrprod:
    """Test :func:`.parse_corrprod`."""

    def test_maximal(self) -> None:
        """Test with all valid arguments."""
        assert parse_corrprod("name=foo,heap_accumulation_threshold=52,dst=239.2.3.4:7148") == XOutput(
            name="foo",
            heap_accumulation_threshold=52,
            dst=Endpoint("239.2.3.4", 7148),
        )

    @pytest.mark.parametrize(
        "missing,value",
        [
            ("name", "heap_accumulation_threshold=52,dst=239.2.3.4:7148"),
            ("heap_accumulation_threshold", "name=foo,dst=239.2.3.4:7148"),
            ("dst", "name=foo,heap_accumulation_threshold=52"),
        ],
    )
    def test_missing_key(self, missing: str, value: str) -> None:
        """Test without one of the required keys."""
        with pytest.raises(ValueError, match=f"--corrprod: {missing} is missing"):
            parse_corrprod(value)

    def test_duplicate_key(self) -> None:
        """Test with a key specified twice."""
        with pytest.raises(ValueError, match="--corrprod: name already specified"):
            parse_corrprod("name=foo,name=bar,heap_accumulation_threshold=52,dst=239.2.3.4:7148")

    def test_invalid_key(self) -> None:
        """Test with an unknown key/value pair."""
        with pytest.raises(ValueError, match="--corrprod: unknown key fizz"):
            parse_corrprod("fizz=buzz,name=foo,heap_accumulation_threshold=52,dst=239.2.3.4:7148")
