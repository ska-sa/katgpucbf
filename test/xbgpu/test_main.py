################################################################################
# Copyright (c) 2023, National Research Foundation (SARAO)
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

from katgpucbf.xbgpu.main import parse_beam


class TestParseBeam:
    """Test :func:`.parse_beam`."""

    def test_required_only(self) -> None:
        """Test with just required arguments."""
        assert parse_beam("name=beam1,dst=239.1.2.3:7148") == {
            "name": "beam1",
            "dst": Endpoint("239.1.2.3", 7148),
        }

    def test_maximal(self) -> None:
        """Test with all valid arguments."""
        assert parse_beam("name=beam1,channels_per_substream=512,spectra_per_heap=256,dst=239.1.2.3:7148") == {
            "name": "beam1",
            "channels_per_substream": 512,
            "dst": Endpoint("239.1.2.3", 7148),
            "spectra_per_heap": 256,
        }

    @pytest.mark.parametrize(
        "missing,value",
        [
            ("dst", "name=foo,channels_per_substream=512,spectra_per_heap=256"),
            ("name", "channels_per_substream=512,spectra_per_heap=256,dst=239.1.2.3:7148"),
        ],
    )
    def test_missing_key(self, missing: str, value: str) -> None:
        """Test without one of the required keys."""
        with pytest.raises(ValueError, match=f"--beam: {missing} is missing"):
            parse_beam(value)

    def test_duplicate_key(self) -> None:
        """Test with a key specified twice."""
        with pytest.raises(ValueError, match="--beam: channels_per_substream specified twice"):
            parse_beam("name=food,channels_per_substream=8,channels_per_substream=9,dst=239.1.2.3:7148")
