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

from katgpucbf.xbgpu.main import parse_bengine


class TestParseBengine:
    """Test :func:`.parse_bengine`."""

    def test_maximal(self) -> None:
        """Test with the required arguments."""
        assert parse_bengine(
            "beams=2,channels_per_substream=512,send_rate_factor=1.0,spectra_per_heap=256,dst=239.1.2.3+1:7148"
        ) == {
            "beams": 2,
            "channels_per_substream": 512,
            "dst": [Endpoint("239.1.2.3", 7148), Endpoint("239.1.2.4", 7148)],
            "send_rate_factor": 1.0,
            "spectra_per_heap": 256,
        }

    def test_beamcount_mismatch(self) -> None:
        """Test where number of beams does not match dst addresses."""
        with pytest.raises(
            ValueError, match="--beamformer: Mismatch in number of beams and dest multicast address range."
        ):
            parse_bengine(
                "beams=2,channels_per_substream=512,send_rate_factor=1.0,spectra_per_heap=256,dst=239.1.2.3+7:7148"
            )

    @pytest.mark.parametrize(
        "missing,value",
        [
            ("dst", "beams=8,channels_per_substream=512,send_rate_factor=1.0,spectra_per_heap=256"),
            ("beams", "channels_per_substream=512,spectra_per_heap=256,dst=239.1.2.3+7:7148"),
        ],
    )
    def test_missing_key(self, missing: str, value: str) -> None:
        """Test without one of the required keys."""
        with pytest.raises(ValueError, match=f"--beamformer: {missing} is missing"):
            parse_bengine(value)

    def test_duplicate_key(self) -> None:
        """Test with a key specified twice."""
        with pytest.raises(ValueError, match="--beamformer: channels_per_substream specified twice"):
            parse_bengine("beams=8,channels_per_substream=8,channels_per_substream=9,dst=239.1.2.3+7:7148")
