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

from katgpucbf.fgpu.main import comma_split, parse_args, parse_narrowband
from katgpucbf.fgpu.output import NarrowbandOutput


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


class TestParseNarrowband:
    """Test :func:`.parse_narrowband`."""

    def test_minimal(self) -> None:
        """Test with the minimum required arguments."""
        assert parse_narrowband("channels=1024,decimation=8,dst=239.1.2.3+1:7148") == {
            "channels": 1024,
            "decimation": 8,
            "dst": [Endpoint("239.1.2.3", 7148), Endpoint("239.1.2.4", 7148)],
        }

    def test_maximal(self) -> None:
        """Test with all valid arguments."""
        assert parse_narrowband("channels=1024,decimation=8,taps=8,w_cutoff=0.5,dst=239.1.2.3+1:7148") == {
            "channels": 1024,
            "decimation": 8,
            "taps": 8,
            "w_cutoff": 0.5,
            "dst": [Endpoint("239.1.2.3", 7148), Endpoint("239.1.2.4", 7148)],
        }

    @pytest.mark.parametrize(
        "missing,value",
        [
            ("channels", "decimation=8,dst=239.1.2.3+1:7148"),
            ("decimation", "channels=1024,dst=239.1.2.3+1:7148"),
            ("dst", "channels=1024,decimation=8"),
        ],
    )
    def test_missing_key(self, missing: str, value: str) -> None:
        """Test without one of the required keys."""
        with pytest.raises(ValueError, match=f"--narrowband: {missing} is missing"):
            parse_narrowband(value)

    def test_duplicate_key(self) -> None:
        """Test with a key specified twice."""
        with pytest.raises(ValueError, match="--narrowband: channels specified twice"):
            parse_narrowband("channels=8,channels=9,decimation=8,dst=239.1.2.3+1:7148")


class TestParseArgs:
    """Test :func:`.katgpucbf.fgpu.main.parse_args`."""

    def test_narrowband_defaults(self) -> None:
        """Test that missing narrowband config is taken from the global config."""
        raw_args = [
            "--src-interface=lo",
            "--dst-interface=lo",
            "--adc-sample-rate=1712000000.0",
            "--channels=1024",
            "--sync-epoch=0",
            "--taps=64",
            "--w-cutoff=0.9",
            "--narrowband=dst=239.1.0.0+1,channels=32768,decimation=8,taps=4,w_cutoff=0.8",
            "--narrowband=dst=239.2.0.0+0:7149,channels=8192,decimation=16",
            "239.0.1.0+7:7148",
            "239.0.2.0+7:7148",
            "239.0.3.0+7:7148",
        ]
        args = parse_args(raw_args)
        assert args.narrowband == [
            NarrowbandOutput(
                dst=[Endpoint("239.1.0.0", 7148), Endpoint("239.1.0.1", 7148)],
                channels=32768,
                decimation=8,
                taps=4,
                w_cutoff=0.8,
            ),
            NarrowbandOutput(dst=[Endpoint("239.2.0.0", 7149)], channels=8192, decimation=16, taps=64, w_cutoff=0.9),
        ]