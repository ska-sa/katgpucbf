################################################################################
# Copyright (c) 2023-2025, National Research Foundation (SARAO)
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

from katsdptelstate.endpoint import Endpoint

from katgpucbf.fgpu.main import (
    DEFAULT_DDC_TAPS_RATIO,
    DEFAULT_JONES_PER_BATCH,
    DEFAULT_TAPS,
    DEFAULT_W_CUTOFF,
    DEFAULT_WEIGHT_PASS,
    parse_args,
)
from katgpucbf.fgpu.output import NarrowbandOutputDiscard, NarrowbandOutputNoDiscard, WidebandOutput, WindowFunction
from katgpucbf.utils import DitherType


class TestParseArgs:
    """Test :func:`.katgpucbf.fgpu.main.parse_args`."""

    def test_narrowband_defaults(self) -> None:
        """Test that missing narrowband config is taken from the global config."""
        raw_args = [
            "--recv-interface=lo",
            "--send-interface=lo",
            "--adc-sample-rate=1712000000.0",
            "--sync-time=0",
            (
                "--wideband=name=wideband,dst=239.0.3.0+1:7148,dither=none,"
                "channels=1024,taps=64,w_cutoff=0.9,jones_per_batch=262144"
            ),
            (
                "--narrowband=name=nb0,dst=239.1.0.0+1,channels=32768,"
                "centre_frequency=400e6,decimation=8,taps=4,w_cutoff=0.8,"
                "window_function=hann,ddc_taps=64,weight_pass=0.3,jones_per_batch=524288,"
                "pass_bandwidth=64e6"
            ),
            "--narrowband=name=nb1,dst=239.2.0.0+0:7149,channels=8192,centre_frequency=300e6,decimation=16",
            (
                "--narrowband=name=nb2,dst=239.3.0.0+0:7149,channels=8192,"
                "centre_frequency=300e6,decimation=16,pass_bandwidth=64e6"
            ),
            "239.0.1.0+15:7148",
        ]
        args = parse_args(raw_args)
        assert args.outputs == [
            WidebandOutput(
                name="wideband",
                dst=[Endpoint("239.0.3.0", 7148), Endpoint("239.0.3.1", 7148)],
                dither=DitherType.NONE,
                channels=1024,
                taps=64,
                w_cutoff=0.9,
                window_function=WindowFunction.DEFAULT,
                jones_per_batch=262144,
            ),
            NarrowbandOutputNoDiscard(
                name="nb0",
                dst=[Endpoint("239.1.0.0", 7148), Endpoint("239.1.0.1", 7148)],
                dither=DitherType.DEFAULT,
                channels=32768,
                centre_frequency=400e6,
                decimation=8,
                taps=4,
                w_cutoff=0.8,
                window_function=WindowFunction.HANN,
                ddc_taps=64,
                weight_pass=0.3,
                jones_per_batch=524288,
                pass_bandwidth=64e6,
            ),
            NarrowbandOutputDiscard(
                name="nb1",
                dst=[Endpoint("239.2.0.0", 7149)],
                dither=DitherType.DEFAULT,
                channels=8192,
                centre_frequency=300e6,
                decimation=16,
                taps=DEFAULT_TAPS,
                w_cutoff=DEFAULT_W_CUTOFF,
                window_function=WindowFunction.DEFAULT,
                ddc_taps=DEFAULT_DDC_TAPS_RATIO * 16,
                weight_pass=DEFAULT_WEIGHT_PASS,
                jones_per_batch=DEFAULT_JONES_PER_BATCH,
            ),
            NarrowbandOutputNoDiscard(
                name="nb2",
                dst=[Endpoint("239.3.0.0", 7149)],
                dither=DitherType.DEFAULT,
                channels=8192,
                centre_frequency=300e6,
                decimation=16,
                taps=DEFAULT_TAPS,
                w_cutoff=DEFAULT_W_CUTOFF,
                window_function=WindowFunction.DEFAULT,
                ddc_taps=DEFAULT_DDC_TAPS_RATIO * 32,
                weight_pass=DEFAULT_WEIGHT_PASS,
                jones_per_batch=DEFAULT_JONES_PER_BATCH,
                pass_bandwidth=64e6,
            ),
        ]
