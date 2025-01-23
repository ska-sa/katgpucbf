################################################################################
# Copyright (c) 2024-2025, National Research Foundation (SARAO)
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

"""Test specific features of narrowband."""

import numpy as np
import pytest

from katgpucbf.meerkat import BANDS


@pytest.mark.requirements("CBF-REQ-0044")
def test_narrowband_centre_frequency() -> None:
    r"""Test the resolution of the narrowband centre frequency.

    Verification method
    -------------------
    Verified by analysis. The F-engine converts the text representation
    of the centre frequency to double-precision floating point. The
    resolution is thus determined by float-point accuracy and depends on
    the magnitude of the value. The magnitude is limited to 856 MHz
    (since the value is specified at baseband) and at this magnitude,
    1 ULP is approximately :math:`1.2\times 10^{-7}` Hz.
    """
    for band in ["u", "l"]:
        max_centre_frequency = BANDS[band].adc_sample_rate / 2
        resolution = max_centre_frequency - np.nextafter(max_centre_frequency, 0.0)
        assert resolution < 100.0
