################################################################################
# Copyright (c) 2024, National Research Foundation (SARAO)
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

import pytest

from katgpucbf.meerkat import BANDS


@pytest.mark.requirements("CBF-REQ-0044")
def test_narrowband_centre_frequency() -> None:
    r"""Test the resolution of the narrowband centre frequency.

    Verification method
    -------------------
    Verified by analysis. The F-engine converts the frequency of the DDC mixer
    to a fixed-point representation with 32 fractional bits. The units are
    cycles per output (decimated) sample. The resolution is thus
    :math:`\frac{f}{2^{32} d}`
    where :math:`f` is the ADC sample rate and :math:`d` is the decimation
    factor. For Narrowband Fine L-band,
    :math:`f = 1712 \text{MHz}` and :math:`d = 16`, giving
    a resolution of 0.025 Hz.

    The interfaces which set and report the centre frequency use IEEE-754
    double precision, which has significantly more precision than this and
    is not the limiting factor.
    """
    for band in ["u", "l"]:
        assert BANDS[band].adc_sample_rate / 2**32 / 16 < 100.0
