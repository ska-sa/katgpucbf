################################################################################
# Copyright (c) 2021, National Research Foundation (SARAO)
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

"""Unit tests for signal generation functions."""

import numpy as np

from katgpucbf.dsim import signal


class TestQuantise:
    """Tests for :func:`katgpucbf.dsim.signal.quantise`."""

    DATA = [0.1, -0.2, 0.05, 1.0, -1.0, 1.5, -1.5]

    def test_no_dither(self) -> None:
        """Test without dithering, so that values are exact."""
        out = signal.quantise(self.DATA, 10, dither=False)
        np.testing.assert_equal(out, [51, -102, 26, 511, -511, 511, -511])

    def test_dither(self) -> None:
        """Test with dithering.

        The random nature of the dithering is not tested; just that the
        output values are in the expected range.
        """
        out = signal.quantise(self.DATA, 10)
        low = [51, -103, 25, 510, -511, 511, -511]
        high = [52, -102, 26, 511, -510, 511, -511]
        for i in range(len(self.DATA)):
            assert low[i] <= out[i] <= high[i]
