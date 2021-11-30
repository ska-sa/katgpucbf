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
import pytest

from katgpucbf.dsim import signal


class TestCW:
    """Tests for :class:`katgpucbf.dsim.signal.CW`."""

    @pytest.mark.parametrize(
        "amplitude, frequency, timestamp, n",
        [(1.0, 200e6, 0, 4096), (0.1, 500e6, 123456789, 65535), (234.5, 123456789.1, 9876543210, 654321)],
    )
    def test_sample(self, frequency: float, amplitude: float, timestamp: int, n: int) -> None:
        """Test accuracy of basic functionality."""
        adc_sample_rate = 1e9
        cw = signal.CW(amplitude, frequency)
        out = cw.sample(timestamp, n, adc_sample_rate)
        timestamps = np.arange(timestamp, timestamp + n)
        expected = np.cos(timestamps * (frequency / adc_sample_rate * 2 * np.pi)) * amplitude
        np.testing.assert_allclose(out, expected, atol=1e-6 * amplitude)


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


class TestPackbits:
    """Tests for :func:`katgpucbf.dsim.signal.packbits`."""

    def test_basic(self):
        """Test normal usage."""
        # Some arbitrary values
        data = [0b10110, 0b01011, 0b11111, 0b10101, 0b00110, 0b00010, 0b11011, 0b11000]
        expected = [0b10110010, 0b11111111, 0b01010011, 0b00001011, 0b01111000]
        np.testing.assert_equal(signal.packbits(data, 5), expected)

    @pytest.mark.parametrize("bits", range(1, 32))
    def test_bits(self, bits: int):
        """Test with a variety of bit counts."""
        rng = np.random.default_rng(1)
        data = rng.integers(0, 2 ** bits, dtype=np.int32, size=200)
        packed = signal.packbits(data, bits)
        # Unpack and check that the original data is returned
        big_int = int.from_bytes(packed, byteorder="big")
        unpacked = []
        for _ in range(len(data)):
            unpacked.append(big_int & (2 ** bits - 1))
            big_int >>= bits
        unpacked.reverse()
        np.testing.assert_equal(data, unpacked)

    def test_partial_bytes(self):
        """Test that :exc:`ValueError` is raised if the data does not form a whole number of bytes."""
        with pytest.raises(ValueError):
            signal.packbits([0], 10)
