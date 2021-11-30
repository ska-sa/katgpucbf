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

import operator
from typing import Any, Callable, Sequence, cast
from unittest import mock

import numpy as np
import pyparsing as pp
import pytest

from katgpucbf.dsim import signal
from katgpucbf.dsim.signal import CW, WGN, Signal


class TestCW:
    """Tests for :class:`katgpucbf.dsim.signal.CW`."""

    @pytest.mark.parametrize(
        "amplitude, frequency, timestamp, n",
        [(1.0, 200e6, 0, 4096), (0.1, 500e6, 123456789, 65535), (234.5, 123456789.1, 9876543210, 654321)],
    )
    def test_sample(self, frequency: float, amplitude: float, timestamp: int, n: int) -> None:
        """Test accuracy of basic functionality."""
        adc_sample_rate = 1e9
        cw = CW(amplitude, frequency)
        out = cw.sample(timestamp, n, adc_sample_rate)
        timestamps = np.arange(timestamp, timestamp + n)
        expected = np.cos(timestamps * (frequency / adc_sample_rate * 2 * np.pi)) * amplitude
        np.testing.assert_allclose(out, expected, atol=1e-6 * amplitude)


class TestWGN:
    """Tests for :class:`katgpucbf.dsim.signal.WGN`."""

    def test_stats(self) -> None:
        """Test that the signal has the right mean and standard deviation."""
        wgn = WGN(3.0, 1)
        data = wgn.sample(0, 10000, 1)
        assert np.mean(data) == pytest.approx(0, abs=0.15)  # 5 sigma tolerance
        # Sum of squares should be a chi-squared distribution with n degrees of
        # freedom. The magic numbers are (approximately) generated with
        # scipy.stats.chi2.ppf(1e-6, df=10000, scale=9) and
        # scipy.stats.chi2.isf(1e-6, df=10000, scale=9)
        # (where 9 is the expected population variance). The values are
        # hardcoded to avoid a dependence on scipy just for this test.
        assert 84079 <= np.sum(data ** 2) <= 96180

    def test_timestamp(self) -> None:
        """Test that different timestamps give consistent results."""
        wgn = WGN(3.0, 1)
        data1 = wgn.sample(100, 1000, 1)
        data2 = wgn.sample(150, 1000, 1)
        np.testing.assert_equal(data1[50:], data2[:-50])

    def test_entropy(self) -> None:
        """Test that different instances with the same entropy give the same results."""
        wgn1 = WGN(3.0, 12345)
        wgn2 = WGN(1.5, 12345)
        np.testing.assert_allclose(wgn1.sample(0, 1000, 1), wgn2.sample(0, 1000, 1) * 2)

    def test_non_correlated(self) -> None:
        """Test that instances with different entropy are uncorrelated."""
        wgn1 = WGN(3.0, 1)
        wgn2 = WGN(2.0)
        data1 = wgn1.sample(0, 10000, 1)
        data2 = wgn2.sample(0, 10000, 1)
        # Variance of a product of zero-mean random variables is the product of
        # the individual variances. So the standard deviation of each element of
        # data1 * data2 should be 6.
        assert np.mean(data1 * data2) == pytest.approx(0, abs=0.3)  # 5 sigma


@pytest.mark.parametrize("op, name", [(operator.add, "+"), (operator.sub, "-"), (operator.mul, "*")])
def test_combine(op: Callable[[Any, Any], Any], name: str) -> None:  # noqa: D
    """Test :class:`katgpucbf.dsim.signal.CombinedSignal`."""
    timestamp = 100
    n = 200
    frequency = 1000.0
    sig1 = CW(1.0, 200.0)
    sig2 = WGN(0.5, 123)
    combined = op(sig1, sig2)
    sig1_sample = sig1.sample(timestamp, n, frequency)
    sig2_sample = sig2.sample(timestamp, n, frequency)
    combined_sample = combined.sample(timestamp, n, frequency)
    np.testing.assert_equal(combined_sample, op(sig1_sample, sig2_sample))
    assert str(combined) == f"(cw(1.0, 200.0) {name} wgn(0.5, 123))"


class TestParseSignals:
    """Tests for :func:`katgpucbf.dsim.signal.parse_signals`."""

    @pytest.mark.parametrize(
        "text, expected",
        [
            ("cw(1, 2.0);", [CW(1.0, 2.0)]),
            ("cw(1.5, 2.5); cw(3.0, 2.0);", [CW(1.5, 2.5), CW(3.0, 2.0)]),
            ("c = wgn(1.5, 12345); c; c;", [WGN(1.5, 12345), WGN(1.5, 12345)]),
            ("c = cw(1.5, +2.5e0); c + cw(1, 2);", [CW(1.5, 2.5) + CW(1, 2)]),
            ("cw(1, 2) - cw(3, 4) * cw(5, 6);", [CW(1, 2) - CW(3, 4) * CW(5, 6)]),
            ("wgn(0.5);", [WGN(0.5, mock.ANY)]),
        ],
    )
    def test_success(self, text: str, expected: Signal) -> None:
        """Test result when parsing succeeds."""
        assert signal.parse_signals(text) == expected

    def test_wgn_auto_entropy(self) -> None:
        """Test that wgn with unspecified entropy gets a different value each time."""
        signals = cast(Sequence[WGN], signal.parse_signals("a = wgn(0.5); b = wgn(0.7); a; a; b;"))
        assert isinstance(signals[0].entropy, int)
        assert signals[0].entropy == signals[1].entropy
        assert signals[0].entropy != signals[2].entropy

    @pytest.mark.parametrize("text", ["cw(1, 2) +", "a = cw(1, 2)"])
    def test_truncated_expression(self, text: str) -> None:
        """Test error when an expression is cut off part-way through."""
        with pytest.raises(pp.ParseSyntaxException, match="Expected ';'"):
            signal.parse_signals(text)

    def test_unknown_variable(self) -> None:
        """Test error when using an unknown variable."""
        with pytest.raises(pp.ParseFatalException, match="Unknown variable 'foo'"):
            signal.parse_signals("foo;")

    def test_bad_real(self) -> None:
        """Test error when expecting a number but something else is found."""
        with pytest.raises(pp.ParseSyntaxException, match="Expected number"):
            signal.parse_signals("cw(1, foo);")


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
