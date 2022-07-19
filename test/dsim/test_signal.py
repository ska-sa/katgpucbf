################################################################################
# Copyright (c) 2021-2022, National Research Foundation (SARAO)
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
from typing import Any, Callable, List, Sequence, cast
from unittest import mock

import dask.array as da
import numpy as np
import pyparsing as pp
import pytest
import xarray as xr

from katgpucbf.dsim import signal
from katgpucbf.dsim.signal import CW, WGN, Constant, Delay, Nodither, Signal, TerminalError

from .. import unpackbits


@pytest.fixture(autouse=True)
def small_chunks(monkeypatch) -> None:
    """Reduce CHUNK_SIZE for the duration of the test.

    This allows testing with multiple chunks without needing a huge array.
    """
    monkeypatch.setattr(signal, "CHUNK_SIZE", 512)


class TestConstant:
    """Tests for :class:`katgpucbf.dsim.signal.Constant`."""

    @pytest.mark.parametrize("n", [123, 4096, 12345])
    def test_sample(self, n: int) -> None:
        """Test basic functionality."""
        sig = Constant(3.5)
        expected = np.full(n, 3.5, np.float32)
        np.testing.assert_equal(sig.sample(n, 1e9), expected)


class TestCW:
    """Tests for :class:`katgpucbf.dsim.signal.CW`."""

    @pytest.mark.parametrize(
        "amplitude, frequency, n",
        [(1.0, 200e6, 4096), (0.1, 500e6, 65535), (234.5, 123456789.1, 654321)],
    )
    def test_sample(self, frequency: float, amplitude: float, n: int) -> None:
        """Test accuracy of basic functionality."""
        adc_sample_rate = 1e9
        cw = CW(amplitude, frequency)
        out = cw.sample(n, adc_sample_rate)
        timestamps = np.arange(0, n)
        expected = np.cos(timestamps * (frequency / adc_sample_rate * 2 * np.pi)) * amplitude
        np.testing.assert_allclose(out, expected, atol=1e-6 * amplitude)


class TestWGN:
    """Tests for :class:`katgpucbf.dsim.signal.WGN`."""

    def test_stats(self) -> None:
        """Test that the signal has the right mean and standard deviation."""
        wgn = WGN(3.0, 1)
        data = wgn.sample(10000, 1)
        assert np.mean(data) == pytest.approx(0, abs=0.15)  # 5 sigma tolerance
        # Sum of squares should be a chi-squared distribution with n degrees of
        # freedom. The magic numbers are (approximately) generated with
        # scipy.stats.chi2.ppf(1e-6, df=10000, scale=9) and
        # scipy.stats.chi2.isf(1e-6, df=10000, scale=9)
        # (where 9 is the expected population variance). The values are
        # hardcoded to avoid a dependence on scipy just for this test.
        assert 84079 <= np.sum(data**2) <= 96180

    def test_entropy(self) -> None:
        """Test that different instances with the same entropy give the same results."""
        wgn1 = WGN(3.0, 12345)
        wgn2 = WGN(1.5, 12345)
        np.testing.assert_allclose(wgn1.sample(1000, 1), wgn2.sample(1000, 1) * 2)

    def test_non_correlated(self) -> None:
        """Test that instances with different entropy are uncorrelated."""
        wgn1 = WGN(3.0, 1)
        wgn2 = WGN(2.0)
        data1 = wgn1.sample(10000, 1)
        data2 = wgn2.sample(10000, 1)
        # Variance of a product of zero-mean random variables is the product of
        # the individual variances. So the standard deviation of each element of
        # data1 * data2 should be 6.
        assert np.mean(data1 * data2) == pytest.approx(0, abs=0.3)  # 5 sigma


class TestDelay:
    """Tests for :class:`katgpucbf.dsim.signal.Delay`."""

    # Parameter values are all equivalent, but test wraparound
    @pytest.mark.parametrize("d", [50, 100050, -99950])
    def test_delay(self, d: int) -> None:
        """Test basic functionality."""
        wgn = WGN(3.0, 1)
        delay = Delay(wgn, d)
        orig = wgn.sample(100000, 1).compute()
        delayed = delay.sample(100000, 1).compute()
        np.testing.assert_equal(orig[:-50], delayed[50:])
        np.testing.assert_equal(orig[-50:], delayed[:50])


@pytest.mark.parametrize("op, name", [(operator.add, "+"), (operator.sub, "-"), (operator.mul, "*")])
def test_combine(op: Callable[[Any, Any], Any], name: str) -> None:  # noqa: D
    """Test :class:`katgpucbf.dsim.signal.CombinedSignal`."""
    n = 200
    frequency = 1000.0
    sig1 = CW(1.0, 200.0)
    sig2 = WGN(0.5, 123)
    combined = op(sig1, sig2)
    sig1_sample = sig1.sample(n, frequency)
    sig2_sample = sig2.sample(n, frequency)
    combined_sample = combined.sample(n, frequency)
    np.testing.assert_equal(combined_sample.compute(), op(sig1_sample, sig2_sample).compute())
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
            ("cw(1, 2) + 0.25;", [CW(1, 2) + Constant(0.25)]),
            ("delay(cw(1, 2) - wgn(1.5, 12345), 42);", [Delay(CW(1, 2) - WGN(1.5, 12345), 42)]),
            ("delay(1, -123);", [Delay(Constant(1), -123)]),
            ("nodither(cw(1, 2));", [Nodither(CW(1.0, 2.0))]),
            ("x = nodither(1); x; x;", [Nodither(Constant(1)), Nodither(Constant(1))]),
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

    @pytest.mark.parametrize(
        "text",
        [
            "nodither(1) + 0;",
            "x = nodither(1); x + 0;",
            "delay(nodither(1), 1);",
            "nodither(nodither(1));",
        ],
    )
    def test_nodither_terminal(self, text: str) -> None:
        """Test that nodither expressions cannot be used in larger expressions."""
        with pytest.raises(TerminalError, match=r"nodither\(1\.0\)"):
            signal.parse_signals(text)


class TestMakeDither:
    """Tests for :func:`katgpucbf.dsim.signal.make_dither`."""

    def test_no_seed(self) -> None:
        """Test operation with no seed given."""
        dither = signal.make_dither(2, 5000)
        assert dither.sizes["pol"] == 2
        assert dither.sizes["data"] == 5000
        assert dither.dtype == np.float32
        assert dither.chunks is not None
        assert dither.chunks[1][0] == signal.CHUNK_SIZE
        computed = dither.values
        # Very simple test that it's generating the distribution we expect.
        assert np.all(-0.5 <= computed)
        assert np.all(computed <= 0.5)
        assert np.any(computed < -0.4)
        assert np.any(computed > 0.4)
        # Since no seed was used, must get a different result next time
        dither2 = signal.make_dither(2, 5000)
        assert not np.all(computed == dither2.values)

    def test_seed(self) -> None:
        """Test that using a seed gives reproducible results."""
        dither1 = signal.make_dither(2, 5000, 12345).values
        dither2 = signal.make_dither(2, 5000, 12345).values
        np.testing.assert_equal(dither1, dither2)


class TestQuantise:
    """Tests for :func:`katgpucbf.dsim.signal.quantise`."""

    DATA = [0.1, -0.2, 0.05, 1.0, -1.0, 1.5, -1.5]
    DATA_DASK = da.from_array(DATA, chunks=3)
    DITHER = [0.5, -0.4, 0.3, -0.4, -0.4, 0.1, 0.2]
    DITHER_DASK = da.from_array(DITHER, chunks=4)  # Different chunk size to exercise that

    def test_no_dither(self) -> None:
        """Test without dithering, so that values are exact."""
        out = signal.quantise(self.DATA_DASK, 10, dither=self.DITHER_DASK * 0).compute()
        np.testing.assert_equal(out, [51, -102, 26, 511, -511, 511, -511])

    def test_dither(self) -> None:
        """Test with dithering."""
        out = signal.quantise(self.DATA_DASK, 10, dither=self.DITHER_DASK).compute()
        np.testing.assert_equal(out, [52, -103, 26, 511, -511, 511, -511])


class TestPackbits:
    """Tests for :func:`katgpucbf.dsim.signal.packbits`."""

    def test_basic(self) -> None:
        """Test normal usage."""
        # Some arbitrary values
        data = da.from_array([0b10110, 0b01011, 0b11111, 0b10101, 0b00110, 0b00010, 0b11011, 0b11000])
        expected = [0b10110010, 0b11111111, 0b01010011, 0b00001011, 0b01111000]
        np.testing.assert_equal(signal.packbits(data, 5).compute(), expected)

    @pytest.mark.parametrize("bits", range(2, 32))
    def test_bits(self, bits: int) -> None:
        """Test with a variety of bit counts."""
        rng = np.random.default_rng(1)
        data = da.from_array(rng.integers(-(2 ** (bits - 1)), 2 ** (bits - 1) - 1, dtype=np.int32, size=200), chunks=64)
        packed = signal.packbits(data, bits).compute()
        # Unpack and check that the original data is returned
        unpacked = unpackbits(packed, bits)
        np.testing.assert_equal(data.compute(), unpacked)

    def test_partial_bytes(self) -> None:
        """Test that :exc:`ValueError` is raised if the data does not form a whole number of bytes."""
        with pytest.raises(ValueError):
            signal.packbits(da.from_array([0]), 10)

    def test_chunk_boundaries(self) -> None:
        """Test that correct results are returned even if chunks aren't nicely aligned."""
        data = da.arange(-512, 512, dtype=np.int32, chunks=32)
        packed = signal.packbits(data, 10).compute()
        unpacked = unpackbits(packed, 10)
        np.testing.assert_equal(data.compute(), unpacked)


class TestSample:
    """Tests for :func:`katgpucbf.dsim.signal.sample`.

    These are mostly smoke tests for the integration of the pieces and the
    different ways it can be called, since most of the functionality is tested
    at lower levels.
    """

    @pytest.fixture
    def out(self) -> xr.DataArray:
        """Output array with 2 pols, 4 heaps and 4096 samples per heap."""
        raw = np.zeros((2, 4, 5120), np.uint8)
        return xr.DataArray(raw, dims=["pol", "time", "data"])

    @pytest.fixture
    def signals(self) -> List[Signal]:
        """Dual-pol signals."""
        return signal.parse_signals("wgn(0.2, 1); wgn(0.5, 2);")

    def test_bad_period(self, signals: List[Signal], out: xr.DataArray) -> None:
        """Test that an error is raised if `period` does not divide into data size."""
        with pytest.raises(ValueError):
            signal.sample(signals, 0, 17, 1.0, 10, out)

    def test_period(self, signals: List[Signal], out: xr.DataArray) -> None:
        """Test that a short period is respected."""
        signal.sample(signals, 0, 4096, 1.0, 10, out)
        # Each heap should be identical
        for i in range(out.sizes["time"]):
            np.testing.assert_array_equal(out.isel(time=i), out.isel(time=0))
        # Period shouldn't be shorter then requested
        assert not (out.isel(time=0, data=np.s_[:2560]) == out.isel(time=0, data=np.s_[2560:])).all()

    def test_period_none(self, signals: List[Signal], out: xr.DataArray) -> None:
        """Test that specifying the period as None works."""
        signal.sample(signals, 0, None, 1.0, 10, out)
        # Must not have a shorter period
        assert not (out.isel(time=np.s_[:2]) == out.isel(time=np.s_[2:])).all()

    def test_nodither_signals(self, signals: List[Signal], out: xr.DataArray) -> None:
        """Test that ``nodither()`` signals disable dither."""
        signal.sample(signals, 0, None, 1.0, 10, out, dither=False)
        orig = out.copy()
        out.data.fill(0)  # To ensure that it is actually updated
        signal.sample([Nodither(signal) for signal in signals], 0, None, 1.0, 10, out)
        np.testing.assert_equal(orig.data, out.data)
