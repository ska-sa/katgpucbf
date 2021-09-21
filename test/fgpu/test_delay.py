################################################################################
# Copyright (c) 2020-2021, National Research Foundation (SARAO)
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

"""Unit tests for DelayModel functions."""
import numpy as np
import pytest

from katgpucbf.fgpu.delay import LinearDelayModel


@pytest.fixture
def linear() -> LinearDelayModel:
    """Create a LinearDelayModel with a fixed set of parameters for testing."""
    return LinearDelayModel(12345, 100.0, 0.25, 0.1, 0.1)


def test_linear_call(linear: LinearDelayModel) -> None:
    """Test the calling of the DelayModel, monotonically.

    .. todo:: Try to the model in unexpected ways to confirm it handles failures gracefully?
    """
    assert linear(12000.0) == 13.75
    assert linear(12345.0) == 100.0
    assert linear(12345.5) == 100.125
    assert linear(12945.5) == 250.125
    assert linear(12945.75) == 250.1875


def test_linear_invert(linear: LinearDelayModel) -> None:
    """Test `invert()` against manually-calculated correct outputs."""
    assert linear.invert(12445) == (12345, 0.0, 10.1)
    assert linear.invert(13001) == pytest.approx((12790, 0.2, 65.7))
    assert linear.invert(12999) == pytest.approx((12788, -0.2, 65.5))


def test_linear_invert_range(linear: LinearDelayModel) -> None:
    """Test `invert_range()` against manually-calculated correct outputs."""
    time, residual, phase = linear.invert_range(12999, 13005, 1)
    np.testing.assert_array_equal(time, [12788, 12789, 12790, 12791, 12791, 12792])
    np.testing.assert_array_almost_equal(residual, [-0.2, 0.0, 0.2, 0.4, -0.4, -0.2])
    np.testing.assert_array_almost_equal(phase, [65.5, 65.6, 65.7, 65.8, 65.9, 66.0])

    time, residual, _phase = linear.invert_range(13000, 14000, 100)
    exact_time = time - residual
    forward = exact_time + np.apply_along_axis(linear, 0, exact_time)
    np.testing.assert_array_almost_equal(forward, np.arange(13000, 14000, 100))
