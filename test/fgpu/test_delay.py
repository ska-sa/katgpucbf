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
from collections import Sequence

import numpy as np
import pytest

from katgpucbf.fgpu.delay import LinearDelayModel, MultiDelayModel, NonMonotonicQueryWarning, wrap_angle


@pytest.mark.parametrize(
    "input,output", [(0.0, 0.0), (1.0, 1.0), (1.1, 1.1), (4.0, 4.0 - 2 * np.pi), (10.0, 10.0 - 4 * np.pi)]
)
def test_wrap_angle(input: float, output: float) -> None:
    """Test :func:`.wrap_angle`."""
    assert wrap_angle(input) == pytest.approx(output)
    assert wrap_angle(-input) == pytest.approx(-output)


def mdelay_model_callback(linear_delay_models: Sequence[LinearDelayModel]) -> None:
    """Test functionality in MultiDelayModel."""
    pass


@pytest.fixture
def linear() -> LinearDelayModel:
    """Create a LinearDelayModel with a fixed set of parameters for testing."""
    return LinearDelayModel(12345, 100.0, 0.25, 0.1, 0.1)


@pytest.fixture
def multi(linear) -> MultiDelayModel:
    """Create a MultiDelayModel with a fixed set of parameters for testing."""
    out = MultiDelayModel(callback_func=mdelay_model_callback)
    # First model is the same as the linear fixture
    out.add(linear)
    out.add(LinearDelayModel(30000, 50.5, -0.0025, 0.5, 0.01))
    out.add(LinearDelayModel(50000, 2000.0, 0.25, -0.5, -0.1))
    return out


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
    assert linear.invert(12445) == (12345, 0.0, wrap_angle(0.1))
    assert linear.invert(13001) == pytest.approx((12790, 0.2, wrap_angle(44.58)))
    assert linear.invert(12999) == pytest.approx((12788, -0.2, wrap_angle(44.42)))


def test_linear_invert_range(linear: LinearDelayModel) -> None:
    """Test `invert_range()` against manually-calculated correct outputs."""
    time, residual, phase = linear.invert_range(12999, 13005, 1)
    np.testing.assert_array_equal(time, [12788, 12789, 12790, 12791, 12791, 12792])
    np.testing.assert_array_almost_equal(residual, [-0.2, 0.0, 0.2, 0.4, -0.4, -0.2])
    np.testing.assert_array_almost_equal(phase, wrap_angle(np.array([44.42, 44.50, 44.58, 44.66, 44.74, 44.82])))

    time, residual, _phase = linear.invert_range(13000, 14000, 100)
    exact_time = time - residual
    forward = exact_time + np.apply_along_axis(linear, 0, exact_time)
    np.testing.assert_array_almost_equal(forward, np.arange(13000, 14000, 100))


def test_linear_bad_delay_rate() -> None:
    """Delay rate can't be -1 or less."""
    with pytest.raises(ValueError):
        LinearDelayModel(1, 2.0, -1.0, 0.0, 0.0)


def test_multi_add_older(multi) -> None:
    """New delay model must overwrite older ones from the start time."""
    old_models = list(multi._models)
    assert len(old_models) == 4  # Just to check that the fixture wasn't changed
    new_linear = LinearDelayModel(20000, 0.0, 0.0, 0.0, 0.0)
    multi.add(new_linear)
    assert list(multi._models) == [old_models[0], old_models[1], new_linear]


def test_multi_call(multi) -> None:
    """Test :func:`katgpucbf.fgpu.delay.MultiDelayModel.__call__`."""
    assert multi(100.0) == 0.0
    assert multi(12345.0) == 100.0
    assert multi(12945.75) == 250.1875
    assert multi(50001.0) == 2000.25
    assert len(multi._models) == 1  # Should have popped the older models
    # Going backwards returns unspecified result
    with pytest.warns(NonMonotonicQueryWarning):
        multi(100.0)


def test_multi_invert_range(multi) -> None:
    """Test :func:`katgpucbf.fgpu.delay.MultiDelayModel.invert_range`."""
    time, residual, phase = multi.invert_range(0, 60000, 11000)
    np.testing.assert_array_equal(time, [0, 11000, 19989, 32957, 43984, 52400])
    np.testing.assert_allclose(residual, [0.0, 0.0, 0.0, 0.11, -0.46, 0.0], atol=0.01)
    np.testing.assert_allclose(phase, [0.0, 0.0, -2.05, -1.35, 2.11, -1.74], atol=0.01)
    # Warn if queries are not monotonic
    multi.invert_range(50000, 60000, 1)
    with pytest.warns(NonMonotonicQueryWarning):
        multi.invert_range(0, 60000, 11000)


@pytest.mark.parametrize("model", ["multi", "linear"])
def test_invert_range_empty(request, model) -> None:
    """Test invert_range with an empty range."""
    model = request.getfixturevalue(model)
    time, residual, phase = model.invert_range(0, 0, 1000)
    assert time.shape == (0,)
    assert residual.shape == (0,)
    assert phase.shape == (0,)
