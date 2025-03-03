################################################################################
# Copyright (c) 2020-2023, 2025, National Research Foundation (SARAO)
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
import copy
from collections.abc import Sequence
from functools import partial

import numpy as np
import pytest

from katgpucbf.fgpu.delay import (
    AlignedDelayModel,
    LinearDelayModel,
    MultiDelayModel,
    NonMonotonicQueryWarning,
    wrap_angle,
)


@pytest.mark.parametrize(
    "input,output", [(0.0, 0.0), (1.0, 1.0), (1.1, 1.1), (4.0, 4.0 - 2 * np.pi), (10.0, 10.0 - 4 * np.pi)]
)
def test_wrap_angle(input: float, output: float) -> None:
    """Test :func:`.wrap_angle`."""
    assert wrap_angle(input) == pytest.approx(output)
    assert wrap_angle(-input) == pytest.approx(-output)


def mdelay_model_callback(linear_delay_models: Sequence[LinearDelayModel], *, update_list: list) -> None:
    """Test functionality in MultiDelayModel."""
    update_list.append(
        (
            linear_delay_models[0].start,
            linear_delay_models[0].delay,
            linear_delay_models[0].delay_rate,
            linear_delay_models[0].phase,
            linear_delay_models[0].phase_rate,
        )
    )


@pytest.fixture
def mdelay_callback_list() -> list:
    """Create an empty list to populate with delay model values via callback."""
    return []


@pytest.fixture
def linear() -> LinearDelayModel:
    """Create a :class:`.LinearDelayModel` with a fixed set of parameters for testing."""
    return LinearDelayModel(12345, 100.0, 0.25, 0.1, 0.1)


@pytest.fixture
def multi(linear, mdelay_callback_list) -> MultiDelayModel:
    """Create a :class:`.MultiDelayModel` with a fixed set of parameters for testing."""
    out = MultiDelayModel(callback_func=partial(mdelay_model_callback, update_list=mdelay_callback_list))
    # First model is the same as the linear fixture
    out.add(linear)
    out.add(LinearDelayModel(30000, 50.5, -0.0025, 0.5, 0.01))
    out.add(LinearDelayModel(50000, 2000.0, 0.25, -0.5, -0.1))
    return out


@pytest.fixture
def aligned(linear: LinearDelayModel) -> AlignedDelayModel:
    """An :class:`AlignedDelayModel` with even alignment."""
    return AlignedDelayModel(linear, 16)


@pytest.fixture
def aligned_odd(linear: LinearDelayModel) -> AlignedDelayModel:
    """An :class:`AlignedDelayModel` with odd alignment."""
    return AlignedDelayModel(linear, 17)


def test_linear_call(linear: LinearDelayModel) -> None:
    """Test calling the delay model against manually-calculated correct outputs."""
    assert linear(12345) == (12245, 0.0, wrap_angle(0.1))
    assert linear(12790) == (12579, 0.25, pytest.approx(wrap_angle(44.6)))
    assert linear(12788) == (12577, -0.25, pytest.approx(wrap_angle(44.4)))


def test_linear_range(linear: LinearDelayModel) -> None:
    """Test :meth:`.LinearDelayModel.range` against manually-calculated correct outputs."""
    time, residual, phase = linear.range(12999, 13005, 1)
    np.testing.assert_array_equal(time, [12735, 12736, 12737, 12738, 12739, 12739])
    np.testing.assert_array_equal(residual, [-0.5, -0.25, 0.0, 0.25, 0.5, -0.25])
    np.testing.assert_array_almost_equal(phase, wrap_angle(np.array([65.5, 65.6, 65.7, 65.8, 65.9, 66.0])))


@pytest.mark.parametrize(
    "target,start,step",
    [
        (100, 0, 1),
        (100, 12345, 1),
        (100, 12345, 64),
        (12788, 0, 1),
        (12790, 0, 1),
        (200000, 12345, 100),
    ],
)
def test_linear_skip(linear: LinearDelayModel, target: int, start: int, step: int) -> None:
    """Test :meth:`LinearDelayModel.skip`."""
    t = linear.skip(target, start, step)
    assert t >= start
    assert t % step == 0
    assert t - step < start or linear(t - step)[0] < target
    assert linear(t)[0] >= target


def test_linear_stability_delay() -> None:
    """Test that delay is preserved over large timescales."""
    model = LinearDelayModel(10**14, 1000000.001, 0.0, 0.0, 0.0)
    time, residual, phase = model(2 * 10**14)
    assert time == 2 * 10**14 - 1000000
    assert residual == pytest.approx(0.001, abs=1e-9)
    assert phase == 0.0


def test_linear_stability_delay_rate() -> None:
    """Test that delay rate can be accurately applied over long periods."""
    model = LinearDelayModel(10**14, 0.0, 1e-10, 0.0, 0.0)
    time, residual, phase = model(2 * 10**14)
    assert time == 2 * 10**14 - 10**4
    assert residual == 0.0
    assert phase == 0.0


def test_linear_bad_delay_rate() -> None:
    """Delay rate can't be 1 or more."""
    with pytest.raises(ValueError):
        LinearDelayModel(1, 2.0, 1.0, 0.0, 0.0)


def test_multi_add_older(multi) -> None:
    """New delay model must overwrite older ones from the start time."""
    old_models = list(multi._models)
    assert len(old_models) == 4  # Just to check that the fixture wasn't changed
    new_linear = LinearDelayModel(20000, 0.0, 0.0, 0.0, 0.0)
    multi.add(new_linear)
    assert list(multi._models) == [old_models[0], old_models[1], new_linear]


def test_multi_range(multi) -> None:
    """Test :meth:`katgpucbf.fgpu.delay.MultiDelayModel.range`."""
    time, residual, phase = multi.range(0, 60000, 11000)
    np.testing.assert_array_equal(time, [0, 11000, 19486, 32957, 43984, 51750])
    np.testing.assert_allclose(residual, [0.0, 0.0, -0.25, 0.0, -0.5, 0.0], atol=0.01)
    np.testing.assert_allclose(phase, [0.0, 0.0, -2.01, -0.916, 2.270, 2.155], atol=0.01)
    # Warn if queries are not monotonic
    multi.range(50000, 60000, 1)
    with pytest.warns(NonMonotonicQueryWarning):
        multi.range(0, 60000, 11000)


@pytest.mark.parametrize(
    "target,start,step",
    [
        (100, 12345, 1),
        (100000, 12345, 1),
        (30000, 30000, 9),
        (49000, 12345, 7),
        (49000, 60000, 3),
    ],
)
def test_multi_skip(multi: MultiDelayModel, target: int, start: int, step: int) -> None:
    """Test :meth:`.MultiDelayModel.skip`."""
    # skip modifies the model, so make a copy to allow the original to be
    # queried
    orig = copy.deepcopy(multi)
    t = multi.skip(target, start, step)
    assert t >= start
    # Check that it hasn't deleted too much of the model.
    assert multi.skip(target, t, step) == t
    assert t % step == 0
    assert t - step < start or orig(t - step)[0] < target
    assert orig(t)[0] >= target
    # Check that it has deleted as much as it was expected to
    assert len(multi._models) == 1 or t < multi._models[1].start


@pytest.mark.parametrize("model", ["multi", "linear"])
def test_range_empty(request, model) -> None:
    """Test range with an empty range."""
    model = request.getfixturevalue(model)
    time, residual, phase = model.range(0, 0, 1000)
    assert time.shape == (0,)
    assert residual.shape == (0,)
    assert phase.shape == (0,)


def test_aligned_range(linear: LinearDelayModel, aligned: AlignedDelayModel, aligned_odd: AlignedDelayModel) -> None:
    """Test :meth:`.AlignedDelayModel.range`."""
    time1, residual1, phase1 = linear.range(13000, 14000, 3)
    time2, residual2, phase2 = aligned.range(13000, 14000, 3)
    time3, residual3, phase3 = aligned_odd.range(13000, 14000, 3)
    np.testing.assert_allclose(time1 - residual1, time2 - residual2)
    np.testing.assert_allclose(time1 - residual1, time3 - residual3)
    np.testing.assert_allclose(phase1, phase2)
    np.testing.assert_allclose(phase1, phase3)
    # Check that original times are aligned
    np.testing.assert_equal(time2 % 16, 0)
    np.testing.assert_equal(time3 % 17, 0)
    # Check that residuals are not too big
    assert np.max(np.abs(residual2)) <= 8 + 1e-6
    assert np.max(np.abs(residual3)) <= 8.5 + 1e-6


@pytest.mark.parametrize("step", [1, 4, 5, 100])
def test_aligned_skip(aligned: AlignedDelayModel, step: int) -> None:
    """Test :meth:`.AlignedDelayModel.skip`."""
    start = 13100
    for target in range(13000, 13500):
        t = aligned.skip(target, start, step)
        orig = aligned(t)[0]
        assert orig >= target
        assert t % step == 0
        assert orig % 16 == 0
        assert t - step < start or aligned(t - step)[0] < target
