################################################################################
# Copyright (c) 2020-2023, National Research Foundation (SARAO)
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

"""A collection of classes and methods for delay-tracking.

It should be noted that the classes in this module use a slightly different
model than the public katcp interface. The reference channel for phase change
is channel 0, rather than the centre channel. The difference is dealt with
by the request handler for the ``?delays`` katcp request.
"""

import math
import warnings
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable, Sequence
from typing import Generic, TypeVar

import numpy as np

_DM = TypeVar("_DM", bound="AbstractDelayModel")


def wrap_angle(angle):
    """Restrict an angle to [-pi, pi].

    This works on both Python scalars and numpy arrays.
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _div_up(x: int, step: int) -> int:
    """Divide `x` by `step` and round up."""
    return (x + step - 1) // step


def _round_up(x: int, step: int) -> int:
    """Round `x` up to the next multiple of `step`."""
    return (x + step - 1) // step * step


class NonMonotonicQueryWarning(UserWarning):
    """Delay model was queried non-monotonically."""


class AbstractDelayModel(ABC):
    """Abstract base class for delay models.

    All units are samples rather than SI units.
    """

    @abstractmethod
    def range(self, start: int, stop: int, step: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Find input timestamps corresponding to a range of output samples.

        For each output sample with timestamp in ``range(start, stop, step)``,
        it determines a corresponding input sample.

        Parameters
        ----------
        start
            First timestamp (inclusive).
        stop
            Last timestamp (exclusive)
        step
            Interval between timestamps (must be positive).

        Returns
        -------
        orig_time
            Undelayed integer timestamps corresponding to ``range(start, stop, step)``
        residual
            Fractional sample delay not accounted for by ``time - orig_time``.
        phase
            Fringe-stopping phase to be added.
        """

    def __call__(self, time: int) -> tuple[int, float, float]:
        """Find input sample timestamp corresponding to a given output sample.

        Parameters
        ----------
        time
            Delayed timestamp.

        Returns
        -------
        orig_time
            Undelayed integer timestamp corresponding to `time`.
        residual
            Fractional sample delay not accounted for by ``time - orig_time``.
        phase
            Fringe-stopping phase to be added.
        """
        orig_time, residual, phase = self.range(time, time + 1, 1)
        return int(orig_time[0]), float(residual[0]), float(phase[0])

    @abstractmethod
    def skip(self, target: int, start: int, step: int) -> int:
        """Find the next output time for which the input time is at least `target`.

        The output time must also be at least `start` and a multiple of `step`.
        """


class LinearDelayModel(AbstractDelayModel):
    """Delay model that adjusts delay linearly over time.

    Parameters
    ----------
    start
        Output sample at which the model should start being used.
    delay
        Delay to apply at `start`. [seconds]
    delay_rate
        Rate of change of delay. [seconds/second]
    phase
        Fringe-stopping phase to apply with the fine delay. [radians]
    phase_rate
        Rate of change of the fringe-stopping phase. [radians/second]

    Raises
    ------
    ValueError
        if `rate` is greater than or equal to 1 or `start` is negative
    """

    def __init__(self, start: int, delay: float, delay_rate: float, phase: float, phase_rate: float) -> None:
        if delay_rate >= 1.0:
            raise ValueError("delay rate must be less than 1")
        self.start = start
        self.delay = float(delay)
        self.delay_rate = float(delay_rate)
        self.phase = wrap_angle(float(phase))
        self.phase_rate = float(phase_rate)

    def range(self, start: int, stop: int, step: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # noqa: D102
        # Variables with names prefixed rel_ treat start of delay model as t_0.
        # This makes it easier to apply the rate and reduces rounding errors.
        rel_time = np.arange(start - self.start, stop - self.start, step, dtype=np.dtype(np.int64))
        delay = self.delay + rel_time * self.delay_rate
        coarse_delay = np.rint(delay).astype(np.int64)
        fine_delay = delay - coarse_delay

        # Calculate the phase
        phase = wrap_angle(rel_time * self.phase_rate + self.phase)

        # add self.start back again to return the timestamps in the original
        # epoch
        return rel_time - coarse_delay + self.start, fine_delay, phase

    def skip(self, target: int, start: int, step: int) -> int:  # noqa: D102
        # Let r be the output time relative to self.start.
        # Solve: r - delay - r * delay_rate > (target - self.start) - 0.5
        # <=> r * (1 - delay_rate) > target - self.start + delay - 0.5
        r = math.ceil((target - self.start + self.delay - 0.5) / (1.0 - self.delay_rate))
        t = max(r + self.start, start)
        t = _round_up(t, step)
        # Floating-point gremlins means we can't be 100% sure that evaluating
        # at time t will generate a (rounded) input timestamp that is >=
        # target, but we should be close. If delay_rate is extremely close to 1
        # then this could be expensive, but that's not expected in practice.
        while self(t)[0] < target:
            t += step
        return t


class MultiDelayModel(AbstractDelayModel):
    """Piece-wise linear delay model.

    The model evolves over time by calling :meth:`add`. It **must** only be
    queried with monotonically increasing `start` values, because as soon as a
    query is made beyond the end of the first piece it is discarded.
    Additionally, after calling :meth:`skip`, the return value should be
    treated as a lower bound for future `start` values.

    In the initial state it has a model with zero delay.

    It accepts an optional callback function that takes in the
    LinearDelayModels attached to this MultiDelayModel. This callback is
    called whenever the first linear piece changes. It is also called
    immediately by the constructor.
    """

    def __init__(self, callback_func: Callable[[Sequence[LinearDelayModel]], None] | None = None) -> None:
        # The initial time is -1 rather than 0 so that it doesn't get removed
        # if a model is added with start time 0, which can lead to some
        # spurious warnings in unit tests about non-monotonic queries.
        # Ideally it would use -infinity (or a large negative number), but
        # that causes issues with numeric precision.
        self._models = deque([LinearDelayModel(-1, 0.0, 0.0, 0.0, 0.0)])
        self.callback_func = callback_func
        if callback_func is not None:
            callback_func(self._models)

    def _popleft(self) -> None:
        """Carry out a popleft and callback invocation."""
        self._models.popleft()
        if self.callback_func is not None:
            self.callback_func(self._models)

    def _popleft_until(self, start: int) -> None:
        """Pop models that pre-date `start`."""
        if start < self._models[0].start:
            warnings.warn(
                f"Timestamp {start} is before start of first linear model "
                f"at {self._models[0].start} - possibly due to non-monotonic queries",
                NonMonotonicQueryWarning,
            )
        while len(self._models) > 1 and start >= self._models[1].start:
            self._popleft()

    def range(self, start: int, stop: int, step: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # noqa: D102
        self._popleft_until(start)
        assert step > 0
        n = len(range(start, stop, step))
        # Allocate space to hold the combined results from all subqueries
        orig = np.zeros(n, np.dtype(np.int64))
        fine_delay = np.zeros(n, np.dtype(np.float64))
        phase = np.zeros(n, np.dtype(np.float64))
        pos = 0  # Number of entries that have been filled in so far
        for i, model in enumerate(self._models):
            if pos == n:
                break  # We've filled everything in already, so no need to continue
            # Find first position in the query which belongs to the next model
            if i + 1 == len(self._models):
                next_pos = n
            else:
                next_pos = min(n, _div_up(self._models[i + 1].start - start, step))
            if next_pos > pos:
                sub_orig, sub_fine_delay, sub_phase = model.range(start + pos * step, start + next_pos * step, step)
                orig[pos:next_pos] = sub_orig
                fine_delay[pos:next_pos] = sub_fine_delay
                phase[pos:next_pos] = sub_phase
                pos = next_pos
        return orig, fine_delay, phase

    def skip(self, target: int, start: int, step: int) -> int:  # noqa: D102
        self._popleft_until(start)
        assert step > 0
        while True:
            t = self._models[0].skip(target, start, step)
            if len(self._models) == 1 or t < self._models[1].start:
                # The returned time is within the domain of the first
                # linear model.
                return t
            # If we get here, no timestamp within the domain of the first
            # linear model is satisfactory.
            self._popleft()
            start = self._models[0].start

    def add(self, model: LinearDelayModel) -> None:
        """Extend the model with a new linear model.

        The new model is applicable from its start time forever. If the new
        model has an earlier start time than some previous model, the previous
        model will be discarded.
        """
        while self._models and model.start <= self._models[-1].start:
            self._models.pop()
        self._models.append(model)
        if len(self._models) == 1 and self.callback_func is not None:
            self.callback_func(self._models)


class AlignedDelayModel(AbstractDelayModel, Generic[_DM]):
    """Wrap another delay model and enforce an alignment on original timestamp.

    Note that this can cause residual delays to be larger than 1.
    """

    def __init__(self, base: _DM, align: int) -> None:
        self.base = base
        self.align = align

    def range(self, start: int, stop: int, step: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # noqa: D102
        orig, residual, phase = self.base.range(start, stop, step)
        if self.align == 1:
            # Fast path to make the no-op case cheap
            return orig, residual, phase
        if self.align % 2 == 0:
            # Use the sign of the residual to break ties in the direction that
            # minimises the updated residual.
            sign = np.empty_like(residual, dtype=np.int64)
            np.sign(residual, out=sign, casting="unsafe")
            aligned = (2 * orig + self.align - sign) // (2 * self.align) * self.align
        else:
            # With odd `align`, there are no ties to break
            aligned = (orig + self.align // 2) // self.align * self.align
        residual += aligned - orig
        return aligned, residual, phase

    def skip(self, target: int, start: int, step: int) -> int:  # noqa: D102
        target = _round_up(target, self.align)
        # If base has an orig_time of target - align/2, it could round either
        # way (depending on the residual), so it might be the timestamp we
        # need.
        t = self.base.skip(target - self.align // 2, start, step)
        if self(t)[0] >= target:
            return t
        # If not, this will definitely round to at least the target
        return self.base.skip(target - self.align // 2 + 1, start, step)
