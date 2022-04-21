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

"""A collection of classes and methods for delay-tracking.

It should be noted that the classes in this module use a slightly different
model than the public katcp interface. The reference channel for phase change
is channel 0, rather than the centre channel. The difference is dealt with
by the request handler for the ``?delays`` katcp request.
"""

import warnings
from abc import ABC, abstractmethod
from collections import deque
from typing import Callable, Optional, Sequence, Tuple

import numpy as np


def wrap_angle(angle):
    """Restrict an angle to [-pi, pi].

    This works on both Python scalars and numpy arrays.
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


class NonMonotonicQueryWarning(UserWarning):
    """Delay model was queried non-monotonically."""


class AbstractDelayModel(ABC):
    """Abstract base class for delay models.

    All units are samples rather than SI units.
    """

    @abstractmethod
    def __call__(self, time: float) -> float:
        """Determine delay at a given sample.

        Note that this returns only the delay that was applied to the original
        timestamp, not the timestamp itself. No check is made that the sample
        comes after `start` - the function will happily interpolate backwards.
        """

    @abstractmethod
    def invert_range(self, start: int, stop: int, step: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
            Undelayed timestamps corresponding to ``range(start, stop, step)``
        residual
            Fractional sample delay not accounted for by ``time - orig_time``.
        phase
            Fringe-stopping phase to be added.
        """

    def invert(self, time: int) -> Tuple[int, float, float]:
        """Find input sample timestamp corresponding to a given output sample.

        Parameters
        ----------
        time
            Delayed timestamp.

        Returns
        -------
        orig_time
            Undelayed timestamp corresponding to `time`.
        residual
            Fractional sample delay not accounted for by ``time - orig_time``.
        phase
            Fringe-stopping phase to be added.
        """
        orig_time, residual, phase = self.invert_range(time, time + 1, 1)
        return int(orig_time[0]), float(residual[0]), float(phase[0])


class LinearDelayModel(AbstractDelayModel):
    """Delay model that adjusts delay linearly over time.

    Parameters
    ----------
    start
        Sample at which the model should start being used.
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
        if `rate` is less than or equal to -1 or `start` is negative
    """

    def __init__(self, start: int, delay: float, delay_rate: float, phase: float, phase_rate: float) -> None:
        if delay_rate <= -1.0:
            raise ValueError("delay rate must be greater than -1")
        self.start = start
        self.delay = float(delay)
        self.delay_rate = float(delay_rate)
        self.phase = wrap_angle(float(phase))
        self.phase_rate = float(phase_rate)

    def __call__(self, time: float) -> float:  # noqa: D102
        rel_time = time - self.start
        return rel_time * self.delay_rate + self.delay

    def invert_range(self, start: int, stop: int, step: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:  # noqa: D102
        time = np.arange(start, stop, step)
        # Variables with names prefixed rel_ treat start of delay model as t_0.
        # Makes it easier to apply the rate.
        rel_time = time - self.start
        # Solve `rel_time = rel_orig + delay + rel_orig*rate` for rel_orig and
        # you end up with the following equation.
        # rel_time is the (relative) corrected timestamp, i.e. after the delay
        # model has been applied, while rel_orig is the (relative) original
        # timestamp, the one that would have come from the digitiser before
        # correction.
        rel_orig = (rel_time - self.delay) / (self.delay_rate + 1)
        rel_orig_rnd = np.rint(rel_orig).astype(np.int64)
        # Prevent coarse delay from becoming negative
        np.minimum(rel_orig_rnd, rel_time, out=rel_orig_rnd)
        residual = rel_orig_rnd - rel_orig

        # Calculate the phase
        phase = wrap_angle(rel_orig * self.phase_rate + self.phase)

        # add self.start back again to return the timestamps in the original
        # epoch
        return rel_orig_rnd + self.start, residual, phase


class MultiDelayModel(AbstractDelayModel):
    """Piece-wise linear delay model.

    The model evolves over time by calling :meth:`add`. It **must** only be
    queried monotonically, because as soon as a query is made beyond the end
    of the first piece it is discarded.

    In the initial state it has a model with zero delay.

    It accepts an optional callback function that takes in the
    LinearDelayModels attached to this MultiDelayModel. This callback is
    called whenever the first linear piece changes. It is also called
    immediately by the constructor.
    """

    def __init__(self, callback_func: Optional[Callable[[Sequence[LinearDelayModel]], None]] = None) -> None:
        # The initial time is -1 rather than 0 so that it doesn't get removed
        # if a model is added with start time 0, which can lead to some
        # spurious warnings in unit tests about non-monotonic queries.
        # Ideally it would use -infinity (or a large negative number), but
        # that causes issues with numeric precision.
        self._models = deque([LinearDelayModel(-1, 0.0, 0.0, 0.0, 0.0)])
        self.callback_func = callback_func
        if callback_func is not None:
            callback_func(self._models)

    def __call__(self, time: float) -> float:  # noqa: D102
        while len(self._models) > 1 and time >= self._models[1].start:
            self._popleft()
        if time < self._models[0].start:
            warnings.warn(
                f"Timestamp {time} is before start of first linear model "
                f"at {self._models[0].start} - possibly due to non-monotonic queries",
                NonMonotonicQueryWarning,
            )
        return self._models[0](time)

    def _popleft(self) -> None:
        """Carry out a popleft and callback invocation."""
        self._models.popleft()
        if self.callback_func is not None:
            self.callback_func(self._models)

    def invert_range(self, start: int, stop: int, step: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:  # noqa: D102
        orig, fine_delay, phase = self._models[0].invert_range(start, stop, step)

        if len(orig) == 0:  # Defence against corner case breaking things.
            return orig, fine_delay, phase

        if orig[0] < self._models[0].start:
            warnings.warn(
                f"Timestamp {orig[0]} is before start of first linear model "
                f"at {self._models[0].start} - possibly due to non-monotonic queries",
                NonMonotonicQueryWarning,
            )

        # Step through later models and overwrite the first one where later ones
        # are valid. This is not particularly optimal since we evaluate the full
        # range for each combination of model and timestamp. However, we expect
        # to have only a small number of models.
        cull = 0
        for i, model in enumerate(self._models):
            if i == 0:
                continue  # We've already done the first one
            if stop <= model.start:
                # Models are assumed to have positive delays, so the
                # inverse of stop in any model is <= stop.
                break
            new_orig, new_fine_delay, new_phase = model.invert_range(start, stop, step)
            mask = new_orig >= model.start
            np.copyto(orig, new_orig, where=mask)
            np.copyto(fine_delay, new_fine_delay, where=mask)
            np.copyto(phase, new_phase, where=mask)
            if mask[0]:
                # The previous model is completely overwritten
                cull = i
        for _ in range(cull):
            self._popleft()
        return orig, fine_delay, phase

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
