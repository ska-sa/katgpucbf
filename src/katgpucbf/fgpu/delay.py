"""A collection of classes and methods for delay-tracking."""

import warnings
from abc import ABC, abstractmethod
from collections import deque
from typing import Tuple

import numpy as np


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
        """Find  input sample timestamp corresponding to a given output sample.

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
        Unit-less rate of change of delay. [seconds/second]
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
        if delay <= -1.0:
            raise ValueError("delay rate must be greater than -1")
        if start < 0:
            raise ValueError("start must be non-negative")
        self.start = start
        self.delay = float(delay)
        self.delay_rate = float(delay_rate)
        self.phase = float(phase)
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
        residual = rel_orig_rnd - rel_orig

        # Calculate the phase
        phase = rel_time * self.phase_rate + self.phase

        # add self.start back again to return the timestamps in the original
        # epoch
        return rel_orig_rnd + self.start, residual, phase


class MultiDelayModel(AbstractDelayModel):
    """Piece-wise linear delay model.

    The model evolves over time by calling :meth:`add`. It **must** only be
    queried monotonically, because as soon as a query is made beyond the end of
    the first piece it is discarded.

    In the initial state it has a model with zero delay.
    """

    def __init__(self) -> None:
        self._models = deque([LinearDelayModel(0, 0.0, 0.0, 0.0, 0.0)])

    def __call__(self, time: float) -> float:  # noqa: D102
        while len(self._models) > 1 and time >= self._models[1].start:
            self._models.popleft()
        if time < self._models[0].start:
            warnings.warn("Timestamp is before start of first linear model - possibly due to non-monotonic queries")
        return self._models[0](time)

    def invert_range(self, start: int, stop: int, step: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:  # noqa: D102
        orig, fine_delay, phase = self._models[0].invert_range(start, stop, step)

        if len(orig) == 0:  # Defence against corner case breaking things.
            return orig, fine_delay, phase

        if orig[0] < self._models[0].start:
            warnings.warn("Timestamp is before start of first linear model - possibly due to non-monotonic queries")

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
            self._models.popleft()
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
