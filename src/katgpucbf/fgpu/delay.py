from collections import deque
from typing import Tuple
import warnings
from abc import ABC, abstractmethod

import numpy as np


class AbstractDelayModel(ABC):
    """Abstract base class for delay models.

    All units are samples rather than SI units.

    """

    @abstractmethod
    def __call__(self, time: float) -> float:
        """Determine delay at a given sample.

        No check is made that the sample comes after `start` - it will
        happily interpolate backwards.
        """

    @abstractmethod
    def invert_range(self, start: int, stop: int, step: int) -> Tuple[np.ndarray, np.ndarray]:
        """Find input sample timestamps corresponding to a range of output samples.

        For each output sample with timestamp in ``range(start, stop, step)``, it
        determines a corresponding input sample.

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
        """

    def invert(self, time: int) -> Tuple[int, float]:
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
        """
        orig_time, residual = self.invert_range(time, time + 1, 1)
        return int(orig_time[0]), float(residual[0])


class LinearDelayModel(AbstractDelayModel):
    """Delay model that adjusts delay linearly over time.

    Parameters
    ----------
    start
        Sample at which the model should start being used.
    delay
        Delay to apply at `start`.
    rate
        Unit-less rate of change of delay.

    Raises
    ------
    ValueError
        if `rate` is less than or equal to -1 or `start` is negative
    """

    def __init__(self, start: int, delay: float, rate: float) -> None:
        if delay <= -1.0:
            raise ValueError('delay rate must be greater than -1')
        if start < 0:
            raise ValueError('start must be non-negative')
        self.start = start
        self.delay = float(delay)
        self.rate = float(rate)

    def __call__(self, time: float) -> float:
        rel_time = time - self.start
        return rel_time * self.rate + self.delay

    def invert_range(self, start: int, stop: int, step: int) -> Tuple[np.ndarray, np.ndarray]:
        time = np.arange(start, stop, step)
        rel_time = time - self.start
        rel_orig = (rel_time - self.delay) / (self.rate + 1)
        rel_orig_rnd = np.rint(rel_orig).astype(np.int64)
        residual = rel_orig_rnd - rel_orig
        return rel_orig_rnd + self.start, residual


class MultiDelayModel(AbstractDelayModel):
    """Piece-wise linear delay model.

    The model evolves over time by calling :meth:`add`. It **must** only be
    queried monotonically, because as soon as a query is made beyond the end of
    the first piece it is discarded.

    In the initial state it has a model with zero delay.
    """

    def __init__(self) -> None:
        self._models = deque([LinearDelayModel(0, 0.0, 0.0)])

    def __call__(self, time: float) -> float:
        while len(self._models) > 1 and time >= self._models[1].start:
            self._models.popleft()
        if time < self._models[0].start:
            warnings.warn('Timestamp is before start of first linear model - '
                          'possibly due to non-monotonic queries')
        return self._models[0](time)

    def invert_range(self, start: int, stop: int, step: int) -> Tuple[np.ndarray, np.ndarray]:
        orig, fine_delay = self._models[0].invert_range(start, stop, step)
        if len(orig) == 0:
            return orig, fine_delay

        if orig[0] < self._models[0].start:
            warnings.warn('Timestamp is before start of first linear model - '
                          'possibly due to non-monotonic queries')
        # Step through later models and apply them where valid. This is not
        # particularly optimal since we evaluate the full range for each
        # combination of model and timestamp. However, we expect to have
        # only a small number of models.
        cull = 0
        for i, model in enumerate(self._models):
            if i == 0:
                continue    # We've already done it
            if stop <= model.start:
                # Models are assumed to have positive delays, so the
                # inverse of stop in any model is <= stop.
                break
            new_orig, new_fine_delay = model.invert_range(start, stop, step)
            mask = new_orig >= model.start
            np.copyto(orig, new_orig, where=mask)
            np.copyto(fine_delay, new_fine_delay, where=mask)
            if mask[0]:
                # The previous model is completely overwritten
                cull = i
        for i in range(cull):
            self._models.popleft()
        return orig, fine_delay

    def add(self, model: LinearDelayModel) -> None:
        """Extend the model with a new linear model.

        The new model is applicable from its start time forever. If the new
        model has an earlier start time than some previous model, the previous
        model will be discarded.
        """
        while self._models and model.start <= self._models[-1].start:
            self._models.pop()
        self._models.append(model)
