from collections import deque
from typing import Tuple
import warnings
from abc import ABC, abstractmethod


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


class LinearDelayModel(ABC):
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

    def invert(self, time: int) -> Tuple[int, float]:
        rel_time = time - self.start
        rel_orig = int(round((rel_time - self.delay) / (self.rate + 1)))
        delay = rel_orig * self.rate + self.delay
        residual = delay - (rel_time - rel_orig)
        return rel_orig + self.start, residual


class MultiDelayModel:
    """Piece-wise linear delay model.

    The model evolves over time by calling :meth:`add`. It **must** only be
    queried monotonically, because as soon as a query is made beyond the end of
    the first piece it is discarded.

    In the initial state it has a model with zero delay.
    """

    def __init__(self):
        self._models = deque([LinearDelayModel(0, 0.0, 0.0)])

    def __call__(self, time: float) -> float:
        while len(self._models) > 1 and time >= self._models[1].start:
            self._models.popleft()
        if time < self._models[0].start:
            warnings.warn('Timestamp is before start of first linear model - '
                          'possibly due to non-monotonic queries')
        return self._models[0](time)

    def invert(self, time: int) -> Tuple[int, float]:
        while True:
            ans = self._models[0].invert(time)
            if len(self._models) <= 1 or ans[0] < self._models[1].start:
                break
            self._models.popleft()
        if ans[0] < self._models[0].start:
            warnings.warn('Timestamp is before start of first linear model - '
                          'possibly due to non-monotonic queries')
        return ans

    def add(self, model: LinearDelayModel) -> None:
        """Extend the model with a new linear model.

        The new model is applicable from its start time forever. If the new
        model has an earlier start time than some previous model, the previous
        model will be discarded.
        """
        while self._models and model.start <= self._models[-1].start:
            self._models.pop()
        self._models.append(model)
