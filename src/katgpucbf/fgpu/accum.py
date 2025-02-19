################################################################################
# Copyright (c) 2024-2025, National Research Foundation (SARAO)
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

"""Compute statistics by bucketing into windows."""

from dataclasses import dataclass
from typing import Final, Protocol, Self


# Based on typing.SupportsAbs etc.
class _SupportsAdd[T](Protocol):
    def __add__(self, other: Self) -> T:
        pass  # pragma: nocover


@dataclass(frozen=True)
class Measurement[T: _SupportsAdd]:
    """A measurement returned by :meth:`Accum.add`."""

    start_timestamp: int
    end_timestamp: int
    #: Total of the value over the provided data.
    total: T | None


def _add[T: _SupportsAdd](a: T | None, b: T | None) -> T | None:
    """Add two values, returning None if either of them is None."""
    if a is not None and b is not None:
        return a + b
    else:
        return None


class Accum[T: _SupportsAdd]:
    """Accumulator for a single statistic.

    The statistic is a linear measurement over intervals of time.
    Measurements over small intervals are provided to this class, which
    accumulates them. Time is divided into fixed "windows" (all the same
    length), with a total produced for each window for which at least some data
    is provided. If a window is missing some data, either because it was not
    provided or it was explicitly indicated that it was missing, then the
    window measurement will instead indicate that the sum is unknown.

    Time is considered to be discrete (integer). Window intervals are
    timestamps that are multiples of `window_size`. Data cannot be added for
    intervals that cross window boundaries.

    Parameters
    ----------
    window_size
        Factor that divides timestamp window boundaries
    zero
        Value to initialise the accumulator to
    """

    def __init__(self, window_size: int, zero: T) -> None:
        self._window_size: Final = window_size
        self._zero: Final = zero
        self._total: T | None = zero  # Sum for current window, if valid
        # Point up to which we have received calls to :meth:`add`
        self._end_timestamp = 0
        self._window_id = 0  # Start timestamp divided by the window size

    def _flush(self, new_window_id: int) -> Measurement | None:
        """Generate a :class:`Measurement` (if non-empty) and reset state."""
        # [base_timestamp, next_timestamp) is the full range of the current window
        base_timestamp = self._window_id * self._window_size
        next_timestamp = base_timestamp + self._window_size
        ret: Measurement | None = None
        # Don't send if the old window was completely empty
        if self._end_timestamp > base_timestamp:
            # If we didn't get the end of the window, then we don't know the total
            total = self._total if self._end_timestamp == next_timestamp else None
            ret = Measurement(
                base_timestamp,
                next_timestamp,
                total,
            )
        # Reset the state
        self._window_id = new_window_id
        self._end_timestamp = new_window_id * self._window_size
        self._total = self._zero
        return ret

    def add(self, start_timestamp: int, end_timestamp: int, value: T | None) -> Measurement | None:
        """Add new data.

        If the new data falls into a new window compared to the existing data
        or it completes the current window, then an instance of
        :class:`Measurement` is returned with the total for the previous
        window. Note that if both occur, the result for the old window is
        simply discarded.

        Parameters
        ----------
        start_timestamp, end_timestamp
            The time range for the extension
        value
            The statistic measured over the given time range, or
            None if there was missing data in the range.

        Raises
        ------
        ValueError
            If `start_timestamp` > `end_timestamp`
        ValueError
            If the new data overlaps or preceeds previous data
        ValueError
            If [start_timestamp, end_timestamp) crosses a window boundary
        """
        if start_timestamp > end_timestamp:
            raise ValueError("start_timestamp ({start_timestamp}) > end_timestamp ({end_timestamp})")
        new_window_id = start_timestamp // self._window_size
        new_window_end = (new_window_id + 1) * self._window_size
        if end_timestamp > new_window_end:
            raise ValueError("new data crosses a window boundary")
        if start_timestamp < self._end_timestamp:
            raise ValueError("new data starts before end of previous data")

        ret: Measurement | None = None
        if new_window_id != self._window_id:
            # New data falls into a new window - flush out the old one.
            ret = self._flush(new_window_id)

        if start_timestamp != self._end_timestamp:
            # We skipped some data
            value = None

        self._total = _add(self._total, value)
        self._end_timestamp = end_timestamp

        if end_timestamp == new_window_end:
            # New data completes a window - flush it now.
            ret = self._flush(self._window_id + 1)

        return ret
