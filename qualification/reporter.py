################################################################################
# Copyright (c) 2022-2025, National Research Foundation (SARAO)
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

"""Mechanism for logging Pytest's output to a PDF.

The immediate output is not actually a PDF. Instead, custom JSON-compatible
data structures are stored in the log output. These are post-processed by
:file:`report/generate_pdf.py` to produce a PDF.
"""

import base64
import io
import logging
import time
from collections.abc import Sequence
from typing import Any

import matplotlib.axes
import matplotlib.colors
import matplotlib.figure
import matplotlib.lines
import matplotlib.ticker
import matplotlib.transforms
import numpy as np
import pytest
from numpy.typing import ArrayLike

logger = logging.getLogger(__name__)


class POTLocator(matplotlib.ticker.Locator):
    """Tick locator that uses a power-of-two step size.

    This code is based on examining the source of MaxNLocator and
    MultipleLocator. There may be some cargo-culting.
    """

    # Method docstrings are omitted because they're provided by the base class

    def __init__(self, nbins: int = 10) -> None:
        self.set_params(nbins=nbins)

    def set_params(self, nbins: int | None = None) -> None:  # noqa: D102
        if nbins is not None:
            self._nbins = nbins

    def __call__(self) -> Sequence[float]:  # noqa: D102
        assert self.axis is not None
        vmin, vmax = self.axis.get_view_interval()
        return self.tick_values(vmin, vmax)

    def tick_values(self, vmin: float, vmax: float) -> Sequence[float]:  # noqa: D102
        vmin, vmax = matplotlib.transforms.nonsingular(vmin, vmax, expander=1e-13, tiny=1e-14)
        step = 2 ** np.ceil(np.log2((vmax - vmin) / self._nbins))
        # Note: MultipleLocator uses a private helper class to ensure that
        # floating-point rounding issues don't get in the way. For the
        # qualification report the user can't zoom/pan arbitrarily, so we don't
        # bother.
        vmin = vmin // step * step
        vmax = (vmax // step + 1) * step
        n = (vmax - vmin + 0.001 * step) // step
        locs = vmin - step + np.arange(n + 3) * step
        return self.raise_if_exceeds(locs)


def plot_focus(
    ax: matplotlib.axes.Axes, focus: slice, x: ArrayLike, y: ArrayLike, *args, **kwargs
) -> list[matplotlib.lines.Line2D]:
    """Plot a line where only an interval is of primary interest.

    The parts outside the interval are shown semi-transparent and do not
    affect the Y scaling. The implementation makes several calls to the
    underlying plot function.

    Parameters
    ----------
    ax
        Axes on which to plot the line
    focus
        Slice from `x` and `y` which is to be in focus. This must have
        a step of ``None`` and not fall outside the size of `x` and `y`.
    x, y
        Data to plot, which must be 1D and of the same size.
    *args, **kwargs
        Passed to :meth:`matplotlib.axes.Axes.plot`.
    """
    xarr = np.asarray(x)
    yarr = np.asarray(y)
    assert xarr.shape == yarr.shape
    assert xarr.ndim == 1
    assert focus.step is None or focus.step == 1
    assert 0 <= focus.start < focus.stop <= len(xarr)

    # Draw the focused part first, to get the color set
    lines = ax.plot(xarr[focus], yarr[focus], *args, **kwargs)
    # Make partially transparent color
    faded = matplotlib.colors.to_rgba(lines[0].get_color(), alpha=0.3)
    # Back up the y data limits
    y0, y1 = ax.dataLim.intervaly
    # We don't want a separate legend entry for the faded sections
    kwargs.pop("label", None)
    # Draw the faded sections
    if 0 < focus.start:
        lines += ax.plot(xarr[: focus.start], yarr[: focus.start], *args, **kwargs, color=faded)
    if focus.stop < len(xarr):
        lines += ax.plot(xarr[focus.stop :], yarr[focus.stop :], *args, **kwargs, color=faded)
    # Restore y data limits
    ax.dataLim.set_points(np.array([[ax.dataLim.x0, y0], [ax.dataLim.x1, y1]]))
    return lines


class Reporter:
    """Provides mechanisms to log steps taken in a test.

    If `raw_data` is true, raw data from line plots in figures will
    be added to the report.
    """

    def __init__(self, data: list, raw_data: bool = False) -> None:
        self._data = data
        self._cur_step: list | None = None
        self._raw_data = raw_data

    def config(self, **kwargs) -> None:
        """Report the test configuration."""
        test_config = {"$msg_type": "config"}
        test_config.update(kwargs)
        self._data.append(test_config)

    def step(self, message: str) -> None:
        """Report the start of a high-level step."""
        self._cur_step = []
        logger.info(message)
        self._data.append({"$msg_type": "step", "message": message, "items": self._cur_step})

    def detail(self, message: str) -> None:
        """Report a low-level detail, associated with the previous call to :meth:`step`."""
        if self._cur_step is None:
            raise ValueError("Cannot have detail without a current step")
        logger.debug(message)
        self._cur_step.append({"$msg_type": "detail", "message": message, "timestamp": time.time()})

    def failure(self, message: str) -> None:
        """Report a non-fatal test failure.

        This should generally not be done directly; use pytest_check.
        """
        if self._cur_step is None:
            raise ValueError("Cannot have failure without a current step")
        self._cur_step.append({"$msg_type": "failure", "message": message, "timestamp": time.time()})

    def raw_figure(self, code: str, data: list = []) -> None:  # noqa: B006
        """Add raw LaTeX to the document.

        It will be set inside a minipage and is intended for figures, but could
        potentially contain tables too.
        """
        if self._cur_step is None:
            raise ValueError("Cannot have raw_figure without a current step")
        value: dict[str, Any] = {"$msg_type": "figure", "code": code}
        if self._raw_data:
            value["data"] = data
        self._cur_step.append(value)

    def figure(self, figure: matplotlib.figure.Figure) -> None:
        """Add a matplotlib figure to the report.

        Parameters
        ----------
        figure
            The figure to plot
        """
        if self._cur_step is None:
            raise ValueError("Cannot have figure without a current step")
        data = []
        for ax in figure.axes:
            for line in ax.get_lines():
                data.append(np.asarray(line.get_xydata()).tolist())
        content = io.BytesIO()
        figure.savefig(content, format="pdf", backend="pdf")
        # The .decode converts from bytes to str
        content_b64 = base64.standard_b64encode(content.getvalue()).decode()
        value: dict[str, Any] = {"$msg_type": "binary_figure", "content": content_b64, "type": "pdf"}
        if self._raw_data:
            value["data"] = data
        self._cur_step.append(value)


def custom_report_log(pytestconfig: pytest.Config, data) -> None:
    """Log a custom JSON line in the report log."""
    # There doesn't seem to be an easy way to avoid using these private interfaces
    try:
        report_log_plugin = pytestconfig._report_log_plugin  # type: ignore
    except AttributeError:
        pytest.fail("pytest_reportlog plugin not found (possibly you forgot to specify --report-log)")
    report_log_plugin._write_json_data(data)
