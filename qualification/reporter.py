################################################################################
# Copyright (c) 2022, National Research Foundation (SARAO)
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

"""Mechanism for logging Pytest's output to a PDF."""
import io
import logging
import time
from typing import Optional

import matplotlib.figure
import matplotlib.ticker
import matplotlib.transforms
import numpy as np

logger = logging.getLogger(__name__)


class POTLocator(matplotlib.ticker.Locator):
    """Tick locator that uses a power-of-two step size.

    This code is based on examining the source of MaxNLocator and
    MultipleLocator. There may be some cargo-culting.
    """

    # Method docstrings are omitted because they're provided by the base class

    def __init__(self, nbins: int = 10) -> None:
        self.set_params(nbins=nbins)

    def set_params(self, nbins: Optional[int]) -> None:  # noqa: D102
        if nbins is not None:
            self._nbins = nbins

    def __call__(self) -> np.ndarray:  # noqa: D102
        vmin, vmax = self.axis.get_view_interval()
        return self.tick_values(vmin, vmax)

    def tick_values(self, vmin: float, vmax: float) -> np.ndarray:  # noqa: D102
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


class Reporter:
    """Provides mechanisms to log steps taken in a test."""

    def __init__(self, data: list) -> None:
        self._data = data
        self._cur_step: Optional[list] = None

    def config(self, **kwargs) -> None:
        """Report the test configuration."""
        test_config = {"$msg_type": "config"}
        test_config.update(kwargs)
        self._data.append(test_config)

    def step(self, message: str) -> None:
        """Report the start of a high-level step."""
        self._cur_step = []
        logger.info(message)
        self._data.append({"$msg_type": "step", "message": message, "details": self._cur_step})

    def detail(self, message: str) -> None:
        """Report a low-level detail, associated with the previous call to :meth:`step`."""
        if self._cur_step is None:
            raise ValueError("Cannot have detail without a current step")
        logger.debug(message)
        self._cur_step.append({"$msg_type": "detail", "message": message, "timestamp": time.time()})

    def raw_figure(self, code: str) -> None:
        """Add raw LaTeX to the document.

        It will be set inside a minipage and is intended for figures, but could
        potentially contain tables too.
        """
        if self._cur_step is None:
            raise ValueError("Cannot have figure without a current step")
        self._cur_step.append({"$msg_type": "figure", "code": code})

    def figure(self, figure: matplotlib.figure.Figure) -> None:
        """Add a matplotlib figure to the report.

        Parameters
        ----------
        figure
            The figure to plot
        """
        content = io.StringIO()
        figure.savefig(content, format="pgf", backend="pgf")
        self.raw_figure(content.getvalue())
