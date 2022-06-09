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
import logging
import time
from typing import Any, Mapping, Optional

import matplotlib.figure
import tikzplotlib

logger = logging.getLogger(__name__)


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

    def figure(
        self,
        figure: matplotlib.figure.Figure,
        clean_figure: bool = True,
        tikzplotlib_kwargs: Mapping[str, Any] = {},  # noqa: B006
    ) -> None:
        """Add a matplotlib figure to the report.

        Parameters
        ----------
        figure
            The figure to plot
        clean_figure
            If true (default), use :func:`tikzplotlib.clean_figure` on the
            figure to remove points outside the axis limits, etc.
            Note that this may *modify* the figure.
        tikzplotlib_kwargs
            Extra keyword arguments to pass to :func:`tikzplotlib.get_tikz_code`.
        """
        if clean_figure:
            tikzplotlib.clean_figure(figure)
        kwargs = dict(tikzplotlib_kwargs)
        kwargs.setdefault("table_row_sep", r"\\")
        self.raw_figure(tikzplotlib.get_tikz_code(figure, **kwargs))
