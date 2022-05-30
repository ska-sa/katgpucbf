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
from typing import List, Optional, Union

import numpy as np
import numpy.typing as npt

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

    def plot(
        self,
        x: npt.ArrayLike,
        y: npt.ArrayLike,
        *,
        caption: Optional[str] = "",
        xlabel: Optional[str] = "",
        ylabel: Optional[str] = "",
        legend_labels: Optional[Union[str, List[str]]] = "",
    ) -> None:
        """Capture numerical data for plotting.

        Parameters
        ----------
        x
            X-data for plotting. Must be one-dimensional.
        y
            Y-data for plotting. Can be up to two-dimensional, but the length
            of the second dimension must agree with the length of `x`.
        caption
            Title for the graph.
        xlabel
            Label for the X-axis.
        ylabel
            Label for the Y-axis.
        legend_labels
            Legend labels for the various sets of data plotted. Optional only
            in single-dimension plots, if a 2D `y` is given, a list of labels
            must be passed.

        Raises
        ------
        ValueError
            If called before :func:`Report.step`, as the plot must be associated
            with a step in the test procedure.
        """
        # Coerce to np.ndarray for data validation.
        x = np.asarray(x)
        y = np.asarray(y)

        # I must admit that I'm nervous about using `assert` for this but I
        # guess that we're unlikely ever to run a test suite with `-O`.
        assert x.ndim == 1, f"x has {x.ndim} dimensions, expected 1!"
        assert y.ndim <= 2, "Can't have y with more than 2 dimensions!"
        assert x.size == y.shape[-1], "x and y must have same length for plotting!"
        if y.ndim > 1 and legend_labels is not None:
            assert len(legend_labels) == y.shape[0], "If y is 2-dimensional, we need legend labels."

        # Moving swiftly along.
        if self._cur_step is None:
            raise ValueError("Cannot have a plot without a current step")

        self._cur_step.append(
            {
                "$msg_type": "plot",
                "y": y.tolist(),
                "x": x.tolist(),
                "caption": caption,
                "xlabel": xlabel,
                "ylabel": ylabel,
                "legend_labels": legend_labels,
            }
        )
