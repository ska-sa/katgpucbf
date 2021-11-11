################################################################################
# Copyright (c) 2021, National Research Foundation (SARAO)
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

"""Shared utilities for receiving SPEAD data."""

from typing import Mapping

import spead2.recv
from prometheus_client import Counter


class StatsToCounters:
    """Reflect stream statistics as Prometheus Counters.

    This class tracks the last-known value of each statistic for the
    corresponding stream.

    Parameters
    ----------
    counter_map
        Dictionary that maps a stream statistic (by name) to a counter.
    config
        Configuration of the stream for which the statistics will be retrieved.
    """

    def __init__(self, counter_map: Mapping[str, Counter], config: spead2.recv.StreamConfig) -> None:
        self._updates = [(config.get_stat_index(name), counter) for name, counter in counter_map.items()]
        self._current = [0] * len(self._updates)

    def update(self, stats: spead2.recv.StreamStats) -> None:
        """Update the counters based on the current stream statistics."""
        for i, (index, counter) in enumerate(self._updates):
            new = stats[index]
            counter.inc(new - self._current[i])
            self._current[i] = new
