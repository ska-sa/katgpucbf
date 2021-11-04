# noqa: D104

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

from typing import Dict, List, Optional

import prometheus_client


class PromDiff:
    """Collects Prometheus metrics before and after test code, and provides differences.

    Typical usage is::

        with PromDiff() as diff:
            ...  # Do stuff that increments counters
        diff.get_sample_diff(name, labels)
    """

    def __init__(self, registry: prometheus_client.CollectorRegistry = prometheus_client.REGISTRY) -> None:
        self._registry = registry
        self._before: List[prometheus_client.samples.Sample] = []
        self._after: List[prometheus_client.samples.Sample] = []

    def __enter__(self) -> "PromDiff":
        self._before = [s for metric in self._registry.collect() for s in metric.samples]
        return self

    def __exit__(self, *args) -> None:
        self._after = [s for metric in self._registry.collect() for s in metric.samples]

    @staticmethod
    def _get_value(
        samples: List[prometheus_client.samples.Sample], name: str, labels: Dict[str, str]
    ) -> Optional[float]:
        for s in samples:
            if s.name == name and s.labels == labels:
                return s.value
        return None

    def get_sample_value(self, name: str, labels: Optional[Dict[str, str]] = None) -> Optional[float]:
        """Return the value of the metric at the end of the context manager protocol.

        If it is not found, returns ``None``.
        """
        if labels is None:
            labels = {}
        return self._get_value(self._after, name, labels)

    def get_sample_diff(self, name: str, labels: Optional[Dict[str, str]] = None) -> Optional[float]:
        """Return the increase in the metric during the context manager protocol.

        If it is not found, returns ``None``. If it was not found on entry,
        returns the value on exit.
        """
        if labels is None:
            labels = {}
        before = self._get_value(self._before, name, labels)
        after = self._get_value(self._after, name, labels)
        if before is not None and after is not None:
            return after - before
        else:
            return after
