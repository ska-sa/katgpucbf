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

from typing import Any, Dict, List, Optional

import aiokatcp
import prometheus_client


class PromDiff:
    """Collects Prometheus metrics before and after test code, and provides differences.

    Typical usage is::

        with PromDiff(namespace=METRIC_NAMESPACE) as diff:
            ...  # Do stuff that increments counters
        diff.get_sample_diff(name, labels)

    Parameters
    ----------
    registry
        Prometheus metric registry
    namespace
        Namespace to prepend to metric names
    """

    def __init__(
        self,
        *,
        registry: prometheus_client.CollectorRegistry = prometheus_client.REGISTRY,
        namespace: Optional[str] = None
    ) -> None:
        self._registry = registry
        self._before: List[prometheus_client.samples.Sample] = []
        self._after: List[prometheus_client.samples.Sample] = []
        self._prefix = namespace + "_" if namespace is not None else ""

    def __enter__(self) -> "PromDiff":
        self._before = [s for metric in self._registry.collect() for s in metric.samples]
        return self

    def __exit__(self, *args) -> None:
        self._after = [s for metric in self._registry.collect() for s in metric.samples]

    def _get_value(
        self, samples: List[prometheus_client.samples.Sample], name: str, labels: Dict[str, str]
    ) -> Optional[float]:
        for s in samples:
            if s.name == self._prefix + name and s.labels == labels:
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


async def get_sensor(client: aiokatcp.Client, name: str) -> Any:
    """Get the value of a sensor.

    .. todo:

       This should probably be implemented in aiokatcp.
    """
    # This is not a complete list of sensor types. Extend as necessary.
    sensor_types = {
        b"integer": int,
        b"float": float,
        b"boolean": bool,
        b"discrete": str,  # Gets the string name of the enum
        b"string": str,  # Allows passing through arbitrary values even if not UTF-8
    }

    _reply, informs = await client.request("sensor-list", name)
    assert len(informs) == 1
    sensor_type = sensor_types.get(informs[0].arguments[3], bytes)
    _reply, informs = await client.request("sensor-value", name)
    assert len(informs) == 1
    assert informs[0].arguments[3] in {b"nominal", b"warn", b"error"}
    return aiokatcp.decode(sensor_type, informs[0].arguments[4])
