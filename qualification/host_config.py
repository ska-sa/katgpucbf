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

"""Collect and report configuration of hosts."""

from typing import Optional, Set

from prometheus_api_client.prometheus_connect import PrometheusConnect

QUERIES = [
    'node_cpu_info{instance="%h", cpu="0"}',
    'node_ethtool_info{instance="%h"}',
    'node_dmi_info{instance="%h"}',
]


class HostConfigQuerier:
    """Query Prometheus metrics to describe host configuration.

    The raw metric data is dumped into the report JSON, and left to the report
    generator to parse. A cache is kept of previously seen hosts so that each
    host is only reported on once.
    """

    def __init__(self, prometheus_url: str) -> None:
        self._prom = PrometheusConnect(prometheus_url)
        self._seen: Set[str] = set()

    def get_config(self, hostname: str) -> Optional[list]:
        """Retrieve information about a host.

        The return value is a list of JSON dicts, each representing a single
        Prometheus metric, in the raw form returned by the Prometheus API.
        Alternatively, if the host has already been queried, returns ``None``.
        """
        if hostname in self._seen:
            return None
        ans = []
        for query in QUERIES:
            query_str = query.replace("%h", hostname)
            ans.extend(self._prom.custom_query(query_str))
        self._seen.add(hostname)
        return ans
