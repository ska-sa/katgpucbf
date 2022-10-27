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

"""Unit tests for :mod:`katgpucbf.recv`."""

import gc
import weakref
from collections.abc import Iterable
from unittest import mock

import pytest
import spead2.recv
from prometheus_client import CollectorRegistry

from katgpucbf.recv import StatsCollector


@pytest.fixture
def registry() -> CollectorRegistry:
    """Create a non-default registry for a test."""
    return CollectorRegistry()


def mock_stream(extra_stats: Iterable[str] = ("too_old_heaps",)) -> mock.Mock:
    """Create a mock of :class:`spead2.recv.Stream`.

    The mock has just enough in it to work with
    :class:`~katgpucbf.recv.StatsCollector`.
    """
    stream = mock.Mock()
    stream.config = spead2.recv.StreamConfig()
    for name in extra_stats:
        stream.config.add_stat(name)
    stream.stats = {}
    for stat_config in stream.config.stats:
        stream.stats[stream.config.get_stat_index(stat_config.name)] = 0
    return stream


def inc_stat(stream: mock.Mock, name: str, value: int) -> None:
    """Increment a statistic in a stream returned by :func:`mock_stream`."""
    idx = stream.config.get_stat_index(name)
    stream.stats[idx] += value


@pytest.fixture
def stats_collector(registry: CollectorRegistry) -> StatsCollector:
    """Empty stats collector, with some labels."""
    return StatsCollector(
        {
            "incomplete_heaps_evicted": ("input_incomplete_heaps", "help text 1"),
            "too_old_heaps": ("input_too_old_heaps", "help text 2"),
        },
        labelnames=("label1", "label2"),
        namespace="test",
        registry=registry,
    )


class TestStatsCollector:
    """Tests for :class:`~katgpucbf.recv.StatsCollector`."""

    def test_initial(self, registry: CollectorRegistry, stats_collector: StatsCollector) -> None:
        """Test state before adding any streams."""
        metrics = list(registry.collect())
        assert metrics[0].name == "test_input_incomplete_heaps"
        assert metrics[0].documentation == "help text 1"
        assert not metrics[0].samples
        assert metrics[1].name == "test_input_too_old_heaps"
        assert metrics[1].documentation == "help text 2"
        assert not metrics[1].samples

    def test_add_stream_bad_labels_length(self, stats_collector: StatsCollector) -> None:
        """Test exception when adding a stream with the wrong number of labels."""
        with pytest.raises(ValueError):
            stats_collector.add_stream(mock_stream(), ["value1"])

    def test_add_stream_missing_stat(self, stats_collector: StatsCollector) -> None:
        """Test exception when adding a stream that does not export a requested statistic."""
        with pytest.raises(IndexError):
            stats_collector.add_stream(mock_stream([]), ["value1", "value2"])

    def test_basic(self, registry: CollectorRegistry, stats_collector: StatsCollector) -> None:
        """Test basic functionality with one stream."""
        stream = mock_stream()
        inc_stat(stream, "incomplete_heaps_evicted", 2)
        inc_stat(stream, "too_old_heaps", 5)
        labels = {"label1": "value1", "label2": "value2"}
        now = 1234567890.5
        with mock.patch("time.time", return_value=now):
            stats_collector.add_stream(stream, labels.values())
        assert registry.get_sample_value("test_input_incomplete_heaps_total", labels) == 2
        assert registry.get_sample_value("test_input_too_old_heaps_total", labels) == 5
        assert registry.get_sample_value("test_input_incomplete_heaps_created", labels) == now
        assert registry.get_sample_value("test_input_too_old_heaps_created", labels) == now

        # Update, check that the updates are collected, but `created` is not changed
        inc_stat(stream, "incomplete_heaps_evicted", 4)
        inc_stat(stream, "too_old_heaps", 10)
        assert registry.get_sample_value("test_input_incomplete_heaps_total", labels) == 6
        assert registry.get_sample_value("test_input_too_old_heaps_total", labels) == 15
        assert registry.get_sample_value("test_input_incomplete_heaps_created", labels) == now
        assert registry.get_sample_value("test_input_too_old_heaps_created", labels) == now

        # Garbage-collect the heap, and ensure that its stats are retained
        weak = weakref.ref(stream)
        del stream
        # Do it multiple times because some Python implementations need to be
        # pushed to collect everything.
        for _ in range(5):
            gc.collect()
        assert weak() is None, "Stream was not garbage collected"

        assert registry.get_sample_value("test_input_incomplete_heaps_total", labels) == 6
