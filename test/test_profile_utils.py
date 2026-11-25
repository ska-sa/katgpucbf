################################################################################
# Copyright (c) 2026, National Research Foundation (SARAO)
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

"""Tests for :mod:`katgpucbf.profile_utils`."""

import unittest.mock
from typing import override

import numpy as np
import pytest
from katsdpsigproc.abc import AbstractCommandQueue
from katsdpsigproc.accel import Dimension, IOSlot, Operation

from katgpucbf.profile_utils import replicate_slots


class DummyOperation(Operation):
    """Subclass :class:`katsdpsigproc.accel.Operation` to provide a test fixture."""

    def __init__(self, command_queue: AbstractCommandQueue) -> None:
        super().__init__(command_queue)
        self.slots["in"] = IOSlot((Dimension(1000000, min_padded_size=1000064),), np.uint8)
        self.slots["out"] = IOSlot((500, 1000), np.float32)

    @override
    def _run(self) -> None:
        pass  # pragma: noqa


class TestReplicateSlots:
    """Test :meth:`katgpucbf.profile_utils.replicate_slots`."""

    @pytest.mark.parametrize("slots,expected_slots", [(None, {"in", "out"}), (["in"], {"in"})])
    def test_count(
        self, command_queue: AbstractCommandQueue, slots: list[str] | None, expected_slots: set[str]
    ) -> None:
        """Test with an explicit count."""
        op = DummyOperation(command_queue)
        op.ensure_all_bound()
        buffers = replicate_slots(op, slots, count=3)
        assert len(buffers) == 3
        for buffer_set in buffers:
            assert set(buffer_set) == expected_slots
            if "in" in expected_slots:
                assert buffer_set["in"].shape == (1000000,)
                assert buffer_set["in"].dtype == np.dtype(np.uint8)
                assert buffer_set["in"].padded_shape == (1000064,)
            if "out" in expected_slots:
                assert buffer_set["out"].shape == (500, 1000)
                assert buffer_set["out"].dtype == np.dtype(np.float32)
        if "in" in expected_slots:
            assert buffers[0]["in"] is op.buffer("in")
        if "out" in expected_slots:
            assert buffers[0]["out"] is op.buffer("out")

    @unittest.mock.patch("katgpucbf.profile_utils._get_cache_size", return_value=4 * 1024 * 1024)
    def test_cache_factor(self, command_queue: AbstractCommandQueue) -> None:
        """Test with a `cache_factor` argument (explicit and implicit)."""
        op = DummyOperation(command_queue)
        op.ensure_all_bound()
        buffers = replicate_slots(op, ["in"], cache_factor=1)
        assert len(buffers) == 5
        buffers = replicate_slots(op)
        assert len(buffers) == 3

    def test_size(self, command_queue: AbstractCommandQueue) -> None:
        """Test with an explicit target `size`."""
        op = DummyOperation(command_queue)
        op.ensure_all_bound()
        buffers = replicate_slots(op, ["in"], size=4 * 1024 * 1024)
        assert len(buffers) == 5
        buffers = replicate_slots(op, size=9000000)
        assert len(buffers) == 3  # Fits exactly
        buffers = replicate_slots(op, size=9000001)
        assert len(buffers) == 4  # Target is one byte too big for 3
