################################################################################
# Copyright (c) 2024, National Research Foundation (SARAO)
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

"""Unit test :mod:`katgpucbf.fgpu.accum."""

import pytest

from katgpucbf.fgpu.accum import Accum, Measurement


@pytest.fixture
def empty_accum() -> Accum:
    """Accum in initial state."""
    return Accum(100, 0)


@pytest.fixture
def valid_accum() -> Accum:
    """Accum with valid data."""
    accum = Accum(100, 0)
    accum.add(200, 220, 200)
    return accum


@pytest.fixture
def invalid_accum() -> Accum:
    """Accum with missing data."""
    accum = Accum(100, 0)
    accum.add(200, 220, None)
    return accum


@pytest.mark.parametrize(
    "start_timestamp, stop_timestamp",
    [
        (0, 100),  # Before existing data
        (210, 230),  # Overlaps existing data
        (250, 249),  # Negative length
        (220, 301),  # Overlaps window boundary
    ],
)
def test_bad_add(valid_accum: Accum, start_timestamp: int, stop_timestamp: int) -> None:
    """Bad calls to :meth:`.Accum.add` raise :exc:`ValueError`."""
    with pytest.raises(ValueError):
        valid_accum.add(start_timestamp, stop_timestamp, 1)


def test_add_contiguous(valid_accum: Accum) -> None:
    """Extending contiguously does not invalidate."""
    assert valid_accum.add(220, 250, 10) is None
    assert valid_accum.add(250, 300, 90) == Measurement(200, 300, 300)
    assert valid_accum.add(500, 600, 200) == Measurement(500, 600, 200)


def test_add_to_invalid(invalid_accum: Accum) -> None:
    """Extending an invalid window does not validate it."""
    assert invalid_accum.add(220, 300, 10) == Measurement(200, 300, None)


def test_add_gap(valid_accum: Accum) -> None:
    """Extending a window with a gap invalidates it."""
    assert valid_accum.add(230, 250, 10) is None
    assert valid_accum.add(250, 300, 20) == Measurement(200, 300, None)


def test_add_skip(valid_accum: Accum) -> None:
    """Adding to a new window returns the existing one, invalidated."""
    assert valid_accum.add(500, 550, 10) == Measurement(200, 300, None)
    assert valid_accum.add(550, 600, 40) == Measurement(500, 600, 50)


def test_add_full_skip(valid_accum: Accum) -> None:
    """Adding a complete new window ignores the previous one."""
    assert valid_accum.add(500, 600, 700) == Measurement(500, 600, 700)


def test_add_none(valid_accum: Accum) -> None:
    """Adding with power=None invalidates."""
    assert valid_accum.add(220, 250, None) is None
    assert valid_accum.add(250, 300, 10) == Measurement(200, 300, None)
