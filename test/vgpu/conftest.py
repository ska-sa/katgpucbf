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

"""Fixtures for use in vgpu unit tests."""

import functools
from collections.abc import Buffer, Callable, Iterable
from typing import Any

import pytest

from katgpucbf.vgpu.engine import VEngine
from katgpucbf.vgpu.main import make_engine, parse_args


@pytest.fixture
def make_engine_impl(engine_arglist: list[str]) -> Callable[[], VEngine]:
    """Create a dummy :class:`.VEngine` for unit testing.

    The command-line arguments are provided by a class-level `engine_arglist`
    fixture.
    """
    args = parse_args(engine_arglist)
    return functools.partial(make_engine, args)


@pytest.fixture
def sendmsg_packets(monkeypatch: pytest.MonkeyPatch) -> list[bytes]:
    """Mock out socket.sendmsg to append packets to a list.

    The value of this fixture is the list to which packets are appended.
    This does not capture all features of sendmsg; it is intended only
    for use with :mod:`katgpucbf.vgpu.send`.
    """

    def my_sendmsg(self, buffers: Iterable[Buffer], ancdata: Iterable, flags: int = 0, address: Any = None) -> int:
        packet = b"".join(buffers)
        packets.append(packet)
        return len(packet)

    packets: list[bytes] = []
    monkeypatch.setattr("socket.socket.sendmsg", my_sendmsg)
    return packets
