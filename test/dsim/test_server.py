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

"""Unit tests for katcp server."""

from typing import AsyncGenerator, Sequence

import aiokatcp
import async_timeout
import numpy as np
import pytest

from katgpucbf import DIG_HEAP_SAMPLES, DIG_SAMPLE_BITS
from katgpucbf.dsim.descriptors import DescriptorSender
from katgpucbf.dsim.send import HeapSet, Sender
from katgpucbf.dsim.server import DeviceServer
from katgpucbf.dsim.signal import parse_signals

from .. import get_sensor
from .conftest import ADC_SAMPLE_RATE, SIGNAL_HEAPS


@pytest.fixture
async def katcp_server(
    sender: Sender, heap_sets: Sequence[HeapSet], descriptor_sender: DescriptorSender
) -> AsyncGenerator[DeviceServer, None]:  # noqa: D401
    """A :class:`~katgpucbf.dsim.server.DeviceServer`."""
    # Make up a bogus signal. It's not actually populated in heap_sets
    signals_str = "cw(0.2, 123); cw(0.3, 456);"

    server = DeviceServer(
        sender=sender,
        descriptor_sender=descriptor_sender,
        spare=heap_sets[1],
        adc_sample_rate=ADC_SAMPLE_RATE,
        sample_bits=DIG_SAMPLE_BITS,
        first_timestamp=0,
        dither_seed=42,
        signals_str=signals_str,
        signals=parse_signals(signals_str),
        host="127.0.0.1",
        port=0,
    )
    await server.start()
    yield server
    await server.stop()


@pytest.fixture
async def katcp_client(katcp_server: DeviceServer) -> AsyncGenerator[aiokatcp.Client, None]:  # noqa: D401
    """A katcp client connection to :func:`katcp_server`."""
    host, port = katcp_server.sockets[0].getsockname()[:2]
    async with async_timeout.timeout(5):  # To fail the test quickly if unable to connect
        client = await aiokatcp.Client.connect(host, port)
    yield client
    client.close()
    await client.wait_closed()


async def test_sensors(katcp_server: DeviceServer, katcp_client: aiokatcp.Client) -> None:
    """Test the initial sensor values."""
    assert await get_sensor(katcp_client, "signals-orig") == "cw(0.2, 123); cw(0.3, 456);"
    assert await get_sensor(katcp_client, "signals") == "cw(0.2, 123); cw(0.3, 456);"
    assert await get_sensor(katcp_client, "adc-sample-rate") == ADC_SAMPLE_RATE
    assert await get_sensor(katcp_client, "sample-bits") == DIG_SAMPLE_BITS
    assert await get_sensor(katcp_client, "period") == DIG_HEAP_SAMPLES * SIGNAL_HEAPS


async def test_signals(
    katcp_server: DeviceServer, katcp_client: aiokatcp.Client, sender: Sender, heap_sets: Sequence[HeapSet], mocker
) -> None:
    """Test the ``?signals`` katcp command."""
    signals_str = "cw(0.0, 1000.0); cw(1.0, 1000.0);"
    set_heaps = mocker.patch.object(sender, "set_heaps", autospec=True, return_value=1234567)
    reply, _ = await katcp_client.request("signals", signals_str)
    assert reply == [b"1234567"]
    set_heaps.assert_called_once_with(heap_sets[1])
    # Check that pol 0 is now indeed all zeros (and pol 1 is not).
    np.testing.assert_equal(heap_sets[1].data["payload"].isel(pol=0).data, 0)
    assert not np.all(heap_sets[1].data["payload"].isel(pol=1).data == 0)
    # Check that sensors were updated
    assert await get_sensor(katcp_client, "signals-orig") == signals_str
    assert parse_signals(await get_sensor(katcp_client, "signals")) == parse_signals(signals_str)


async def test_signals_unparsable(katcp_server: DeviceServer, katcp_client: aiokatcp.Client, mocker) -> None:
    """Test that ``?signals`` with an invalid signal specification fails gracefully."""
    with pytest.raises(aiokatcp.FailReply, match="Unknown variable 'foo'"):
        await katcp_client.request("signals", "foo")


async def test_signals_wrong_length(katcp_server: DeviceServer, katcp_client: aiokatcp.Client, mocker) -> None:
    """Test that ``?signals`` fails gracefully when given the wrong number of signals."""
    with pytest.raises(aiokatcp.FailReply, match="expected 2 signals, received 1"):
        await katcp_client.request("signals", "cw(0, 0);")
