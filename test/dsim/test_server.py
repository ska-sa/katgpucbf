################################################################################
# Copyright (c) 2021-2022, 2024-2025, National Research Foundation (SARAO)
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

import asyncio
import re
from collections.abc import AsyncGenerator, Sequence

import aiokatcp
import numpy as np
import pytest

from katgpucbf import DIG_HEAP_SAMPLES, DIG_SAMPLE_BITS
from katgpucbf.dsim.send import HeapSet, Sender
from katgpucbf.dsim.server import DeviceServer
from katgpucbf.dsim.signal import parse_signals
from katgpucbf.send import DescriptorSender
from katgpucbf.spead import DIGITISER_STATUS_SATURATION_COUNT_SHIFT, DIGITISER_STATUS_SATURATION_FLAG_BIT

from .conftest import ADC_SAMPLE_RATE, SIGNAL_HEAPS


@pytest.fixture
async def katcp_server(
    sender: Sender, heap_sets: Sequence[HeapSet], descriptor_sender: DescriptorSender
) -> AsyncGenerator[DeviceServer, None]:
    """A :class:`~katgpucbf.dsim.server.DeviceServer`."""
    signals_str = "cw(0.2, 123); cw(0.3, 456);"
    dither_seed = 42
    server = DeviceServer(
        sender=sender,
        descriptor_sender=descriptor_sender,
        spare=heap_sets[1],
        adc_sample_rate=ADC_SAMPLE_RATE,
        sample_bits=DIG_SAMPLE_BITS,
        dither_seed=dither_seed,
        host="127.0.0.1",
        port=0,
    )
    await server.set_signals(parse_signals(signals_str), signals_str, SIGNAL_HEAPS)
    await server.start()
    yield server
    await server.stop()


@pytest.fixture
async def katcp_client(katcp_server: DeviceServer) -> AsyncGenerator[aiokatcp.Client, None]:
    """A katcp client connection to :func:`katcp_server`."""
    host, port = katcp_server.sockets[0].getsockname()[:2]
    async with asyncio.timeout(5):  # To fail the test quickly if unable to connect
        client = await aiokatcp.Client.connect(host, port)
    yield client
    client.close()
    await client.wait_closed()


async def test_sensors(katcp_server: DeviceServer, katcp_client: aiokatcp.Client) -> None:
    """Test the initial sensor values."""
    assert await katcp_client.sensor_value("signals-orig", str) == "cw(0.2, 123); cw(0.3, 456);"
    assert await katcp_client.sensor_value("signals", str) == "cw(0.2, 123); cw(0.3, 456);"
    assert await katcp_client.sensor_value("adc-sample-rate") == ADC_SAMPLE_RATE
    assert await katcp_client.sensor_value("sample-bits") == DIG_SAMPLE_BITS
    assert await katcp_client.sensor_value("max-period") == DIG_HEAP_SAMPLES * SIGNAL_HEAPS
    assert await katcp_client.sensor_value("period") == SIGNAL_HEAPS


@pytest.mark.parametrize("period", [8192, None])
async def test_signals(
    katcp_server: DeviceServer,
    katcp_client: aiokatcp.Client,
    sender: Sender,
    heap_sets: Sequence[HeapSet],
    period: int | None,
    mocker,
) -> None:
    """Test the ``?signals`` katcp command."""
    # CW with magnitude 2 should saturate 2/3 of the time. The frequency is
    # set low enough that some heaps will have no saturation.
    signals_str = "cw(0.0, 2e4); cw(2.0, 2e4);"
    set_heaps = mocker.patch.object(sender, "set_heaps", autospec=True, return_value=1234567)
    args: list = [signals_str]
    if period is not None:
        args.append(period)
    reply, _ = await katcp_client.request("signals", *args)
    assert reply == [b"1234567"]
    assert await katcp_client.sensor_value("steady-state-timestamp", int) == 1234567
    set_heaps.assert_called_once_with(heap_sets[0])
    # Check that pol 0 is now indeed all zeros (and pol 1 is not).
    payload = heap_sets[0].data["payload"]
    status = heap_sets[0].data["digitiser_status"]
    np.testing.assert_equal(payload.isel(pol=0).data, 0)
    assert not np.all(payload.isel(pol=1).data == 0)
    # Check that the saturation flag is consistent with the saturation count
    np.testing.assert_equal(
        (status.data & (1 << DIGITISER_STATUS_SATURATION_FLAG_BIT)) > 0,
        (status.data >> DIGITISER_STATUS_SATURATION_COUNT_SHIFT) > 0,
    )
    if period is None:  # The tests below depend on having enough unique data
        # Check that pol 1 is saturated about as much as expected
        assert np.mean(status.isel(pol=1).data >> 32) / DIG_HEAP_SAMPLES == pytest.approx(2 / 3, abs=0.01)
        # Check that some heaps are entirely unsaturated, so that the
        # saturation flag consistency test is not vacuous.
        assert not np.all(status.isel(pol=1).data & (1 << DIGITISER_STATUS_SATURATION_FLAG_BIT))
    # Check that sensors were updated
    assert await katcp_client.sensor_value("signals-orig", str) == signals_str
    assert await katcp_client.sensor_value("period") == (period or DIG_HEAP_SAMPLES * SIGNAL_HEAPS)
    assert parse_signals(await katcp_client.sensor_value("signals", str)) == parse_signals(signals_str)


@pytest.mark.parametrize(
    "spec,match",
    [
        ("foo", "Unknown variable 'foo'"),
        ("nodither(1.0) + 0.0", "Signal 'nodither(1.0)' cannot be used in a larger expression"),
    ],
)
async def test_signals_unparsable(
    katcp_server: DeviceServer, katcp_client: aiokatcp.Client, mocker, spec: str, match: str
) -> None:
    """Test that ``?signals`` with an invalid signal specification fails gracefully."""
    with pytest.raises(aiokatcp.FailReply, match=re.escape(match)):
        await katcp_client.request("signals", spec)


async def test_signals_wrong_length(katcp_server: DeviceServer, katcp_client: aiokatcp.Client, mocker) -> None:
    """Test that ``?signals`` fails gracefully when given the wrong number of signals."""
    with pytest.raises(aiokatcp.FailReply, match="expected 2 signals, received 1"):
        await katcp_client.request("signals", "cw(0, 0);")


async def test_time(katcp_server: DeviceServer, katcp_client: aiokatcp.Client, mocker) -> None:
    """Test ?time request."""
    mocker.patch("time.time", return_value=1234567890.0)
    reply, _ = await katcp_client.request("time")
    assert aiokatcp.decode(float, reply[0]) == 1234567890.0
