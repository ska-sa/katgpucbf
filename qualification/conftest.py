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

"""Fixtures and options for qualification testing of the correlator."""

import asyncio
import inspect
import json
import logging
import subprocess
import time
from typing import AsyncGenerator, Generator

import aiokatcp
import matplotlib.style
import pytest
from async_timeout import timeout
from katsdpservices import get_interface_address

from katgpucbf.meerkat import BANDS

from . import BaselineCorrelationProductsReceiver, CorrelatorRemoteControl, get_dsim_endpoint
from .reporter import Reporter

logger = logging.getLogger(__name__)


def pytest_addoption(parser, pluginmanager):  # noqa: D103
    # I'm adding the image override as a cmd-line parameter. It seems best (at
    # this stage anyway) that it be explicit which image you're testing.
    parser.addoption(
        "--image-override", action="append", required=True, metavar="NAME:IMAGE:TAG", help="Override a single image"
    )
    # On the other hand, the MC details aren't likely to change frequently, so
    # an INI file is probably appropriate.
    parser.addini("master_controller_host", "Hostname (or IP address) of the SDP master controller", type="string")
    parser.addini("master_controller_port", "TCP port of the SDP master controller", type="string", default="5001")
    # I'm on the fence about these. Probably on a given machine, you'd set and
    # forget.
    parser.addini("interface", "Name of network to use for ingest.", type="string")
    parser.addini("use_ibv", "Use ibverbs", type="bool", default="false")
    parser.addini("product_name", "Name of subarray product", type="string", default="qualification_correlator")


def pytest_report_collectionfinish(config):  # noqa: D103
    # Using this hook to collect configuration information, because it's run
    # once, after collection but before the actual tests. Couldn't really find a
    # better place, and I did look around quite a bit.
    git_information = subprocess.check_output(["git", "describe", "--tags", "--dirty", "--always"]).decode()
    logger.info("Git information: %s", git_information)
    config._report_log_plugin._write_json_data(
        {"$report_type": "TestConfiguration", "Test Suite Git Info": git_information}
    )


# Need to redefine this from pytest-asyncio to have it at session scope
@pytest.fixture(scope="session")
def event_loop():  # noqa: D103
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def n_antennas():  # noqa: D401
    """Number of antennas, i.e. size of the array."""
    return 8


@pytest.fixture(scope="session")
def n_dsims():  # noqa: D401
    """Number of simulated digitisers."""
    return 1


@pytest.fixture(
    scope="session",
    params=[
        8192,
    ],
)
def n_channels(request):  # noqa: D401
    """Number of channels for the channeliser."""
    return request.param


@pytest.fixture(
    scope="session",
    params=[
        "l",
    ],
)
def band(request) -> str:  # noqa: D104
    """Band ID."""
    return request.param


@pytest.fixture(scope="session")
def int_time() -> float:  # noqa: D104
    """Integration time in seconds."""
    return 0.5


@pytest.fixture
def pdf_report(request) -> Reporter:
    """Fixture for logging steps in a test."""
    data = [{"$msg_type": "test_info", "blurb": inspect.getdoc(request.node.function), "test_start": time.time()}]
    request.node.user_properties.append(("pdf_report_data", data))
    return Reporter(data)


@pytest.fixture(autouse=True)
def matplotlib_report_style() -> Generator[None, None, None]:
    """Set the style of all matplotlib plots."""
    with matplotlib.style.context("ggplot"):
        yield


@pytest.fixture(scope="session")
async def correlator_config(
    pytestconfig, n_antennas: int, n_channels: int, n_dsims: int, band: str, int_time: float
) -> dict:
    """Produce the configuration dict from the given parameters."""
    # Adapted from `sim_correlator.py` but with logic for using multiple dsims
    # removed. For the time being, we're going to use a single dsim for
    # consistency with MeerKAT's qualification testing.

    config: dict = {
        "version": "3.1",
        "config": {},
        "inputs": {},
        "outputs": {},
    }
    if pytestconfig.getoption("image_override") is not None:
        image_overrides = {}
        for override in pytestconfig.getoption("image_override"):
            name, image = override.split(":", 1)
            image_overrides[name] = image
        config["config"]["image_overrides"] = image_overrides

    dig_names = []

    adc_sample_rate = BANDS[band].adc_sample_rate
    centre_frequency = BANDS[band].centre_frequency

    for i in range(n_dsims):
        dsim_name = f"dsim{i:03}"
        for pol in ["v", "h"]:
            name = f"{dsim_name}{pol}"
            dig_names.append(name)
            config["outputs"][name] = {
                "type": "sim.dig.raw_antenna_voltage",
                "band": band,
                "adc_sample_rate": adc_sample_rate,
                "centre_frequency": centre_frequency,
                "antenna": f"{dsim_name}, 0:0:0, 0:0:0, 0, 0",
            }
    config["outputs"]["antenna_channelised_voltage"] = {
        "type": "gpucbf.antenna_channelised_voltage",
        "src_streams": [dig_names[i % len(dig_names)] for i in range(2 * n_antennas)],
        # m8xx is used to avoid possible confusion with real antennas
        "input_labels": [f"m{800 + i}{pol}" for i in range(n_antennas) for pol in ["v", "h"]],
        "n_chans": n_channels,
    }
    config["outputs"]["baseline_correlation_products"] = {
        "type": "gpucbf.baseline_correlation_products",
        "src_streams": ["antenna_channelised_voltage"],
        "int_time": int_time,
    }

    logger.info(f"Config for {n_antennas}-A, {n_channels}-chan {band}-band correlator generated.")
    return config


@pytest.fixture(scope="session")
async def session_correlator(
    pytestconfig, correlator_config, band: str, n_dsims: int
) -> AsyncGenerator[CorrelatorRemoteControl, None]:
    """Start a correlator using the SDP master controller.

    Shut the correlator down afterwards also.

    Generally this fixture should not be used directly. Use :meth:`correlator`
    instead, which will reuse the same correlator across multiple tests.
    """
    host = pytestconfig.getini("master_controller_host")
    port = int(pytestconfig.getini("master_controller_port"))
    try:
        logger.debug("Connecting to master controller at %s:%d to try and create a correlator.", host, port)
        async with timeout(10):
            master_controller_client = await aiokatcp.Client.connect(host, port)
    except (ConnectionError, asyncio.TimeoutError):
        logger.exception("unable to connect to master controller!")
        raise

    product_name = pytestconfig.getini("product_name")
    try:
        reply, _ = await master_controller_client.request(
            "product-configure", product_name, json.dumps(correlator_config)
        )

    except aiokatcp.FailReply:
        logger.exception("Something went wrong with starting the correlator!")
        raise

    product_controller_host = reply[1].decode()
    product_controller_port = int(reply[2].decode())
    logger.info(
        "Correlator created, connecting to product controller at %s:%d",
        product_controller_host,
        product_controller_port,
    )
    try:
        pcc = await aiokatcp.Client.connect(product_controller_host, product_controller_port)
        dsim_clients = []
        for i in range(n_dsims):
            dsim_host, dsim_port = await get_dsim_endpoint(pcc, BANDS[band].adc_sample_rate, i)
            dsim_clients.append(await aiokatcp.Client.connect(dsim_host, dsim_port))

        remote_control = await CorrelatorRemoteControl.connect(pcc, dsim_clients, correlator_config)
        yield remote_control

        logger.info("Tearing down correlator.")
        clients = dsim_clients + [pcc]
        for client in clients:
            client.close()
        await asyncio.gather(*[client.wait_closed() for client in clients])

    finally:
        # In case anything does go wrong, we want to make sure that we the
        # deconfigure the product.
        await master_controller_client.request("product-deconfigure", product_name)
        master_controller_client.close()
        await master_controller_client.wait_closed()


@pytest.fixture
async def correlator(
    session_correlator: CorrelatorRemoteControl,
) -> AsyncGenerator[CorrelatorRemoteControl, None]:
    """Set up a correlator for a single test.

    The returned correlator might not be specific to this test, but it will have
    been reset to a default state, with the dsim outputting zeros.
    """
    # Reset the correlator to default state
    pcc = session_correlator.product_controller_client
    await pcc.request("gain-all", "antenna_channelised_voltage", "default")
    await pcc.request("delays", "antenna_channelised_voltage", 0, *(["0,0:0,0"] * session_correlator.n_inputs))
    await asyncio.gather(*[client.request("signals", "0;0;") for client in session_correlator.dsim_clients])
    await pcc.request("capture-start", "baseline_correlation_products")

    yield session_correlator

    await pcc.request("capture-stop", "baseline_correlation_products")


@pytest.fixture
async def receive_baseline_correlation_products(
    pytestconfig, correlator: CorrelatorRemoteControl
) -> AsyncGenerator[BaselineCorrelationProductsReceiver, None]:
    """Create a spead2 receive stream for ingesting X-engine output."""
    interface_address = get_interface_address(pytestconfig.getini("interface"))
    # This will require running pytest with spead2_net_raw which is unusual.
    use_ibv = pytestconfig.getini("use_ibv")

    receiver = BaselineCorrelationProductsReceiver(
        correlator=correlator,
        interface_address=interface_address,
        use_ibv=use_ibv,
    )
    # Ensure that the data is flowing, and that we throw away any data that
    # predates the start of this test (to prevent any state leaks from previous
    # tests).
    _, chunk = await receiver.next_complete_chunk(max_delay=0)
    receiver.stream.add_free_chunk(chunk)
    yield receiver
    receiver.stream.stop()
