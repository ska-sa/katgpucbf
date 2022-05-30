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
from typing import AsyncGenerator

import aiokatcp
import pytest
import spead2.recv
from async_timeout import timeout
from katsdpservices import get_interface_address

from katgpucbf import N_POLS
from katgpucbf.meerkat import BANDS

from . import CorrelatorRemoteControl, create_baseline_correlation_product_receive_stream, get_dsim_endpoint
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


def pytest_report_collectionfinish(config):  # noqa: D103
    # Using this hook to collect configuration information, because it's run
    # once, after collection but before the actual tests. Couldn't really find a
    # better place, and I did look around quite a bit.
    git_information = subprocess.check_output(["git", "describe", "--tags", "--dirty", "--always"]).decode()
    logger.info("Git information: %s", git_information)
    config._report_log_plugin._write_json_data(
        {"$report_type": "TestConfiguration", "Test Suite Git Info": git_information}
    )


@pytest.fixture(params=[4, 8, 14])
def n_antennas(request):  # noqa: D401
    """Number of antennas, i.e. size of the array."""
    return request.param


@pytest.fixture(
    params=[
        8192,
    ]
)
def n_channels(request):  # noqa: D401
    """Number of channels for the channeliser."""
    return request.param


@pytest.fixture(
    params=[
        "l",
    ]
)
def band(request) -> str:  # noqa: D104
    """Band ID."""
    return request.param


@pytest.fixture
def int_time() -> float:  # noqa: D104
    """Integration time in seconds."""
    return 0.5


@pytest.fixture
def pdf_report(request) -> Reporter:
    """Fixture for logging steps in a test."""
    data = [{"$msg_type": "test_info", "blurb": inspect.getdoc(request.node.function), "test_start": time.time()}]
    request.node.user_properties.append(("pdf_report_data", data))
    return Reporter(data)


@pytest.fixture
async def correlator_config(pytestconfig, n_antennas: int, n_channels: int, band: str, int_time: float) -> dict:
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
    antenna_name = "m800"  # Avoid confusion with real antennas

    adc_sample_rate = BANDS[band].adc_sample_rate
    centre_frequency = BANDS[band].centre_frequency

    for pol in ["v", "h"]:
        name = f"{antenna_name}{pol}"
        dig_names.append(name)
        config["outputs"][name] = {
            "type": "sim.dig.raw_antenna_voltage",
            "band": band,
            "adc_sample_rate": adc_sample_rate,
            "centre_frequency": centre_frequency,
            "antenna": f"{antenna_name}, 0:0:0, 0:0:0, 0, 0",
        }
    config["outputs"]["antenna_channelised_voltage"] = {
        "type": "gpucbf.antenna_channelised_voltage",
        "src_streams": [dig_names[i % N_POLS] for i in range(2 * n_antennas)],
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


@pytest.fixture
async def correlator(pytestconfig, correlator_config, band: str) -> AsyncGenerator[CorrelatorRemoteControl, None]:
    """Start a correlator using the SDP master controller.

    Shut the correlator down afterwards also.
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

    # We'll always name the correlator the same thing, so that if there are
    # zombies left behind from past runs, it'll bail straight away and alert
    # the user that there's a problem.
    product_name = "qualification_correlator"
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
        dsim_host, dsim_port = await get_dsim_endpoint(pcc, BANDS[band].adc_sample_rate)
        dsim_client = await aiokatcp.Client.connect(dsim_host, dsim_port)

        remote_control = await CorrelatorRemoteControl.connect(pcc, dsim_client, correlator_config)
        yield remote_control

        logger.info("Tearing down correlator.")
        dsim_client.close()
        pcc.close()
        await asyncio.gather(pcc.wait_closed(), dsim_client.wait_closed())

    finally:
        # In case anything does go wrong, we want to make sure that we the
        # deconfigure the product.
        await master_controller_client.request("product-deconfigure", product_name)
        master_controller_client.close()
        await master_controller_client.wait_closed()


@pytest.fixture
async def receive_baseline_correlation_products_stream(
    pytestconfig, correlator: CorrelatorRemoteControl
) -> spead2.recv.ChunkRingStream:
    """Create a spead2 receive stream for ingesting X-engine output."""
    interface_address = get_interface_address(pytestconfig.getini("interface"))
    # This will require running pytest with spead2_net_raw which is unusual.
    use_ibv = pytestconfig.getini("use_ibv")

    return create_baseline_correlation_product_receive_stream(
        interface_address=interface_address,
        multicast_endpoints=correlator.multicast_endpoints,  # type: ignore
        n_bls=correlator.n_bls,
        n_chans=correlator.n_chans,
        n_chans_per_substream=correlator.n_chans_per_substream,
        n_bits_per_sample=correlator.n_bits_per_sample,
        n_spectra_per_acc=correlator.n_spectra_per_acc,
        int_time=correlator.int_time,
        n_samples_between_spectra=correlator.n_samples_between_spectra,
        use_ibv=use_ibv,
    )
