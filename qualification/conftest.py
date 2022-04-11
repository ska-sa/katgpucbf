"""Fixtures and options for qualification testing of the correlator."""
import json
import logging
import time
from dataclasses import dataclass

import aiokatcp
import pytest


def pytest_addoption(parser, pluginmanager):  # noqa: D103
    parser.addoption("--image-override", action="append", metavar="NAME:IMAGE:TAG", help="Override a single image")
    parser.addini("master_controller_host", "Hostname (or IP address) of the SDP master controller", type="string")
    parser.addini("master_controller_port", "TCP port of the SDP master controller", type="string", default="5001")


@pytest.fixture(params=[4, 5])
def n_antennas(request):  # noqa: D401
    """Number of antennas, i.e. size of the array."""
    return request.param


@pytest.fixture(params=[4096, 8192])
def n_channels(request):  # noqa: D401
    """Number of channels for the channeliser."""
    return request.param


# TODO: We may want to keep these in some kind of central place.
# Though having said that, it may end up making more overhead than it's worth.
@dataclass
class Band:
    """Holds presets for a known band."""

    adc_sample_rate: float
    centre_frequency: float


BANDS = {
    "l": Band(adc_sample_rate=1712e6, centre_frequency=1284e6),
    "u": Band(adc_sample_rate=1088e6, centre_frequency=816e6),
}


@pytest.fixture(params=["l", "u"])
def band(request) -> str:  # noqa: D104
    """Band ID."""
    return request.param


@pytest.fixture
def int_time() -> float:  # noqa: D104
    """Integration time in seconds."""
    return 0.5


@pytest.fixture
async def correlator_config(pytestconfig, n_antennas: int, n_channels: int, band: str, int_time: float) -> dict:
    """Produce the configuration dict from the parsed command-line arguments."""
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
    dig_number = 800  # Avoid confusion with real antennas

    adc_sample_rate = BANDS[band].adc_sample_rate
    centre_frequency = BANDS[band].centre_frequency

    for pol in ["v", "h"]:
        name = f"m{dig_number}{pol}"
        dig_names.append(name)
        config["outputs"][name] = {
            "type": "sim.dig.raw_antenna_voltage",
            "band": band,
            "adc_sample_rate": adc_sample_rate,
            "centre_frequency": centre_frequency,
            "antenna": f"m{dig_number}, 0:0:0, 0:0:0, 0, 0",
        }
    config["outputs"]["antenna_channelised_voltage"] = {
        "type": "gpucbf.antenna_channelised_voltage",
        # Cycle through digitisers as necessary
        "src_streams": [dig_names[i % len(dig_names)] for i in range(2 * n_antennas)],
        "input_labels": [f"m{800 + i}{pol}" for i in range(n_antennas) for pol in ["v", "h"]],
        "n_chans": n_channels,
    }
    config["outputs"]["baseline_correlation_products"] = {
        "type": "gpucbf.baseline_correlation_products",
        "src_streams": ["antenna_channelised_voltage"],
        "int_time": int_time,
    }

    return config


@pytest.fixture
async def correlator(pytestconfig, correlator_config):
    """Start a correlator using the SDP master controller.

    Shut the correlator down afterwards also.
    """
    host = pytestconfig.getini("master_controller_host")
    port = int(pytestconfig.getini("master_controller_port"))
    name = f"qualification_correlator_{int(time.time())}"
    client = await aiokatcp.Client.connect(host, port)
    try:
        reply, _ = await client.request("product-configure", name, json.dumps(correlator_config))
        host = reply[1].decode()
        port = int(reply[2].decode())
        logging.info(f"Product controller is at {host}:{port}")
        yield
    except ConnectionError:
        logging.exception("unable to connect")
        raise
    except aiokatcp.FailReply:
        logging.exception("Something went wrong with starting the correlator!")
        raise
    finally:

        await client.request("product-deconfigure", name)
