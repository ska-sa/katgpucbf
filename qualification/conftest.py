"""Fixtures and options for qualification testing of the correlator."""
import json
import logging

import aiokatcp
import pytest
import spead2.recv
from katsdpservices import get_interface_address
from katsdptelstate.endpoint import endpoint_list_parser

from . import BANDS, Band, CorrelatorRemoteControl, create_stream, get_dsim_endpoint, get_sensor_val

logger = logging.getLogger(__name__)


def pytest_addoption(parser, pluginmanager):  # noqa: D103
    # I'm adding the image override as a cmd-line parameter. It seems best (at
    # this stage anyway) that it be explicit what image you're testing.
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


@pytest.fixture(params=[2, 3])
def n_antennas(request):  # noqa: D401
    """Number of antennas, i.e. size of the array."""
    return request.param


@pytest.fixture(params=[4096, 8192])
def n_channels(request):  # noqa: D401
    """Number of channels for the channeliser."""
    return request.param


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
    """Produce the configuration dict from the given parameters."""
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
        # Right now I'm just using a single dsim, but this logic will scale for more.
        "src_streams": [dig_names[i % len(dig_names)] for i in range(2 * n_antennas)],
        "input_labels": [f"m{800 + i}{pol}" for i in range(n_antennas) for pol in ["v", "h"]],
        "n_chans": n_channels,
    }
    config["outputs"]["baseline_correlation_products"] = {
        "type": "gpucbf.baseline_correlation_products",
        "src_streams": ["antenna_channelised_voltage"],
        "int_time": int_time,
    }

    logger.debug("Config for correlator generated.")
    return config


@pytest.fixture
async def correlator(pytestconfig, correlator_config, band: Band):
    """Start a correlator using the SDP master controller.

    Shut the correlator down afterwards also.
    """
    host = pytestconfig.getini("master_controller_host")
    port = int(pytestconfig.getini("master_controller_port"))
    try:
        logger.debug("Connecting to master controller to try and create a correlator.")
        client = await aiokatcp.Client.connect(host, port)
        async with client:
            reply, _ = await client.request(
                "product-configure", "qualification_correlator*", json.dumps(correlator_config)
            )
        product_controller_host = reply[1].decode()
        product_controller_port = int(reply[2].decode())
        logger.info(
            f"Product controller for qualification correlator is at {product_controller_host}:{product_controller_port}"
        )
        product_controller_client = await aiokatcp.Client.connect(product_controller_host, product_controller_port)
        dsim_host, dsim_port = await get_dsim_endpoint(product_controller_client, BANDS[band].adc_sample_rate)
        dsim_client = await aiokatcp.Client.connect(dsim_host, dsim_port)
        yield CorrelatorRemoteControl(product_controller_client, dsim_client)
    except ConnectionError:
        logger.exception("unable to connect")
        raise
    except aiokatcp.FailReply:
        logger.exception("Something went wrong with starting the correlator!")
        raise
    finally:
        dsim_client.close()
        await product_controller_client.request("product-deconfigure")
        product_controller_client.close()


@pytest.fixture
async def receive_stream(pytestconfig, correlator: CorrelatorRemoteControl) -> spead2.recv.ChunkRingStream:
    """Create a spead2 receive stream for ingesting X-engine output."""
    interface_address = get_interface_address(pytestconfig.getini("interface"))
    # This will require running pytest with spead2_net_raw which is unusual.
    use_ibv = pytestconfig.getini("use_ibv")

    # I'm still in two minds as to whether to pass these things around as data
    # members of the CorrelatorRemoteControl class, or to get them this way
    # via katcp.
    pc_client = correlator.product_controller_client
    n_bls = await get_sensor_val(pc_client, "baseline_correlation_products-n-bls")
    n_chans = await get_sensor_val(pc_client, "baseline_correlation_products-n-chans")
    n_chans_per_substream = await get_sensor_val(pc_client, "baseline_correlation_products-n-chans-per-substream")
    n_bits_per_sample = await get_sensor_val(pc_client, "baseline_correlation_products-xeng-out-bits-per-sample")
    n_spectra_per_acc = await get_sensor_val(pc_client, "baseline_correlation_products-n-accs")
    int_time = await get_sensor_val(pc_client, "baseline_correlation_products-int-time")
    n_samples_between_spectra = await get_sensor_val(pc_client, "antenna_channelised_voltage-n-samples-between-spectra")

    multicast_endpoints = [
        tuple(endpoint)
        for endpoint in endpoint_list_parser(7148)(
            await get_sensor_val(pc_client, "baseline_correlation_products-destination")
        )
    ]

    return create_stream(
        interface_address=interface_address,
        multicast_endpoints=multicast_endpoints,
        n_bls=n_bls,
        n_chans=n_chans,
        n_chans_per_substream=n_chans_per_substream,
        n_bits_per_sample=n_bits_per_sample,
        n_spectra_per_acc=n_spectra_per_acc,
        int_time=int_time,
        n_samples_between_spectra=n_samples_between_spectra,
        use_ibv=use_ibv,
    )
