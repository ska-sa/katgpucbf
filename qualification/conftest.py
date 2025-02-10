################################################################################
# Copyright (c) 2022-2024, National Research Foundation (SARAO)
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

"""Fixtures and options for qualification testing of the CBF."""

import ast
import asyncio
import copy
import inspect
import logging
import os
import subprocess
import time
from collections import deque, namedtuple
from collections.abc import AsyncGenerator, Generator, Iterable, Sequence

import matplotlib.style
import pytest
import pytest_asyncio
import pytest_check
from katsdpservices import get_interface_address

from katgpucbf.meerkat import BANDS

from .cbf import CBFCache, CBFRemoteControl, FailedCBF
from .recv import DEFAULT_TIMEOUT, BaselineCorrelationProductsReceiver, TiedArrayChannelisedVoltageReceiver
from .reporter import Reporter, custom_report_log

logger = logging.getLogger(__name__)
FULL_ANTENNAS = [1, 4, 8, 10, 16, 20, 32, 40, 55, 64, 65, 80]
pdf_report_data_key = pytest.StashKey[dict]()
_CAPTURE_TYPES = {"gpucbf.baseline_correlation_products", "gpucbf.tied_array_channelised_voltage"}


# Storing ini options this way makes pytest.ini easier to validate up-front.
IniOption = namedtuple("IniOption", ["name", "help", "type", "default"], defaults=[None])
ini_options = [
    IniOption(
        name="master_controller_host", help="Hostname (or IP address) of the SDP master controller", type="string"
    ),
    IniOption(
        name="master_controller_port", help="TCP port of the SDP master controller", type="string", default="5001"
    ),
    IniOption(
        name="prometheus_url", help="URL to Prometheus server for querying hardware configuration", type="string"
    ),
    IniOption(name="interface", help="Name of network to use for ingest.", type="string"),
    IniOption(
        name="interface_gbps", help="Maximum bandwidth to subscribe to on 'interface'", type="string", default="90"
    ),
    IniOption(name="use_ibv", help="Use ibverbs", type="bool", default=False),
    IniOption(name="cores", help="Space-separate list of cores to use for worker threads", type="args", default=[]),
    IniOption(name="product_name", help="Name of subarray product", type="string", default="qualification_cbf"),
    IniOption(name="tester", help="Name of person executing this qualification run", type="string", default="Unknown"),
    IniOption(
        name="default_antennas",
        help="Number of antennas for antenna-channelised-voltage tests",
        type="string",
        default="8",
    ),
    IniOption(name="max_antennas", help="Maximum number of antennas to test", type="string", default="8"),
    IniOption(
        name="wideband_channels",
        help="Space-separated list of channel counts to test in wideband",
        type="args",
        default=["8192"],
    ),
    IniOption(
        name="narrowband_channels",
        help="Space-separated list of channel counts to test in narrowband",
        type="args",
        default=["32768"],
    ),
    IniOption(
        name="narrowband_decimation",
        help="Space-separated list of narrowband decimation factors to test",
        type="args",
        default=["8"],
    ),
    IniOption(name="bands", help="Space-separated list of bands to test", type="args", default=["l"]),
    IniOption(name="beams", help="Number of beams to produce", type="string", default="4"),
    IniOption(name="raw_data", help="Include raw data for figures", type="bool", default=False),
]


def pytest_addoption(parser, pluginmanager):  # noqa: D103
    # I'm adding the image override as a cmd-line parameter. It seems best (at
    # this stage anyway) that it be explicit which image you're testing.
    parser.addoption(
        "--image-override", action="append", required=True, metavar="NAME:IMAGE:TAG", help="Override a single image"
    )
    for option in ini_options:
        parser.addini(*option)


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers and validate the pytest.ini file.

    .. note::
        This hook checks whether all the expected ini options are there, not
        whether they're correct or useful.
    """
    config.addinivalue_line("markers", "requirements(reqs): indicate which system engineering requirements are tested")
    config.addinivalue_line("markers", "name(name): human-readable name for the test")
    config.addinivalue_line("markers", "wideband_only: do not run the test in narrowband configurations")
    config.addinivalue_line(
        "markers", "no_capture_start([stream, ...]): do not issue capture-start (on all streams if none specified)"
    )
    for option in ini_options:
        assert config.getini(option.name) is not None, f"{option.name} missing from pytest.ini"


def pytest_report_collectionfinish(config: pytest.Config) -> None:  # noqa: D103
    # Using this hook to collect configuration information, because it's run
    # once, after collection but before the actual tests. Couldn't really find a
    # better place, and I did look around quite a bit.
    git_information = subprocess.check_output(["git", "describe", "--tags", "--dirty", "--always"]).decode()
    logger.info("Git information: %s", git_information)
    custom_report_log(
        config,
        {
            "$report_type": "TestConfiguration",
            "Tester": config.getini("tester"),
            "Test Suite Git Info": git_information,
        },
    )


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Dynamically parametrize number of antennas etc based on command line arguments."""
    if "n_antennas" in metafunc.fixturenames:
        rel_path = metafunc.definition.path.relative_to(metafunc.config.rootpath)
        max_antennas = int(metafunc.config.getini("max_antennas"))
        default_antennas = int(metafunc.config.getini("default_antennas"))
        if rel_path.parts[0] != "antenna_channelised_voltage":
            values = FULL_ANTENNAS
        else:
            values = [min(max_antennas, default_antennas)]
        values = [value for value in values if value <= max_antennas]
        metafunc.parametrize("n_antennas", values)
    if "band" in metafunc.fixturenames:
        metafunc.parametrize("band", metafunc.config.getini("bands"))
    if "n_channels" in metafunc.fixturenames or "narrowband_decimation" in metafunc.fixturenames:
        configs = [(int(n_channels), 1) for n_channels in metafunc.config.getini("wideband_channels")]
        if not metafunc.definition.get_closest_marker("wideband_only"):
            configs.extend(
                (int(n_channels), int(decimation))
                for decimation in metafunc.config.getini("narrowband_decimation")
                for n_channels in metafunc.config.getini("narrowband_channels")
            )
        metafunc.parametrize("n_channels, narrowband_decimation", configs)


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(session: pytest.Session, config: pytest.Config, items: list[pytest.Item]) -> None:
    """Reorder the tests to improve :class:`CBFCache` hit rate."""

    def key(item: pytest.Item) -> tuple:
        # This is a little hacky: we can't access fixture values directly,
        # only parametrisations. The antenna_channelised_voltage package
        # overrides the 'n_dsims' fixture, so it will almost always have
        # different configurations than the other directories. So we just
        # hard-code that into the sort key.
        rel_path = item.path.relative_to(config.rootpath)
        ans: list = [rel_path.parts[0] != "antenna_channelised_voltage"]
        callspec = getattr(item, "callspec", None)
        if callspec is not None:
            for name in ["n_antennas", "narrowband_decimation", "n_channels", "band"]:
                ans.append((name, callspec.params.get(name)))
        return tuple(ans)

    items.sort(key=key)
    # Make all tests run using the session-scoped event loop, so that they can
    # use session-scoped asynchronous fixtures.
    scope_marker = pytest.mark.asyncio(loop_scope="session")
    for item in items:
        if pytest_asyncio.is_async_test(item):
            item.add_marker(scope_marker, append=False)


@pytest.fixture(scope="package")
def n_dsims() -> int:
    """Number of simulated digitisers."""  # noqa: D401
    return 1


@pytest.fixture
def int_time() -> float:
    """Integration time in seconds."""
    return 0.5


@pytest.fixture(autouse=True)
def pdf_report(request, monkeypatch) -> Reporter:
    """Fixture for logging steps in a test."""
    reporter = Reporter(request.node.stash[pdf_report_data_key], raw_data=request.config.getini("raw_data"))
    orig_log_failure = pytest_check.check_log.log_failure
    orig_stack = inspect.stack

    def stack():
        # The real log_failure function constructs a backtrace, and inserting
        # our wrapper into the call stack messes that up. We need to have it
        # skip an extra level for each wrapper we're injecting.
        return orig_stack()[2:]

    def log_failure(msg="", check_str="", tb=None):
        __tracebackhide__ = True
        if check_str:
            reporter.failure(f"Failed assertion: {msg}: {check_str}")
        else:
            reporter.failure(f"Failed assertion: {msg}")
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(inspect, "stack", stack)
            return orig_log_failure(msg, check_str, tb)

    # Patch the central point where pytest-check logs failures so that we can
    # insert them into the test procedure.
    monkeypatch.setattr(pytest_check.check_log, "log_failure", log_failure)
    # context_manager uses `from .check_log import log_failure` so we have to
    # patch it under that name.
    monkeypatch.setattr(pytest_check.context_manager, "log_failure", log_failure)
    return reporter


@pytest.hookimpl(wrapper=True)
def pytest_runtest_setup(item) -> Generator[None, None, None]:
    """Set up the user property for passing data to the report generator."""
    blurb = inspect.getdoc(item.function)
    if blurb is None:
        raise AssertionError(f"Test {item.name} has no docstring")
    reqs: list[str] = []
    for marker in item.iter_markers("requirements"):
        if isinstance(marker.args[0], (tuple, list)):
            reqs.extend(marker.args[0])
        else:
            reqs.extend(name.strip() for name in marker.args[0].split(",") if name.strip())
    data = [{"$msg_type": "test_info", "blurb": blurb, "test_start": time.time(), "requirements": reqs}]
    name_marker = item.get_closest_marker("name")
    if name_marker is not None:
        data[0]["test_name"] = name_marker.args[0]
    item.user_properties.append(("pdf_report_data", data))
    item.stash[pdf_report_data_key] = data
    yield


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item) -> Generator[None, None, None]:
    """Update the test_start field when the test is actually started.

    This gives a more accurate start time than the one recorded by
    :func:`pytest_runtest_setup`, which is the time at which setup
    started.
    """
    item.stash[pdf_report_data_key][0]["test_start"] = time.time()
    yield


@pytest.fixture(autouse=True)
def matplotlib_report_style() -> Generator[None, None, None]:
    """Set the style of all matplotlib plots."""
    with matplotlib.style.context("ggplot"), matplotlib.rc_context(
        {
            # Serif fonts better match the rest of the document
            "font.family": "serif",
            "font.serif": ["Liberation Serif"],
            # A lot of the graphs are noisy and a narrower linewidth makes
            # the detail easier to see.
            "lines.linewidth": 0.3,
        }
    ):
        yield


@pytest.fixture(autouse=True)
def quiet_spead2(caplog: pytest.LogCaptureFixture) -> None:
    """Ensure spead2 only logs warnings and above.

    Every time we shut down a stream we get INFO logs about heaps that were
    dropped, which drowns out the more useful INFO logs from the test itself.
    """
    if logging.getLogger("spead2").getEffectiveLevel() < logging.WARN:
        caplog.set_level(logging.WARN, logger="spead2")


@pytest.fixture
async def _cbf_config_and_description(
    pytestconfig: pytest.Config,
    n_antennas: int,
    n_channels: int,
    n_dsims: int,
    band: str,
    int_time: float,
    narrowband_decimation: int,
) -> tuple[dict, dict]:
    # shutdown_delay is set to zero to speed up the test. We don't care
    # that Prometheus might not get to scrape the final metric updates.
    config: dict = {
        "version": "4.1",
        "config": {"shutdown_delay": 0.0},
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
        for pol in ["h", "v"]:
            name = f"{dsim_name}{pol}"
            dig_names.append(name)
            config["outputs"][name] = {
                "type": "sim.dig.baseband_voltage",
                "band": band,
                "adc_sample_rate": adc_sample_rate,
                "centre_frequency": centre_frequency,
                "antenna": f"{dsim_name}, 0:0:0, 0:0:0, 0, 0",
            }
    config["outputs"]["antenna-channelised-voltage"] = {
        "type": "gpucbf.antenna_channelised_voltage",
        "src_streams": [dig_names[i % len(dig_names)] for i in range(2 * n_antennas)],
        # m8xx is used to avoid possible confusion with real antennas
        "input_labels": [f"m{800 + i}{pol}" for i in range(n_antennas) for pol in ["v", "h"]],
        "n_chans": n_channels,
    }
    if narrowband_decimation > 1:
        # Create a wideband output so that testing is representative of normal
        # usage, although it will not be consumed.
        config["outputs"]["wideband-antenna-channelised-voltage"] = copy.deepcopy(
            config["outputs"]["antenna-channelised-voltage"]
        )
        config["outputs"]["wideband-antenna-channelised-voltage"]["n_chans"] = 8192
        config["outputs"]["wideband-baseline-correlation-products"] = {
            "type": "gpucbf.baseline_correlation_products",
            "src_streams": ["wideband-antenna-channelised-voltage"],
            "int_time": int_time,
        }
        # Pick a centre frequency that is not going to be a multiple of the
        # channel width (to test the most general case), but which is a
        # multiple of the dsim frequency resolution (to avoid rounding the
        # frequency of injected tones) and of the narrowband heap rate (so that
        # the phase of the mixer is the same at the start of each heap - needed
        # for baseline_correlation_products/test_consistency.py). The dsim
        # frequency resolution is 1/2**27 samples, while the largest
        # narrowband heap rate is 1/2**24 (jones-per-batch=2**20 and
        # narrowband factor 8).
        centre_frequency = adc_sample_rate * (3456789 / 2**24)
        config["outputs"]["antenna-channelised-voltage"]["narrowband"] = {
            "decimation_factor": narrowband_decimation,
            "centre_frequency": centre_frequency,
        }
    config["outputs"]["baseline-correlation-products"] = {
        "type": "gpucbf.baseline_correlation_products",
        "src_streams": ["antenna-channelised-voltage"],
        "int_time": int_time,
    }

    n_beams = int(pytestconfig.getini("beams"))
    for beam in range(n_beams):
        for pol_idx, pol in enumerate("xy"):
            config["outputs"][f"tied-array-channelised-voltage-{beam}{pol}"] = {
                "type": "gpucbf.tied_array_channelised_voltage",
                "src_streams": ["antenna-channelised-voltage"],
                "src_pol": pol_idx,
            }

    # The first three key/values are used for the traditional MeerKAT
    # CBF mode string, while the rest are used for a more complete
    # CBF description in the final report.
    # TODO: Update the key to be the actual parameter/fixture name
    cbf_mode_config: dict[str, str] = {
        "antennas": str(n_antennas),
        "channels": str(n_channels),
        "bandwidth": f"{round(adc_sample_rate / 1e6 / 2 / narrowband_decimation)}",
        "band": f"{BANDS[band].long_name}",
        "integration_time": str(int_time),
        "narrowband_decimation": str(narrowband_decimation),
        "dsims": str(n_dsims),
        "beams": str(n_beams),
    }
    long_description = (
        f"{n_antennas} antennas, {n_channels} channels, {n_beams} beams, "
        f"{BANDS[band].long_name}-band, {int_time}s integrations, {n_dsims} dsims"
    )
    if narrowband_decimation > 1:
        long_description += f", 1/{narrowband_decimation} narrowband"
    logger.info("Config for CBF generation: %s.", long_description)

    return config, cbf_mode_config


@pytest.fixture
async def cbf_config(_cbf_config_and_description: tuple[dict, dict]) -> dict:
    """Produce the configuration dict from the given parameters."""
    return _cbf_config_and_description[0]


@pytest.fixture
async def cbf_mode_config(_cbf_config_and_description: tuple[dict, dict]) -> dict:
    """Produce a dictionary describing the CBF config."""
    return _cbf_config_and_description[1]


@pytest.fixture(scope="session")
async def cbf_cache(pytestconfig: pytest.Config) -> AsyncGenerator[CBFCache, None]:
    """Obtain the session-scoped :class:`CBFCache`."""
    cache = CBFCache(pytestconfig)
    yield cache
    await cache.close()


@pytest.fixture
async def capture_start_streams(request: pytest.FixtureRequest, cbf_config: dict) -> list[str]:
    """List of streams for which capture-start will automatically be issued."""
    no_capture_start: set[str] = set()
    for marker in request.node.iter_markers("no_capture_start"):
        if marker.args == ():
            return []  # Requests no streams be automatically started
        no_capture_start.update(marker.args)

    out = []
    for name, conf in cbf_config["outputs"].items():
        if name not in no_capture_start and conf["type"] in _CAPTURE_TYPES:
            out.append(name)
    return out


@pytest.fixture
async def cbf(
    cbf_cache: CBFCache,
    cbf_config: dict,
    cbf_mode_config: dict,
    capture_start_streams: list[str],
    pdf_report: Reporter,
) -> AsyncGenerator[CBFRemoteControl, None]:
    """Set up a CBF for a single test.

    The returned CBF might not be specific to this test, but it will have
    been reset to a default state, with the dsim outputting zeros.
    """
    cbf = await cbf_cache.get_cbf(cbf_config, cbf_mode_config)
    pdf_report.config(cbf=str(cbf.uuid))
    if isinstance(cbf, FailedCBF):
        raise cbf.exc
    assert isinstance(cbf, CBFRemoteControl)
    # Reset the CBF to default state
    pcc = cbf.product_controller_client
    async with asyncio.TaskGroup() as tg:
        for client in cbf.dsim_clients:
            tg.create_task(client.request("signals", "0;0;"))
    for name, conf in cbf.config["outputs"].items():
        if conf["type"] == "gpucbf.antenna_channelised_voltage":
            n_inputs = len(conf["src_streams"])
            sync_time = cbf.init_sensors[f"{name}.sync-time"].value
            await pcc.request("gain-all", name, "default")
            await pcc.request("delays", name, sync_time, *(["0,0:0,0"] * n_inputs))
        elif conf["type"] == "gpucbf.tied_array_channelised_voltage":
            source_indices = ast.literal_eval(cbf.init_sensors[f"{name}.source-indices"].value.decode())
            n_inputs = len(source_indices)
            await pcc.request("beam-quant-gains", name, 1.0)
            await pcc.request("beam-delays", name, *(("0:0",) * n_inputs))
            await pcc.request("beam-weights", name, *((1.0,) * n_inputs))

    for name in capture_start_streams:
        await pcc.request("capture-start", name)

    yield cbf

    for name, conf in cbf.config["outputs"].items():
        if conf["type"] in _CAPTURE_TYPES:
            await pcc.request("capture-stop", name)


class CoreAllocator:
    """Provide CPU cores to receivers that need them.

    It is initialised with a list of cores (from pytest config), with earlier
    entries considered better than later ones. Cores are allocated in this
    order. There is no mechanism to return cores; simply create a new
    allocator to start fresh.
    """

    def __init__(self, cores: Iterable[int]) -> None:
        self._cores = deque(cores)

    def allocate(self, n: int) -> Sequence[int]:
        """Request `n` cores.

        Raises
        ------
        ValueError
            If there are insufficient cores available
        """
        if n > len(self._cores):
            raise ValueError(f"{n} cores requested but only {len(self._cores)} cores available")
        return [self._cores.popleft() for _ in range(n)]


# Note: it's important that this has session scope, so that it's only run
# before core_allocator calls os.sched_setaffinity.
@pytest.fixture(scope="session")
def cores(pytestconfig: pytest.Config) -> list[int]:
    """Get the cores to use for core pinning for this test."""
    cores = [int(x) for x in pytestconfig.getini("cores")]
    if not cores:
        cores = sorted(os.sched_getaffinity(0))
    return cores


@pytest.fixture
def core_allocator(cores: list[int]) -> CoreAllocator:
    """Create a core allocator for the test."""
    alloc = CoreAllocator(cores)
    # Pin the main Python thread to a core, to ensure it won't conflict with
    # any of the worker threads. Note that this is repeated for each test,
    # but that is harmless.
    os.sched_setaffinity(0, alloc.allocate(1))
    return alloc


@pytest.fixture
def receive_baseline_correlation_products_manual_start(
    pytestconfig: pytest.Config, cbf: CBFRemoteControl, core_allocator: CoreAllocator
) -> Generator[BaselineCorrelationProductsReceiver, None, None]:
    """Create a spead2 receive stream for ingesting X-engine output.

    This fixture does not start the receiver.
    """
    interface_address = get_interface_address(pytestconfig.getini("interface"))
    # This will require running pytest with spead2_net_raw which is unusual.
    use_ibv = pytestconfig.getini("use_ibv")

    receiver = BaselineCorrelationProductsReceiver(
        cbf=cbf,
        stream_name="baseline-correlation-products",
        cores=core_allocator.allocate(4),
        interface_address=interface_address,
        use_ibv=use_ibv,
    )
    yield receiver
    receiver.stream_group.stop()


@pytest.fixture
async def receive_baseline_correlation_products(
    receive_baseline_correlation_products_manual_start: BaselineCorrelationProductsReceiver,
    capture_start_streams: list[str],
) -> BaselineCorrelationProductsReceiver:
    """Create a spead2 receive stream for ingesting X-engine output."""
    receiver = receive_baseline_correlation_products_manual_start
    receiver.start()
    # Ensure that the data is flowing, and that we throw away any data that
    # predates the start of this test (to prevent any state leaks from previous
    # tests). The timeout is increased since it may take some time to get the
    # data flowing at the start.
    if "baseline-correlation_products" in capture_start_streams:
        await receiver.wait_complete_chunk(max_delay=0, timeout=3 * DEFAULT_TIMEOUT)
    return receiver


@pytest.fixture
def receive_tied_array_channelised_voltage_manual_start(
    pytestconfig: pytest.Config,
    cbf: CBFRemoteControl,
    cbf_config: dict,
    n_antennas: int,
    n_channels: int,
    int_time: float,
    band: str,
    core_allocator: CoreAllocator,
) -> Generator[TiedArrayChannelisedVoltageReceiver, None, None]:
    """Create a spead2 receive stream for ingest the tied-array-channelised-voltage streams.

    This fixture does not start the receiver.
    """
    interface_address = get_interface_address(pytestconfig.getini("interface"))
    use_ibv = pytestconfig.getini("use_ibv")
    interface_gbps = float(pytestconfig.getini("interface_gbps"))

    stream_names = [
        name
        for name, config in cbf_config["outputs"].items()
        if config["type"] == "gpucbf.tied_array_channelised_voltage"
    ]

    # Subscribe to only as many beams as can reliably be squeezed through a
    # 100 Gb/s adapter.
    n_bls = n_antennas * (n_antennas + 1) * 2
    budget = interface_gbps * 1e9 - n_bls * n_channels / int_time * 64  # 64 bits per visibility
    adc_sample_rate = BANDS[band].adc_sample_rate
    stream_bandwidth = adc_sample_rate * 8  # 8 bits per component
    max_streams = min(len(stream_names), int(budget // stream_bandwidth))
    if max_streams < 4:
        pytest.skip("Not enough network bandwidth for two dual-pol beams")
    else:
        logger.info("Subscribing to %d beam streams", max_streams)

    stream_names = stream_names[:max_streams]
    cores = core_allocator.allocate(len(stream_names))
    receiver = TiedArrayChannelisedVoltageReceiver(
        cbf=cbf, stream_names=stream_names, cores=cores, interface_address=interface_address, use_ibv=use_ibv
    )
    yield receiver
    receiver.stream_group.stop()


@pytest.fixture
async def receive_tied_array_channelised_voltage(
    receive_tied_array_channelised_voltage_manual_start: TiedArrayChannelisedVoltageReceiver,
    cbf_config: dict,
    capture_start_streams: list[str],
) -> TiedArrayChannelisedVoltageReceiver:
    """Create a spead2 receive stream for ingest the tied-array-channelised-voltage streams."""
    receiver = receive_tied_array_channelised_voltage_manual_start
    receiver.start()
    # Ensure that the data is flowing, and that we throw away any data that
    # predates the start of this test (to prevent any state leaks from previous
    # tests). The timeout is increased since it may take some time to get the
    # data flowing at the start.
    if all(
        name in capture_start_streams
        for name, config in cbf_config["outputs"].items()
        if config["type"] == "gpucbf.tied_array_channelised_voltage"
    ):
        await receiver.wait_complete_chunk(max_delay=0, timeout=3 * DEFAULT_TIMEOUT)
    return receiver
