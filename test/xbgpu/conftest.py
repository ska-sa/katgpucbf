################################################################################
# Copyright (c) 2020-2021, National Research Foundation (SARAO)
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

"""Fixtures for use in xbgpu unit tests."""


import logging
from typing import AsyncGenerator, Final, List, Tuple

import pytest
from katsdpsigproc.abc import AbstractContext

from katgpucbf.monitor import NullMonitor
from katgpucbf.xbgpu.engine import XBEngine

from .test_parameters import array_size, num_channels, num_spectra_per_heap

logging.basicConfig()

ADC_SAMPLE_RATE: Final[float] = 1712e6  # L-band
HEAPS_PER_FENGINE_PER_CHUNK: Final[int] = 2
SEND_RATE_FACTOR: Final[float] = 1.1
SAMPLE_BITWIDTH: Final[int] = 8

ENGINE_CONFIG_LIST: Final[List[Tuple[int, int, int]]] = [
    (arr_size, n_spectra_per_heap, n_chans)
    for arr_size, n_spectra_per_heap, n_chans in zip(array_size, num_spectra_per_heap, num_channels)
]


@pytest.fixture(params=ENGINE_CONFIG_LIST)
def counter_fixture1(request):
    """Test the parametrisation of the fixture using `param`."""
    return request.param


@pytest.fixture(params=ENGINE_CONFIG_LIST)
def counter_fixture2(request):
    """Test the parametrisation of the fixture using `param`."""
    return request.param


@pytest.fixture
def square_fixture(request):
    """Return the square of the requested number."""
    return request.param ** 2


@pytest.fixture
def str_repr_fixture(request):
    """Return an f-string version of the requested parameter."""
    return f"Array size: {request.param[0]} | Spectra-per-heap: {request.param[1]} | Channels: {request.param[2]}"


@pytest.fixture
async def engine_server(request, context: AbstractContext) -> AsyncGenerator[XBEngine, None]:
    """Create a dummy :class:`.xbgpu.Engine` for unit testing.

    The arguments passed are based on the default arguments from
    :mod:`~katgpucbf.xbgpu.main`. Certain values will change according to the
    unit test's requirements, e.g. --array-size.
    """
    logging.info(f"Started XBEngine with {request.param}")
    # Parse n_ants, n_spectra_per_heap and n_channels from request.param
    n_ants = request.param[0]
    n_spectra_per_heap = request.param[1]
    n_channels_total = request.param[2]

    # Get a realistic number of engines: round n_ants*4 up to the next power of 2.
    n_engines = 1
    while n_engines < n_ants * 4:
        n_engines *= 2
    n_channels_per_stream = n_channels_total // n_engines
    rx_reorder_tol = 2 ** 26  # Increase if needed; this is small to keep memory usage manageable
    heap_accumulation_threshold = 4

    # 4. Create xbengine
    # 4.1. Create Monitor required by XBEngine
    monitor = NullMonitor()

    server = XBEngine(
        katcp_host="",
        katcp_port=0,
        adc_sample_rate_hz=ADC_SAMPLE_RATE,
        send_rate_factor=SEND_RATE_FACTOR,
        n_ants=n_ants,
        n_channels_total=n_channels_total,
        n_channels_per_stream=n_channels_per_stream,
        n_spectra_per_heap=n_spectra_per_heap,
        sample_bits=SAMPLE_BITWIDTH,
        heap_accumulation_threshold=heap_accumulation_threshold,
        channel_offset_value=n_channels_per_stream * 4,  # Arbitrary value for now
        src_affinity=0,
        chunk_spectra=HEAPS_PER_FENGINE_PER_CHUNK,
        rx_reorder_tol=rx_reorder_tol,
        monitor=monitor,
        context=context,
    )

    await server.start()
    yield server
    await server.stop()
