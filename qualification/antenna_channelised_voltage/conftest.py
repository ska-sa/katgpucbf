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

"""Fixtures and options for testing of antenna-channelised-voltage streams."""

import pytest


@pytest.fixture(scope="package")
def n_dsims(n_antennas: int):
    """Give every simulated antenna its own dsim."""
    return n_antennas


@pytest.fixture(scope="session", params=[1])
def n_antennas(request):  # noqa: D401
    """Number of antennas, i.e. size of the array."""
    return request.param
