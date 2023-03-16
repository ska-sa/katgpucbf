################################################################################
# Copyright (c) 2023, National Research Foundation (SARAO)
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

"""Data structures capturing static configuration of a single output stream."""

from dataclasses import dataclass

from katsdptelstate.endpoint import Endpoint


@dataclass
class Output:
    """Static configuration for an output stream."""

    channels: int
    taps: int
    w_cutoff: float
    dst: list[Endpoint]


@dataclass
class WidebandOutput(Output):
    """Static configuration for a wideband output stream."""

    pass


@dataclass
class NarrowbandOutput(Output):
    """Static configuration for a narrowband stream."""

    decimation: int
