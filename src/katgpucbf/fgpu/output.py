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

from abc import ABC, abstractmethod
from dataclasses import dataclass

from katsdptelstate.endpoint import Endpoint


@dataclass
class Output(ABC):
    """Static configuration for an output stream."""

    name: str
    channels: int
    taps: int
    w_cutoff: float
    dst: list[Endpoint]

    @abstractmethod
    def send_rate_factor(self) -> float:
        """Output stream rate, relative to a wideband stream."""
        raise NotImplementedError  # pragma: nocover

    @property
    @abstractmethod
    def spectra_samples(self) -> int:
        """Number of incoming digitiser samples needed per spectrum.

        Note that this is the spacing between spectra. Each spectrum uses
        an overlapping window with more samples than this.
        """
        raise NotImplementedError  # pragma: nocover


@dataclass
class WidebandOutput(Output):
    """Static configuration for a wideband output stream."""

    def send_rate_factor(self) -> float:  # noqa: D102
        return 1.0

    @property
    def spectra_samples(self) -> int:  # noqa: D102
        return 2 * self.channels


@dataclass
class NarrowbandOutput(Output):
    """Static configuration for a narrowband stream."""

    decimation: int

    def send_rate_factor(self) -> float:  # noqa: D102
        return 1.0 / self.decimation

    @property
    def spectra_samples(self) -> int:  # noqa: D102
        return 2 * self.channels * self.decimation
