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
from typing import TYPE_CHECKING

from katsdptelstate.endpoint import Endpoint


@dataclass
class Output(ABC):
    """Static configuration for an output stream."""

    name: str
    channels: int
    jones_per_heap: int
    taps: int
    w_cutoff: float
    dst: list[Endpoint]

    def __post_init__(self) -> None:
        if self.channels % len(self.dst) != 0:
            raise ValueError("channels must be a multiple of the number of destinations")
        channels_per_substream = self.channels // len(self.dst)
        if self.jones_per_heap % channels_per_substream != 0:
            raise ValueError("jones_per_heap must be a multiple of the number of channels per substream")

    @property
    def spectra_per_heap(self) -> int:
        """Number of spectra in each output heap."""
        return self.jones_per_heap * len(self.dst) // self.channels

    @property
    @abstractmethod
    def internal_channels(self) -> int:
        """Number of channels in the PFB."""
        raise NotImplementedError  # pragma: nocover

    @property
    @abstractmethod
    def spectra_samples(self) -> int:
        """Number of incoming digitiser samples needed per spectrum.

        Note that this is the spacing between spectra. Each spectrum uses
        an overlapping window with more samples than this.
        """
        raise NotImplementedError  # pragma: nocover

    @property
    @abstractmethod
    def subsampling(self) -> int:
        """Number of digitiser samples between PFB input samples."""
        raise NotImplementedError  # pragma: nocover

    @property
    @abstractmethod
    def window(self) -> int:
        """Number of digitiser samples that contribute to each output spectrum."""
        raise NotImplementedError  # pragma: nocover

    if TYPE_CHECKING:
        # Actually defining it at runtime confuses the dataclass decorator,
        # because on NarrowbandOutput it is a data member rather than a
        # property.
        @property
        @abstractmethod
        def decimation(self) -> int:
            """Factor by which bandwidth is reduced at the output."""
            raise NotImplementedError  # pragma: nocover

    @property
    @abstractmethod
    def internal_decimation(self) -> int:
        """Factor by which bandwidth is reduced by the DDC kernel."""
        raise NotImplementedError  # pragma: nocover


@dataclass
class WidebandOutput(Output):
    """Static configuration for a wideband output stream."""

    @property
    def internal_channels(self) -> int:  # noqa: D102
        return self.channels

    @property
    def spectra_samples(self) -> int:  # noqa: D102
        return 2 * self.channels

    @property
    def decimation(self) -> int:  # noqa: D102
        return 1

    @property
    def internal_decimation(self) -> int:  # noqa: D102
        return 1

    @property
    def subsampling(self) -> int:  # noqa: D102
        return 1

    @property
    def window(self) -> int:  # noqa: D102
        return self.taps * self.spectra_samples


@dataclass
class NarrowbandOutput(Output):
    """Static configuration for a narrowband stream."""

    centre_frequency: float
    decimation: int
    ddc_taps: int
    weight_pass: float

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.decimation % 2 != 0:
            raise ValueError("decimation factor must be even")

    @property
    def internal_channels(self) -> int:  # noqa: D102
        return 2 * self.channels

    @property
    def spectra_samples(self) -> int:  # noqa: D102
        return 2 * self.channels * self.decimation

    @property
    def internal_decimation(self) -> int:  # noqa: D102
        return self.decimation // 2

    @property
    def subsampling(self) -> int:  # noqa: D102
        return self.decimation

    @property
    def window(self) -> int:  # noqa: D102
        return self.taps * self.spectra_samples + self.ddc_taps - self.subsampling
