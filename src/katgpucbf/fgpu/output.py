################################################################################
# Copyright (c) 2023-2025, National Research Foundation (SARAO)
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
from enum import Enum
from typing import TYPE_CHECKING

from katsdptelstate.endpoint import Endpoint

from ..utils import DitherType


class WindowFunction(Enum):
    """Window function to use in :func:`.generate_pfb_weights`."""

    HANN = 1
    RECT = 2
    DEFAULT = 1  # Alias used to determine default when none is specified


@dataclass
class Output(ABC):
    """Static configuration for an output stream."""

    name: str
    channels: int
    jones_per_batch: int
    taps: int
    w_cutoff: float
    window_function: WindowFunction
    dst: list[Endpoint]
    dither: DitherType

    def __post_init__(self) -> None:
        if self.channels % len(self.dst) != 0:
            raise ValueError("channels must be a multiple of the number of destinations")
        if self.jones_per_batch % self.channels != 0:
            raise ValueError("jones_per_batch must be a multiple of channels")

    @property
    def spectra_per_heap(self) -> int:
        """Number of spectra in each output heap."""
        return self.jones_per_batch // self.channels

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

    @property
    def spectra_samples(self) -> int:  # noqa: D102
        return self.internal_channels * self.subsampling

    @property
    def window(self) -> int:  # noqa: D102
        return self.taps * self.spectra_samples + self.ddc_taps - self.subsampling


@dataclass
class NarrowbandOutputDiscard(NarrowbandOutput):
    """Static configuration for a narrowband stream that discards channels."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.decimation % 2 != 0:
            raise ValueError("decimation factor must be even")

    @property
    def internal_channels(self) -> int:  # noqa: D102
        return 2 * self.channels

    @property
    def internal_decimation(self) -> int:  # noqa: D102
        return self.decimation // 2

    @property
    def subsampling(self) -> int:  # noqa: D102
        return self.decimation


@dataclass
class NarrowbandOutputNoDiscard(NarrowbandOutput):
    """Static configuration for a narrowband stream that does not discard channels."""

    usable_bandwidth: float

    @property
    def internal_channels(self) -> int:  # noqa: D102
        return self.channels

    @property
    def internal_decimation(self) -> int:  # noqa: D102
        return self.decimation

    @property
    def subsampling(self) -> int:  # noqa: D102
        return 2 * self.decimation
