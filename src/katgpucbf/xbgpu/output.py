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

from abc import ABC
from dataclasses import dataclass

from katsdptelstate.endpoint import Endpoint


@dataclass
class Output(ABC):
    """Static configuration for an output stream."""

    name: str
    dst: Endpoint


@dataclass
class XOutput(Output):
    """Static configuration for an output baseline-correlation-products stream."""

    heap_accumulation_threshold: int
    samples_between_spectra: int
    spectra_per_heap: int

    @property
    def timestamp_increment_per_accumulation(self) -> int:  # noqa: D102
        """Timestamp increment (in samples) per accumulation interval."""
        return self.heap_accumulation_threshold * self.samples_between_spectra * self.spectra_per_heap


@dataclass
class BOutput(Output):
    """Static configuration for an output beam stream."""

    pass
