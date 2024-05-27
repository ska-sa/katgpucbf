################################################################################
# Copyright (c) 2023-2024, National Research Foundation (SARAO)
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

"""Common functionality between :class:`.XSend` and :class:`.BSend`."""

from typing import Final

import spead2.send.asyncio


class Send:
    """Common functionality between :class:`.XSend` and :class:`.BSend`."""

    def __init__(
        self,
        *,
        n_channels: int,
        n_channels_per_substream: int,
        channel_offset: int,
        stream: "spead2.send.asyncio.AsyncStream",
        descriptor_heap: spead2.send.Heap,
    ) -> None:
        if n_channels % n_channels_per_substream != 0:
            raise ValueError("n_channels must be an integer multiple of n_channels_per_substream")
        if channel_offset % n_channels_per_substream != 0:
            raise ValueError("channel_offset must be an integer multiple of n_channels_per_substream")
        self.n_channels: Final[int] = n_channels
        self.n_channels_per_substream: Final[int] = n_channels_per_substream
        self.channel_offset: Final[int] = channel_offset
        self.stream = stream
        self.descriptor_heap = descriptor_heap

        # Set heap count sequence to allow a receiver to ingest multiple
        # X/B-engine outputs, if they should so choose.
        self.stream.set_cnt_sequence(
            channel_offset // n_channels_per_substream,
            n_channels // n_channels_per_substream,
        )
