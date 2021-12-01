################################################################################
# Copyright (c) 2021, National Research Foundation (SARAO)
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

"""katcp server."""

import asyncio
import logging

import aiokatcp
import pyparsing as pp

from .. import __version__
from .send import HeapSet, Sender
from .signal import parse_signals, sample_async

logger = logging.getLogger(__name__)


class DeviceServer(aiokatcp.DeviceServer):
    """katcp server.

    Parameters
    ----------
    sender
        Sender which is streaming data out. It is halted when the server is stopped.
    spare
        Heap set which is not currently being used, but is available to swap in
    adc_sample_rate
        Sampling rate in Hz
    sample_bits
        Number of bits per output sample
    first_timestamp
        The timestamp associated with the first output sample
    *args, **kwargs
        Passed to base class
    """

    # TODO: VERSION means interface version, rather than software version. It
    # will need to wait on a proper ICD for a release.
    VERSION = "katgpucbf-dsim-0.1"
    BUILD_STATE = __version__

    def __init__(
        self,
        sender: Sender,
        spare: HeapSet,
        adc_sample_rate: float,
        sample_bits: int,
        first_timestamp: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.sender = sender
        self.spare = spare
        self.adc_sample_rate = adc_sample_rate
        self.sample_bits = sample_bits
        self.first_timestamp = first_timestamp
        self._signals_lock = asyncio.Lock()  # Serialises request_signals

    async def on_stop(self) -> None:  # noqa: D102
        self.sender.halt()

    async def request_signals(self, ctx, signals_str: str) -> int:
        """Update the signals that are generated.

        Parameters
        ----------
        signals_str
            Textural description of the signals. See the docstring for
            parse_signals for the language description. The description
            must produce one signal per polarisation.

        Returns
        -------
        timestamp
            First timestamp which will use the new signals
        """
        try:
            signals = parse_signals(signals_str)
        except pp.ParseBaseException as exc:
            raise aiokatcp.FailReply(str(exc))
        n_pol = self.spare.data.dims["pol"]
        if len(signals) != n_pol:
            raise aiokatcp.FailReply(f"expected {n_pol} signals, received {len(signals)}")

        async with self._signals_lock:
            await sample_async(
                signals, self.first_timestamp, self.adc_sample_rate, self.sample_bits, self.spare.data["payload"]
            )
            spare = self.sender.heap_set
            timestamp = await self.sender.set_heaps(self.spare)
            self.spare = spare
            return timestamp
