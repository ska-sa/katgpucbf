################################################################################
# Copyright (c) 2021-2022, National Research Foundation (SARAO)
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
import time
from typing import Optional, Sequence

import aiokatcp
import numpy as np
import pyparsing as pp
import xarray as xr

from .. import BYTE_BITS, __version__
from ..send import DescriptorSender
from ..spead import DIGITISER_STATUS_SATURATION_COUNT_SHIFT, DIGITISER_STATUS_SATURATION_FLAG_BIT
from .send import HeapSet, Sender
from .shared_array import SharedArray
from .signal import Signal, SignalService, TerminalError, format_signals, parse_signals

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
    dither_seed
        Dither seed (used only to populate a sensor).
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
        descriptor_sender: DescriptorSender,
        spare: HeapSet,
        adc_sample_rate: float,
        sample_bits: int,
        dither_seed: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.sender = sender
        self.descriptor_sender = descriptor_sender
        self.spare = spare
        self.adc_sample_rate = adc_sample_rate
        self.sample_bits = sample_bits
        # Scratch space for computing saturation counts. It is passed to
        # (and filled in by) the SignalService, so needs to use shared
        # memory.
        saturated_shape = (sender.heap_set.data.sizes["pol"], sender.heap_set.data.sizes["time"])
        shared_saturated = SharedArray.create("saturated", saturated_shape, np.uint64)
        self._saturated = xr.DataArray(
            shared_saturated.buffer, dims=["pol", "time"], attrs={"shared_array": shared_saturated}
        )
        self._signals_lock = asyncio.Lock()  # Serialises request_signals
        heap_sets = [sender.heap_set, spare]
        self._signal_service = SignalService(
            [heap_set.data["payload"] for heap_set in heap_sets] + [self._saturated],
            sample_bits,
            dither_seed,
        )

        self._signals_orig_sensor = aiokatcp.Sensor(
            str,
            "signals-orig",
            "User-provided string used to define the signals",
        )
        self._signals_sensor = aiokatcp.Sensor(
            str,
            "signals",
            "String reproducibly describing how the signals are generated",
        )
        self._period_sensor = aiokatcp.Sensor(
            int,
            "period",
            "Number of samples after which the signals will be repeated",
        )
        self._steady_state_sensor = aiokatcp.Sensor(
            int,
            "steady-state-timestamp",
            "Heaps with this timestamp or greater are guaranteed to reflect the effects of previous katcp requests.",
        )
        self.sensors.add(self._signals_orig_sensor)
        self.sensors.add(self._signals_sensor)
        self.sensors.add(self._period_sensor)
        self.sensors.add(self._steady_state_sensor)
        self.sensors.add(
            aiokatcp.Sensor(
                int,
                "dither-seed",
                "Random seed used in dithering for quantisation",
                initial_status=aiokatcp.Sensor.Status.NOMINAL,
                default=dither_seed,
            )
        )
        self.sensors.add(
            aiokatcp.Sensor(
                int,
                "max-period",
                "Maximum period that may be passed to ?signals",
                initial_status=aiokatcp.Sensor.Status.NOMINAL,
                default=spare.data["payload"].isel(pol=0).size * BYTE_BITS // sample_bits,
            )
        )
        self.sensors.add(
            aiokatcp.Sensor(
                float,
                "adc-sample-rate",
                "Rate at which samples are generated",
                units="Hz",
                initial_status=aiokatcp.Sensor.Status.NOMINAL,
                default=adc_sample_rate,
            )
        )
        self.sensors.add(
            aiokatcp.Sensor(
                int,
                "sample-bits",
                "Number of bits in each output sample",
                initial_status=aiokatcp.Sensor.Status.NOMINAL,
                default=sample_bits,
            )
        )

    async def on_stop(self) -> None:  # noqa: D102
        self.sender.halt()
        self.descriptor_sender.halt()
        await self._signal_service.stop()

    async def set_signals(self, signals: Sequence[Signal], signals_str: str, period: Optional[int] = None) -> int:
        """Change the signals :meth:`request_signals`.

        This is the implementation of :meth:`request_signals`. See that method for
        description of the parameters and return value (`signals` is the parsed
        version of `signals_str`).
        """
        if period is None:
            period = self.sensors["max-period"].value
        async with self._signals_lock:
            await self._signal_service.sample(
                signals,
                0,
                period,
                self.adc_sample_rate,
                self.spare.data["payload"],
                self._saturated,
                self.sender.heap_samples,
            )
            # As per M1000-0001-053: bits [47:32] hold saturation count, while
            # bit 1 holds a boolean flag.
            # np.left_shift is << but xarray doesn't seem to implement the
            # operator overload.
            digitiser_status = np.left_shift(self._saturated, DIGITISER_STATUS_SATURATION_COUNT_SHIFT)
            digitiser_status |= xr.where(
                digitiser_status, np.uint64(1 << DIGITISER_STATUS_SATURATION_FLAG_BIT), np.uint64(0)
            )
            self.spare.data["digitiser_status"][:] = digitiser_status
            spare = self.sender.heap_set
            timestamp = await self.sender.set_heaps(self.spare)
            self.spare = spare
            self._signals_orig_sensor.value = signals_str
            self._signals_sensor.value = format_signals(signals)
            self._period_sensor.value = period
            self._steady_state_sensor.value = max(self._steady_state_sensor.value, timestamp)
            return timestamp

    async def request_signals(self, ctx, signals_str: str, period: int = None) -> int:
        """Update the signals that are generated.

        Parameters
        ----------
        signals_str
            Textural description of the signals. See the docstring for
            parse_signals for the language description. The description
            must produce one signal per polarisation.

        period
            Period for the generated signal. It must divide into the value
            indicated by the ``max-period`` sensor. If not specified, the
            value of ``max-period`` is used.

        Returns
        -------
        timestamp
            First timestamp which will use the new signals
        """
        try:
            signals = parse_signals(signals_str)
        except (pp.ParseBaseException, TerminalError) as exc:
            raise aiokatcp.FailReply(str(exc))
        n_pol = self.spare.data.dims["pol"]
        if len(signals) != n_pol:
            raise aiokatcp.FailReply(f"expected {n_pol} signals, received {len(signals)}")
        return await self.set_signals(signals, signals_str, period)

    async def request_time(self, ctx) -> float:
        """Return the current UNIX timestamp."""
        return time.time()
