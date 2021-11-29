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

from .. import __version__
from .send import Sender

logger = logging.getLogger(__name__)


def _done_callback(future: asyncio.Future) -> None:
    try:
        future.result()  # Evaluate just for exceptions
    except Exception:
        logger.exception("Sending failed with exception")


class DeviceServer(aiokatcp.DeviceServer):
    """katcp server.

    Parameters
    ----------
    sender
        Sender which is streaming data out. It is halted when the server is stopped.
    *args, **kwargs
        Passed to base class
    """

    # TODO: VERSION means interface version, rather than software version. It
    # will need to wait on a proper ICD for a release.
    VERSION = "katgpucbf-dsim-0.1"
    BUILD_STATE = __version__

    def __init__(self, sender: Sender, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sender = sender

    async def on_stop(self) -> None:  # noqa: D102
        self.sender.halt()
