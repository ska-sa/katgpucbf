# noqa: D104

################################################################################
# Copyright (c) 2020-2021, National Research Foundation (SARAO)
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

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as version_func
from typing import Final

try:
    __version__ = version_func(__name__)
except PackageNotFoundError:
    # Package wasn't installed yet?
    __version__ = "unknown"

__all__ = ["__version__"]

BYTE_BITS: Final = 8
COMPLEX: Final = 2
N_POLS: Final = 2
SPEAD_DESCRIPTOR_INTERVAL_S: Final = 5
PREAMBLE_SIZE = 72
SEND_RATE_FACTOR = 1.05
DEFAULT_PACKET_PAYLOAD_BYTES: Final = 8192
DEFAULT_TTL: Final = 4  #: Default TTL for spead multicast transmission
DEFAULT_KATCP_HOST = ""  # All interfaces
DEFAULT_KATCP_PORT = 7147
