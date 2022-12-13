# noqa: D104

################################################################################
# Copyright (c) 2020-2022, National Research Foundation (SARAO)
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

BYTE_BITS: Final = 8
COMPLEX: Final = 2
N_POLS: Final = 2
SPEAD_DESCRIPTOR_INTERVAL_S: Final = 5
SEND_RATE_FACTOR = 1.1
#: Biggest power of 2 that fits in a jumbo MTU. A power of 2 isn't required but
#: it can be convenient to have packet boundaries align with the natural
#: boundaries in the payload (for antenna-channelised-voltage output). Bigger
#: is better to minimise the number of packets/second to process.
DEFAULT_PACKET_PAYLOAD_BYTES: Final = 8192
DEFAULT_TTL: Final = 4  #: Default TTL for spead multicast transmission
DEFAULT_KATCP_HOST: Final = ""  # All interfaces
DEFAULT_KATCP_PORT: Final = 7147
DIG_HEAP_SAMPLES: Final = 4096
DIG_SAMPLE_BITS: Final = 10
#: Minimum update period (in seconds) for katcp sensors where the underlying
#: value may update extremely rapidly.
MIN_SENSOR_UPDATE_PERIOD: Final = 1.0

GPU_PROC_TASK_NAME: Final[str] = "GPU Processing Loop"
RECV_TASK_NAME: Final[str] = "Receiver Loop"
SEND_TASK_NAME: Final[str] = "Sender Loop"
DESCRIPTOR_TASK_NAME: Final[str] = "Descriptor Loop"
TIME_SYNC_TASK_NAME: Final[str] = "Time Sync Check Loop"
