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

"""Common SPEAD-related constants and helper function."""

from typing import Any

import spead2

DIGITISER_ID_ID = 0x3101
DIGITISER_STATUS_ID = 0x3102
FENG_ID_ID = 0x4101
FENG_RAW_ID = 0x4300
FREQUENCY_ID = 0x4103
RAW_DATA_ID = 0x3300  # Digitiser data
XENG_RAW_ID = 0x1800
TIMESTAMP_ID = 0x1600

#: SPEAD flavour used for all send streams
FLAVOUR = spead2.Flavour(4, 64, 48, 0)

# Format for immediate items
IMMEDIATE_FORMAT = [("u", FLAVOUR.heap_address_bits)]


def make_immediate(id: int, value: Any) -> spead2.Item:
    """Synthesize an immediate item.

    Parameters
    ----------
    id
        The SPEAD identifier for the item
    value
        The value of the item
    """
    return spead2.Item(id, "dummy_item", "", (), format=[("u", FLAVOUR.heap_address_bits)], value=value)
