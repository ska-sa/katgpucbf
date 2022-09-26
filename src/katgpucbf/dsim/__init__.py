# noqa: D104

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

from typing import Final

METRIC_NAMESPACE: Final = "dsim"
#: Bit position in digitiser_status SPEAD item for ADC saturation flag
STATUS_SATURATION_FLAG_BIT: Final = 1
#: First bit position in digitiser status SPEAD item for ADC saturation count
STATUS_SATURATION_COUNT_SHIFT: Final = 32
