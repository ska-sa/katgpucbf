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

from typing import Final

SAMPLE_BITS: Final = 10
METRIC_NAMESPACE: Final = "fgpu"

# Range in which the dig-pwr-dbfs sensor is NOMINAL
# TODO these thresholds are inherited from MeerKAT. Are they what we want?
DIG_POWER_DBFS_LOW: Final = -32.0
DIG_POWER_DBFS_HIGH: Final = -22.0
