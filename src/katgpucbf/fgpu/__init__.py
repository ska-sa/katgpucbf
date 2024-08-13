# noqa: D104

################################################################################
# Copyright (c) 2020-2021, 2023-2024 National Research Foundation (SARAO)
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

METRIC_NAMESPACE: Final = "fgpu"

# Range in which the dig-rms-dbfs sensor is NOMINAL
# TODO these thresholds are inherited from MeerKAT. Are they what we want?
DIG_RMS_DBFS_LOW: Final = -32.0
DIG_RMS_DBFS_HIGH: Final = -12.0
DIG_RMS_DBFS_WINDOW: Final = 1.0  # Window length in seconds

#: Valid values for the ``--dig-sample-bits`` command-line option
DIG_SAMPLE_BITS_VALID = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16]

#: Padding to add to GPU allocations of input chunks
# 8 is more than necessary but ensures that we have a simple consistent amount
# of padding between wideband and narrowband.
INPUT_CHUNK_PADDING = 8
