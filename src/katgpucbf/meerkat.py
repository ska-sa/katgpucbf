################################################################################
# Copyright (c) 2022, National Research Foundation (SARAO)
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

"""Constants applicable to the MeerKAT / MeerKAT Extension telescope."""

from dataclasses import dataclass


@dataclass
class Band:
    """Holds presets for a known band."""

    adc_sample_rate: float
    centre_frequency: float


BANDS = {
    "l": Band(adc_sample_rate=1712e6, centre_frequency=1284e6),
    "u": Band(adc_sample_rate=1088e6, centre_frequency=816e6),
}
