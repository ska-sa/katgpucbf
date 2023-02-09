################################################################################
# Copyright (c) 2022-2023, National Research Foundation (SARAO)
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

    long_name: str
    adc_sample_rate: float
    centre_frequency: float


BANDS = {
    "l": Band("L", adc_sample_rate=1712e6, centre_frequency=1284e6),
    "u": Band("UHF", adc_sample_rate=1088e6, centre_frequency=816e6),
    "s0": Band("S0", adc_sample_rate=1750e6, centre_frequency=2187.5e6),
    "s1": Band("S1", adc_sample_rate=1750e6, centre_frequency=2406.25e6),
    "s2": Band("S2", adc_sample_rate=1750e6, centre_frequency=2625.0e6),
    "s3": Band("S3", adc_sample_rate=1750e6, centre_frequency=2843.75e6),
    "s4": Band("S4", adc_sample_rate=1750e6, centre_frequency=3062.5e6),
}
