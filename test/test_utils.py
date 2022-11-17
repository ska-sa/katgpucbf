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

"""Tests for :mod:`katcbfgpu.utils`."""

import pytest

from katgpucbf.utils import TimeConverter


class TestTimeConverter:
    """Tests for :class:`katgpucbf.utils.TimeConverter`."""

    @pytest.fixture
    def time_converter(self) -> TimeConverter:  # noqa: D401
        """A time converter.

        It has power-of-two ADC sample count so that tests do not need to worry
        about rounding effects.
        """
        return TimeConverter(1234567890.0, 1048576.0)

    def test_unix_to_adc(self, time_converter: TimeConverter) -> None:
        """Test :meth:`.TimeConverter.unix_to_adc`."""
        assert time_converter.unix_to_adc(1234567890.0) == 0.0
        assert time_converter.unix_to_adc(1234567890.0 + 10.0) == 10485760.0

    def test_adc_to_unix(self, time_converter: TimeConverter) -> None:
        """Test :meth:`.TimeConverter.adc_to_unix`."""
        assert time_converter.adc_to_unix(0.0) == 1234567890.0
        assert time_converter.adc_to_unix(10485760.0) == 1234567890.0 + 10.0
