################################################################################
# Copyright (c) 2020-2023, National Research Foundation (SARAO)
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

"""Smoke test for Compute class."""
import pytest
from katsdpsigproc.abc import AbstractCommandQueue, AbstractContext

from katgpucbf.fgpu import compute

pytestmark = [pytest.mark.cuda_only]


@pytest.mark.parametrize("mode", ["wideband", "narrowband"])
def test_compute(context: AbstractContext, command_queue: AbstractCommandQueue, mode: str) -> None:
    """Test creation and running of :class:`Compute`.

    .. todo:: This isn't a proper test, just a smoke test.
    """
    if mode == "wide":
        narrowband: compute.NarrowbandConfig | None = None
        spectra = 1280
    else:
        narrowband = compute.NarrowbandConfig(decimation=8, taps=256, mix_frequency=0.2)
        spectra = 256
    template = compute.ComputeTemplate(context, 4, 32768, 10, narrowband)
    # The sample count is the minimum that will produce the required number of
    # output spectra for narrowband mode. For wideband there is more headroom.
    fn = template.instantiate(command_queue, 2**27 + 256 + ((4 - 1) * 32768 - 1) * 16, spectra, 256)
    fn.ensure_all_bound()
    fn()
