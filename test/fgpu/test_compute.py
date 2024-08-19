################################################################################
# Copyright (c) 2020-2024, National Research Foundation (SARAO)
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
from katgpucbf.fgpu.engine import generate_ddc_weights
from katgpucbf.utils import DitherType

pytestmark = [pytest.mark.cuda_only]


@pytest.mark.parametrize("mode", ["wideband", "narrowband"])
@pytest.mark.parametrize("dither", DitherType)
def test_compute(context: AbstractContext, command_queue: AbstractCommandQueue, mode: str, dither: DitherType) -> None:
    """Test creation and running of :class:`Compute`.

    .. todo:: This isn't a proper test, just a smoke test.
    """
    channels = 32768
    dig_sample_bits = 10
    out_bits = 8
    decimation = 8
    ddc_taps = 128
    pfb_taps = 4
    spectra_per_heap = 32
    subsampling = decimation
    nb_spectra = 64

    if mode == "wideband":
        narrowband: compute.NarrowbandConfig | None = None
        spectra = 160
        internal_channels = channels
    else:
        narrowband = compute.NarrowbandConfig(
            decimation=decimation,
            mix_frequency=0.2,
            weights=generate_ddc_weights(ddc_taps, decimation, 0.1),
        )
        spectra = nb_spectra
        internal_channels = 2 * channels
    template = compute.ComputeTemplate(context, pfb_taps, channels, dig_sample_bits, out_bits, dither, narrowband)
    # The sample count is the minimum that will produce the required number of
    # output spectra for narrowband mode. For wideband there is more headroom.
    fn = template.instantiate(
        command_queue,
        nb_spectra * internal_channels * subsampling
        + ddc_taps
        + ((pfb_taps - 1) * internal_channels - 1) * subsampling,
        spectra,
        spectra_per_heap,
        seed=123,
        sequence_first=456,
    )
    fn.ensure_all_bound()
    fn()
