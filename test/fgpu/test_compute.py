################################################################################
# Copyright (c) 2020-2025, National Research Foundation (SARAO)
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

from fractions import Fraction

import pytest
from katsdpsigproc.abc import AbstractCommandQueue, AbstractContext

from katgpucbf.fgpu import compute
from katgpucbf.fgpu.engine import _generate_ddc_weights_discard, _generate_ddc_weights_no_discard
from katgpucbf.utils import DitherType

pytestmark = [pytest.mark.cuda_only]


@pytest.mark.parametrize("mode", ["wideband", "narrowband-discard", "narrowband-no-discard"])
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
    nb_spectra = 64

    if mode == "wideband":
        narrowband: compute.NarrowbandConfig | None = None
        spectra = 160
    elif mode == "narrowband-discard":
        subsampling = decimation
        narrowband = compute.NarrowbandConfig(
            decimation=decimation,
            mix_frequency=Fraction(1, 5),
            weights=_generate_ddc_weights_discard(ddc_taps, subsampling, 0.1),
            discard=False,
        )
        spectra = nb_spectra
    elif mode == "narrowband-no-discard":
        subsampling = 2 * decimation
        narrowband = compute.NarrowbandConfig(
            decimation=decimation,
            mix_frequency=Fraction(1, 5),
            weights=_generate_ddc_weights_no_discard(ddc_taps, decimation, 0.9, 0.1),
            discard=True,
        )
        spectra = nb_spectra
    else:
        raise RuntimeError("unexpected mode")

    template = compute.ComputeTemplate(context, pfb_taps, channels, dig_sample_bits, out_bits, dither, narrowband)
    # The sample count is large enough to produce the required number of
    # output spectra for both narrowband modes. For wideband there is more
    # headroom.
    fn = template.instantiate(
        command_queue,
        nb_spectra * 2 * channels * decimation + ddc_taps + (pfb_taps - 1) * 2 * channels * decimation,
        spectra,
        spectra_per_heap,
        seed=123,
        sequence_first=456,
    )
    fn.ensure_all_bound()
    fn()
