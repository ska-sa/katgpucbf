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

"""Test probability of saturation."""

import math

import pytest
import scipy.stats
from pytest_check import check

from ..recv import BaselineCorrelationProductsReceiver
from ..reporter import Reporter


@pytest.mark.requirements("CBF-REQ-0103")
async def test_saturation(
    receive_baseline_correlation_products: BaselineCorrelationProductsReceiver,
    pdf_report: Reporter,
) -> None:
    r"""Check probability of saturation.

    Verification method
    -------------------
    Verification by analysis. Saturation is most likely for autocorrelations.
    Consider input (channelised) noise :math:`X` where :math:`E[|X|^2] =
    \sigma^2`. Let :math:`k` be the number of spectra accumulated, and the
    autocorrelation of :math:`X` be :math:`Y`. Then :math:`\frac{2Y}{\sigma^2}`
    has a :math:`\chi^2` distribution with :math:`2k` degrees of freedom (the
    factors of 2 are because :math:`X` is a complex variable whose real and
    imaginary components are independent Gaussian distributions).  We can use
    this to analytically determine the value of :math:`\sigma` for which
    :math:`P[Y \ge 2^{31}] = 10^{-4}`.
    """
    pdf_report.step("Calculate sigma that gives 0.01% probability of saturation.")
    dist = scipy.stats.chi2(2 * receive_baseline_correlation_products.n_spectra_per_acc)
    z = dist.isf(1e-4)
    # Solve 2 * 2^31 / sigma^2 = z
    sigma = math.sqrt(2**32 / z)
    pdf_report.detail(f"sigma = {sigma:.2f}")
    with check:
        assert sigma > 3
