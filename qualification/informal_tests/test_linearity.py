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

"""CBF lineraity test."""

from typing import List, Tuple

import numpy as np
import pytest
import spead2.recv

from qualification.conftest import n_channels

from .. import BaselineCorrelationProductsReceiver, CorrelatorRemoteControl
from ..reporter import Reporter

from .. import antenna_channelised_voltage
import matplotlib.pyplot as plt

async def test_linearity(
    correlator: CorrelatorRemoteControl,
    receive_baseline_correlation_products: BaselineCorrelationProductsReceiver,
    pdf_report: Reporter,
) -> None:
    """Test that baseline Correlation Products are linear when input CW is scaled.

    Requirements verified:
        CBF.V.A.IF: CBF Linearity

    Verification method:

    Verify lineraity by testing that the channelised output scales linearly with a
    linear change to the CW input. 
    """
    pdf_report.step("Capture channelised data for various inpiut CW scales and check linearity.")

    # Channel selection. Compute desired channel frequency for Dsim config.
    CW_Scales = [1, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625, 0.001953125]

        
    rng = np.random.default_rng(seed=2021)
    random_channel_center = rng.uniform(0,correlator.n_chans)
    # freq = random_channel_center * (1712e6/(correlator.n_chans*2))

    freq_requested = 1024 # 107MHz

    # Set gain
    gain = 0.003
    await correlator.product_controller_client.request("gain-all", "antenna_channelised_voltage", gain)

    print(f'correlator.n_chans is {correlator.n_chans}')
    print(f'random_channel_center is {random_channel_center}')
    print(f'dsim_freq is {freq_requested*(856e6/correlator.n_chans)}')

    base_corr_prod = []
    for scale in CW_Scales:
        print(f'scale is {scale}')
        base_corr_prod.append(await antenna_channelised_voltage.sample_tone_response(rel_freqs=freq_requested, amplitude=scale, correlator=correlator, receiver=receive_baseline_correlation_products))
    
    linear_scale_result = []
    for product in base_corr_prod:
        linear_scale_result.append(product[freq_requested])

    linear_test_result = np.sqrt(linear_scale_result/np.max(linear_scale_result))

    print(f'tone is at: {np.where(base_corr_prod[0]==np.max(base_corr_prod[0]))}')

    plt.figure()
    plt.plot(10*np.log10(np.square(CW_Scales)))
    plt.plot(10*np.log10(np.square(linear_test_result)))
    plt.xlabel('CW Scale')
    plt.ylabel('dB')
    plt.title(f'CBF Linearity Test')
    labels = np.round(CW_Scales,5)
    labels = ['$2^{0}$', '$2^{-1}$', '$2^{-2}$', '$2^{-3}$', '$2^{-4}$', '$2^{-5}$', '$2^{-6}$', '$2^{-7}$', '$2^{-8}$', '$2^{-9}$']
    plt.xticks(np.arange(0, len(linear_test_result), step=1),labels=labels)
    plt.savefig('linearity.png')
    breakpoint()