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

"""Module containing correlator configuration options that should be tested where appropriate in all unit tests."""

# These are the estimated subarray sizes that will be run.  Values 5, 23, 61 and
# 19 are just there to test that various non-power-of-two array sizes will
# run.
# array_size = [5, 23, 61, 19, 4, 8, 16, 32, 64, 84]
array_size = [5]

# This is always set to 256 for the MeerKAT case. There is a chance that for
# the MeerKAT extension this could be configured to other values. All these
# cases need to be tested.
# num_spectra_per_heap = [256, 512, 1024]
num_spectra_per_heap = [256]

# Number of FFT channels out of the F-Engine
# num_channels = [1024, 4096, 8192, 32768]
num_channels = [1024]
