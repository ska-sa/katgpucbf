# noqa: D104

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

from typing import Final

METRIC_NAMESPACE: Final = "xbgpu"

DEFAULT_XPIPELINE_NAME: Final = "xpipeline"
DEFAULT_BPIPELINE_NAME: Final = "bpipeline"

# NOTE: Too high means too much GPU memory gets allocated
DEFAULT_N_IN_ITEMS: Final = 3
DEFAULT_N_OUT_ITEMS: Final = 2
