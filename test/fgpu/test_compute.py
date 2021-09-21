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

"""Smoke test for Compute class."""
from katsdpsigproc import accel

from katgpucbf.fgpu import compute


def test_compute():
    """Test creation and running of :class:`Compute`.

    .. todo:: This isn't a proper test, just a smoke test.
    """
    ctx = accel.create_some_context(interactive=False)
    queue = ctx.create_command_queue()

    template = compute.ComputeTemplate(ctx, 4)
    fn = template.instantiate(queue, 100000000, 1280, 256, 32768)
    fn.ensure_all_bound()
    fn()
