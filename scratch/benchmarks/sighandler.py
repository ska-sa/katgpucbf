################################################################################
# Copyright (c) 2023-2024, National Research Foundation (SARAO)
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

"""Manage asyncio shutdown when Ctrl-C is pressed."""

import asyncio
import signal
import sys


def add_sigint_handler() -> None:
    """Add a SIGINT handler to cancel the current task.

    This should be called as soon as possible by the coroutine launched by
    :func:`asyncio.run`. On Python 3.11+ it does nothing because Python
    already handles it.
    """
    if sys.version_info >= (3, 11, 0):
        return

    def handler(task: asyncio.Task) -> None:
        loop.remove_signal_handler(signal.SIGINT)
        task.cancel()

    loop = asyncio.get_running_loop()
    task = asyncio.current_task()
    assert task is not None
    loop.add_signal_handler(signal.SIGINT, handler, task)
