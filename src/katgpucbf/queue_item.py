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

"""Provide :class:`QueueItem`."""

from typing import List

import katsdpsigproc.resource
from katsdpsigproc.abc import AbstractCommandQueue, AbstractEvent


class QueueItem:
    """Queue Item for use in synchronisation between command queues.

    Derived classes will have allocated memory regions associated with them,
    appropriately sized for input or output data. Actions (whether kernel
    executions or copies to or from the device) for these memory regions are
    initiated, and then an event marker is added to the list in some variation
    of this manner:

    .. code-block:: python

        my_item.add_marker(command_queue)

    The item can then be passed through a queue to the next stage in the
    program, which waits for the operations to be complete using
    :meth:`enqueue_wait_for_events` or :meth:`async_wait_for_events`.
    This indicates that the operation is complete and the next thing can be
    done with whatever data is in that region of memory.
    """

    #: Timestamp of the item.
    timestamp: int
    #: A list of GPU event markers generated by an
    #: :class:`~katsdpsigproc.abc.AbstractCommandQueue`.
    events: List[AbstractEvent]

    def __init__(self, timestamp: int = 0) -> None:
        self.reset(timestamp)

    def add_marker(self, command_queue: AbstractCommandQueue) -> None:
        """Add an event to the list of events in the QueueItem.

        The event represents all previous work enqueued to `command_queue`.
        """
        self.events.append(command_queue.enqueue_marker())

    def enqueue_wait_for_events(self, command_queue: AbstractCommandQueue) -> None:
        """Block execution of a command queue until all of this item's events are finished.

        Future work enqueued to `command_queue` will be sequenced after any
        work associated with the stored events.
        """
        command_queue.enqueue_wait_for_events(self.events)

    async def async_wait_for_events(self) -> None:
        """Wait for all events in the list of events to be complete."""
        await katsdpsigproc.resource.async_wait_for_events(self.events)

    def reset(self, timestamp: int = 0) -> None:
        """Reset the item's timestamp, and empty the event list.

        Subclasses should override this to reset other state. It is called by
        the constructor so it can also be used for initialisation.
        """
        self.timestamp = timestamp
        self.events = []