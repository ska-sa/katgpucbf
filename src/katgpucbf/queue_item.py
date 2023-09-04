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

"""Provide :class:`QueueItem`."""

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
    #: The latest GPU event marker per
    #: :class:`~katsdpsigproc.abc.AbstractCommandQueue`.
    _events: dict[AbstractCommandQueue, AbstractEvent]

    def __init__(self, timestamp: int = 0) -> None:
        self.reset(timestamp)
        self._events = {}

    def add_marker(self, command_queue: AbstractCommandQueue) -> AbstractEvent:
        """Add an event to the list of events in the QueueItem.

        The event represents all previous work enqueued to `command_queue`.
        """
        marker = command_queue.enqueue_marker()
        self._events[command_queue] = marker
        return marker

    def enqueue_wait_for_events(self, command_queue: AbstractCommandQueue) -> None:
        """Block execution of a command queue until all of this item's events are finished.

        Future work enqueued to `command_queue` will be sequenced after any
        work associated with the stored events.
        """
        command_queue.enqueue_wait_for_events(list(self._events.values()))

    async def async_wait_for_events(self) -> None:
        """Wait for all events in the list of events to be complete."""
        events = self._events.copy()
        await katsdpsigproc.resource.async_wait_for_events(events.values())
        # We can remove the events we waited for. We can't just clear the
        # entire dict, because another task may have asynchronously added
        # events in the meantime.
        for queue, event in events.items():
            if self._events.get(queue) is event:
                del self._events[queue]

    def reset(self, timestamp: int = 0) -> None:
        """Reset the item's timestamp.

        Subclasses should override this to reset other state. It is called by
        the constructor so it can also be used for initialisation.
        """
        self.timestamp = timestamp

    @property
    def events(self) -> tuple[AbstractEvent, ...]:
        """Get a copy of the currently registered events."""
        return tuple(self._events.values())
