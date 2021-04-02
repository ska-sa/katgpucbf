"""TODO: Write this."""

import asyncio
from typing import List

import katsdpsigproc
import katsdpsigproc.abc
import katsdpsigproc.accel
import katsdpsigproc.resource


class XBEngineProcessingLoop:
    """TODO: Write this."""

    class CommunicationItem:
        """
        TODO: Write this.

        Facilitates processing between different loops
        Describe that this could become seperate items like katxgpu
        """

        timestamp: int
        events: List[katsdpsigproc.abc.AbstractEvent]
        spectra: katsdpsigproc.accel.DeviceArray

        def __init__(self, timestamp: int = 0) -> None:
            """TODO: Write this."""
            self.reset(timestamp)

        def reset(self, timestamp: int = 0) -> None:
            """TODO: Write this."""
            self.timestamp = timestamp
            self.events = []

        def add_event(self, event: katsdpsigproc.abc.AbstractEvent):
            """TODO: Write this."""
            self.events.append(event)

        async def async_wait_for_events(self):
            """TODO: Write this."""
            await katsdpsigproc.resource.async_wait_for_events(self.events)

        # def enqueue_wait(self, command_queue: katsdpsigproc.abc.AbstractCommandQueue) -> None:
        #     """
        #     TODO: Write this.

        #     Describe what a "wait for event" is
        #     The name is a bit misleading
        #     This adds the created events to the
        #     """
        #     print("Enquing Wait")
        #     print()
        #     command_queue.enqueue_wait_for_events(self.events)
        #     print("Enquing Wait done")

    def __init__(self):
        """TODO: Write this."""
        print("Created Processing Loop Object")

        self.context = katsdpsigproc.accel.create_some_context(device_filter=lambda x: x.is_cuda)

        # These queues are CUDA streams
        self._upload_command_queue = self.context.create_command_queue()
        self._proc_command_queue = self.context.create_command_queue()
        self._download_command_queue = self.context.create_command_queue()

        # Inter asyncio communication process - monitor make queue is a

    async def _receiver_loop(self):
        """TODO: Write this."""
        print("Receiver Loop Start")
        await asyncio.sleep(1)
        print("Receiver Loop End")

    async def _gpu_proc_loop(self):
        """TODO: Write this."""
        print("GPU Proc Loop Start")
        await asyncio.sleep(1)
        print("GPU Proc Loop End")

    async def _sender_loop(self):
        """TODO: Write this."""
        print("Sender Loop Start")
        await asyncio.sleep(1)
        print("Sender Loop End")

    async def run(self):
        """TODO: Write this."""
        # NOTE: Put in todo about upgrading this to python 3.8
        loop = asyncio.get_event_loop()
        receiver_task = loop.create_task(self._receiver_loop())
        gpu_proc_task = loop.create_task(self._gpu_proc_loop())
        sender_task = loop.create_task(self._sender_loop())
        await receiver_task
        await gpu_proc_task
        await sender_task
