from typing import Any, Optional, Sequence, Tuple

from katsdpsigproc.accel import DeviceArray

from .monitor import Monitor

class Chunk:
    timestamp: int
    channels: int
    acc_len: int
    frames: int
    pols: int
    base: object  # Python buffer protocol
    device: Optional[DeviceArray]
    def __init__(self, base: object, device: Optional[DeviceArray] = None) -> None: ...

class Ringbuffer:
    def __init__(self, cap: int) -> None: ...
    def pop(self) -> Chunk: ...
    def try_pop(self) -> Chunk: ...
    def try_push(self, item: Chunk) -> None: ...
    @property
    def data_fd(self) -> int: ...

class Sender:
    def __init__(
        self,
        free_ring_capacity: int,
        memory_regions: Sequence[object],
        thread_affinity: int,
        comp_vector: int,
        feng_id: int,
        num_ants: int,
        endpoints: Sequence[Tuple[str, int]],
        ttl: int,
        interface_address: str,
        ibv: bool,
        max_packet_size: int,
        rate: float,
        max_heaps: int,
        monitor: Monitor = ...,
    ) -> None: ...
    def send_chunk(self, chunk: Chunk) -> None: ...
    def stop(self) -> None: ...
    def push_free_ring(self, chunk: Chunk) -> None: ...
    @property
    def free_ring(self) -> Ringbuffer: ...
    @property
    def num_substreams(self) -> int: ...
