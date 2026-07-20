"""Classes for decoding VTP and VDIF stream statistics and validating framesets."""

import datetime
import io
import struct
from collections import defaultdict
from collections.abc import Generator
from dataclasses import dataclass

from baseband.vdif.frame import VDIFFrame
from dateutil.relativedelta import relativedelta


class VTPBuffer:
    """Data for storing VTP and VDIF stream statistics and validating framesets."""

    def __init__(self) -> None:
        self.seq_ids: list[int] = []
        self.thread_ids: list[int] = []
        self.seconds: list[int] = []
        self.frame_ids: list[int] = []
        self.samples_per_frame: int = 0
        self.ref_epoch: int = 0

    def add_packet(self, packet: bytes) -> None:
        """Add packet statistics to the buffer."""
        new_seq_id = struct.unpack("<Q", packet[:8])[0]
        frame = VDIFFrame.fromfile(io.BytesIO(packet[8:]))
        frame_id = frame.header["frame_nr"]
        if self.samples_per_frame == 0:
            self.samples_per_frame = frame.header.samples_per_frame
            self.ref_epoch = frame.header["ref_epoch"]
        self.seq_ids.append(new_seq_id)
        self.seconds.append(frame.header["seconds"])
        self.thread_ids.append(frame.header["thread_id"])
        self.frame_ids.append(frame_id)

    def clear(self) -> None:
        """Close the buffer."""
        self.seq_ids.clear()
        self.thread_ids.clear()
        self.seconds.clear()
        self.frame_ids.clear()
        self.samples_per_frame = 0
        self.ref_epoch = 0


@dataclass
class VDIFFramesetData:
    """Data for a VDIF frameset."""

    seq_ids: list[int]
    thread_ids: set[int]


@dataclass(frozen=True)
class VDIFFramesetKey:
    """Identifier of an VDIF frameset."""

    second: int
    frame_id: int
    ref_epoch: int

    def timestamp(self, frame_rate: int) -> float:
        """Timestamp of the VDIF frameset."""
        # ref_epoch is the number of half-years since 2000-01-01.
        return (
            datetime.datetime(2000, 1, 1)
            + relativedelta(months=6 * self.ref_epoch)
            + datetime.timedelta(seconds=self.second + ((float)(self.frame_id) / frame_rate))
        ).timestamp()


class VTPDecoder:
    """Decoder for sorting VTP and VDIF stream statistics and validating VDIF framesets."""

    def __init__(self, vtp_data: VTPBuffer, n_threads: int, bandwidth: float) -> None:
        """Initialize the VTPDecoder.

        Groups the sequence IDs, frame IDs, thread IDs and seconds ordered by sequence ID.

        Parameters
        ----------
        vtp_data
            VTPBuffer containing the VTP and VDIF stream statistics.
        n_threads
            Number of threads in the VDIF stream.
        bandwidth
            Stream bandwidth in Hz, used to derive the VDIF frame rate.
        """
        self.n_threads = n_threads
        self.vtp_meta_list: list[VDIFFramesetData] = []
        self.invalid_framesets: list[VDIFFramesetKey] = []
        self.frame_seq_map: dict[VDIFFramesetKey, VDIFFramesetData] = defaultdict()
        self.seq_ids: set[int] = set()
        self.frame_rate = round(bandwidth / vtp_data.samples_per_frame) if vtp_data.samples_per_frame != 0 else 0

        for i, seq_id in enumerate(vtp_data.seq_ids):
            key = VDIFFramesetKey(
                second=vtp_data.seconds[i], frame_id=vtp_data.frame_ids[i], ref_epoch=vtp_data.ref_epoch
            )

            if key not in self.frame_seq_map:
                self.frame_seq_map[key] = VDIFFramesetData(
                    seq_ids=[seq_id],
                    thread_ids={vtp_data.thread_ids[i]},
                )
            else:
                self.frame_seq_map[key].seq_ids.append(seq_id)
                self.frame_seq_map[key].thread_ids.add(vtp_data.thread_ids[i])

        for meta in self.frame_seq_map.values():
            meta.seq_ids.sort()

    def vtp_framesets(self) -> Generator[tuple[VDIFFramesetData, VDIFFramesetKey], None, None]:
        """
        Decode the VTP Sequence IDs for a complete VDIF frameset.

        Stores incomplete framesets in :attr:`invalid_framesets`.

        Yields
        ------
        list[int]
            Sequence IDs for a complete VDIF frameset and the key (second, frame_id).
        tuple[int, int]
            Key (second, frame_id) for the VDIF frameset.
        """
        for key, data in self.frame_seq_map.items():
            if any(seq_id in self.seq_ids for seq_id in data.seq_ids):
                self.invalid_framesets.append(key)
                continue
            if len(data.seq_ids) != self.n_threads:
                self.invalid_framesets.append(key)
                continue
            if len(data.thread_ids) != self.n_threads:
                self.invalid_framesets.append(key)
                continue
            yield (data, key)
            self.seq_ids.update(data.seq_ids)
