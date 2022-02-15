from typing import Final
from numpy import uint16
import numpy as np
import numba

BYTE_BITS: Final = 8

@numba.njit
def _packbits(data: np.ndarray, bits: int) -> np.ndarray:  # pragma: nocover
    # Note: needs lots of explicit casting to np.uint64, as otherwise
    # numba seems to want to infer double precision.
    out = np.zeros(data.size * bits // BYTE_BITS, np.uint8)
    buf = np.uint64(0)
    buf_size = 0
    mask = (np.uint64(1) << bits) - np.uint64(1)
    out_pos = 0
    for v in data:
        buf = (buf << bits) | (np.uint64(v) & mask)
        buf_size += bits
        while buf_size >= BYTE_BITS:
            out[out_pos] = buf >> (buf_size - BYTE_BITS)
            out_pos += 1
            buf_size -= BYTE_BITS
    return out

@numba.njit
def unpackbits(packed_data):
    unpacked_data = []
    data_sample = np.int16(0)
    idx = 0

    for _ in range(len(packed_data)//5):
        tmp_40b_word = np.uint64(
            packed_data[idx] << (8*4) | 
            packed_data[idx+1] << (8*3)|
            packed_data[idx+2] << (8*2) |
            packed_data[idx+3] << 8 |
            packed_data[idx+4]
            )
        for _ in range(4):
            data_sample = (tmp_40b_word & 1098437885952) >> 30
            if data_sample > 511:
                data_sample = data_sample - 1024
            unpacked_data.append(np.int16(data_sample))
            tmp_40b_word = tmp_40b_word << 10

        idx += 5

    return unpacked_data

# Quick pack - unpack test. This is doesn'yt affect the spead processing.
# test_data = np.ones(64, np.uint16)
random_data = np.zeros(64, np.int8)

rng = np.random.default_rng(seed=2021)
random_data = rng.uniform(
        np.iinfo(random_data.dtype).min,
        np.iinfo(random_data.dtype).max,
        random_data.shape,
    ).astype(random_data.dtype)

packed_data = _packbits(random_data,10)
unpacked_data = unpackbits(packed_data)
np.testing.assert_array_equal(random_data, unpacked_data)