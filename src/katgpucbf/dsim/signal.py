"""Synthesis of simulated signals."""

import operator
from abc import ABC, abstractmethod
from typing import Callable

import numba
import numpy as np
from numpy.typing import ArrayLike

from .. import BYTE_BITS


class Signal(ABC):
    """Abstract base class for signals.

    An instance is simply a real-valued function of time, for a single
    polarisation.
    """

    @abstractmethod
    def sample(self, timestamp: int, n: int, frequency: float) -> np.ndarray:
        """Sample the signal at regular intervals.

        The returned values should be scaled to the range (-1, 1).

        Parameters
        ----------
        timestamp
            Time (in samples since the sync epoch) of the first returned sample.
        n
            Number of samples to generate
        frequency
            Frequency of samples (Hz)

        Returns
        -------
        samples
            Array of samples, float64.
        """

    def __add__(self, other) -> "Signal":
        if not isinstance(other, Signal):
            return NotImplemented
        return CombinedSignal(self, other, operator.add)

    def __mul__(self, other) -> "Signal":
        if not isinstance(other, Signal):
            return NotImplemented
        return CombinedSignal(self, other, operator.mul)


class CombinedSignal(Signal):
    """Signal built by combining two other signals.

    Parameters
    ----------
    a, b
        Input signals
    combine
        Operator to combine two arrays
    """

    def __init__(self, a: Signal, b: Signal, combine: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> None:
        self.a = a
        self.b = b
        self.combine = combine

    def sample(self, timestamp: int, n: int, frequency: float) -> np.ndarray:  # noqa: D102
        return self.combine(self.a.sample(timestamp, n, frequency), self.b.sample(timestamp, n, frequency))


class CW(Signal):
    """Continuous wave."""

    def __init__(self, amplitude: float, frequency: float) -> None:
        self.amplitude = amplitude
        self.frequency = frequency

    def sample(self, timestamp: int, n: int, frequency: float) -> np.ndarray:  # noqa: D102
        return np.cos(np.arange(timestamp, timestamp + n) * (self.frequency / frequency * 2 * np.pi)) * self.amplitude


def quantise(data: ArrayLike, bits: int, dither: bool = True) -> np.ndarray:
    """Convert floating-point data to fixed-point.

    Parameters
    ----------
    data
        Array of values, nominally in the range -1 to 1 (values outside the
        range are clamped).
    bits
        Total number of bits per output sample (including the sign bit). The
        input values are scaled by :math:`2^{bits-1} - 1`.
    dither
        If true, add uniform random values in the range [-0.5, 0.5) after
        scaling to reduce artefacts.
    """
    scale = 2 ** (bits - 1) - 1
    scaled = np.asarray(data) * scale
    if dither:
        # TODO: should it be seeded in some way?
        rng = np.random.default_rng()
        scaled += rng.uniform(low=-0.5, high=0.5, size=scaled.size)
    return np.rint(np.clip(scaled, -scale, scale)).astype(np.int32)


@numba.njit
def _packbits(input: np.ndarray, output: np.ndarray, bits: int) -> None:  # pragma: nocover
    # Note: needs lots of explicit casting to np.uint64, as otherwise
    # numba seems to want to infer double precision.
    buf = np.uint64(0)
    buf_size = 0
    mask = (np.uint64(1) << bits) - np.uint64(1)
    out_pos = 0
    for v in input:
        buf = (buf << bits) | (np.uint64(v) & mask)
        buf_size += bits
        while buf_size >= BYTE_BITS:
            output[out_pos] = buf >> (buf_size - BYTE_BITS)
            out_pos += 1
            buf_size -= BYTE_BITS


def packbits(data: ArrayLike, bits: int) -> np.ndarray:
    """Pack integers into bytes.

    The least-significant `bits` bits of each integer in `data` is collected
    together in big-endian order, and returned as a sequence of bytes. The
    total number of bits must form a whole number of bytes.

    If `data` is multi-dimensional it is flattened.
    """
    array = np.asarray(data).ravel()
    if bits * array.size % BYTE_BITS:
        raise ValueError("Bits do not form a whole number of bytes")
    out = np.empty(bits * array.size // BYTE_BITS, np.uint8)
    _packbits(array, out, bits)
    return out
