################################################################################
# Copyright (c) 2021-2024, National Research Foundation (SARAO)
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

"""Synthesis of simulated signals."""

import asyncio
import logging
import math
import multiprocessing.connection
import operator
import os
import signal
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import ClassVar

import dask.array as da
import numba
import numpy as np
import pyparsing as pp
import xarray as xr

from .. import BYTE_BITS

logger = logging.getLogger(__name__)
#: Dask chunk size for sampling signals (must be a multiple of 8)
CHUNK_SIZE = 2**20


def _sample_helper(n: int, chunk_data: da.Array, sample_chunk: Callable, **kwargs) -> da.Array:
    """Help implement :meth:`sample` in subclasses of :class:`Signal`.

    This can be used when each chunk can be generated from a single value
    per chunk (plus some chunk-independent state).

    Currently the call to `sample_chunk` for the final chunk will use the
    full chunk size, unless `n` is less than CHUNK_SIZE. In future it may
    take care of truncating the final chunk.

    Parameters
    ----------
    n
        Number of samples to generate
    chunk_data
        Array holding per-chunk data. It must have the same number of
        chunks as the output.
    sample_chunk
        Callback to generate a single chunk. It is passed a chunk from
        `chunk_data` as a positional argument, and `chunk_size` as a
        keyword argument.
    kwargs
        Additional keyword arguments forwarded to `sample_chunk`.
    """
    chunk_size = min(n, CHUNK_SIZE)
    n_chunks = (n + chunk_size - 1) // chunk_size
    return da.map_blocks(
        sample_chunk,
        chunk_data,
        chunks=((chunk_size,) * n_chunks,),
        chunk_size=chunk_size,
        **kwargs,
    )[:n]


def _sample_helper_random(n: int, seed_seq: np.random.SeedSequence, sample_chunk: Callable, **kwargs) -> da.Array:
    """Generate a 1D dask array of random data.

    The random generators are seeded from `seed_seq`. Other parameters are
    passed to :func:`_sample_helper`.
    """
    n_chunks = (n + CHUNK_SIZE - 1) // CHUNK_SIZE
    seed_seqs = seed_seq.spawn(n_chunks)
    # Chunk size of 1 so that we can map each SeedSequence to an output chunk
    seed_seqs_dask = da.from_array(np.array(seed_seqs, dtype=object), 1)
    return _sample_helper(n, seed_seqs_dask, sample_chunk, **kwargs)


class TerminalError(TypeError):
    """Indicate that a terminal signal has been used in an expression."""

    def __init__(self, signal: "Signal") -> None:
        self.signal = signal
        assert signal.terminal
        super().__init__(f"Signal '{signal}' cannot be used in a larger expression")


class Signal(ABC):
    """Abstract base class for signals.

    An instance is simply a real-valued function of time, for a single
    polarisation.
    """

    @abstractmethod
    def sample(self, n: int, sample_rate: float) -> da.Array:
        """Sample the signal at regular intervals.

        The returned values should be scaled to the range (-1, 1).

        .. note::

           Calling this method with two different values of `n` may
           yield results that are not consistent with each other.

        Parameters
        ----------
        n
            Number of samples to generate
        sample_rate
            Frequency of samples (Hz)

        Returns
        -------
        samples
            Dask array of samples, float32. The chunk size must be CHUNK_SIZE.
        """

    @property
    def terminal(self) -> bool:
        """Indicate whether the signal is terminal.

        Terminal signals cannot be combined into larger expressions, because
        they contain information about how to handle their postprocessing.
        """
        return False

    def __add__(self, other) -> "Signal":
        if not isinstance(other, Signal):
            return NotImplemented
        return CombinedSignal(self, other, operator.add, "+")

    def __sub__(self, other) -> "Signal":
        if not isinstance(other, Signal):
            return NotImplemented
        return CombinedSignal(self, other, operator.sub, "-")

    def __mul__(self, other) -> "Signal":
        if not isinstance(other, Signal):
            return NotImplemented
        return CombinedSignal(self, other, operator.mul, "*")


@dataclass
class CombinedSignal(Signal):
    """Signal built by combining two other signals.

    Parameters
    ----------
    a, b
        Input signals
    combine
        Operator to combine two arrays
    op_name
        Symbol for the operator
    """

    a: Signal
    b: Signal
    combine: Callable[[da.Array, da.Array], da.Array]
    op_name: str

    def __post_init__(self):
        if self.a.terminal:
            raise TerminalError(self.a)
        if self.b.terminal:
            raise TerminalError(self.b)

    def sample(self, n: int, sample_rate: float) -> da.Array:  # noqa: D102
        return self.combine(self.a.sample(n, sample_rate), self.b.sample(n, sample_rate))

    def __str__(self) -> str:
        return f"({self.a} {self.op_name} {self.b})"


@dataclass
class Constant(Signal):
    """Fixed value."""

    value: float

    def sample(self, n: int, sample_rate: float) -> da.Array:  # noqa: D102
        return da.full((n,), self.value, dtype=np.float32, chunks=CHUNK_SIZE)

    def __str__(self) -> str:
        return f"{self.value}"


@dataclass
class Periodic(Signal):
    """Base class for period signals.

    The frequency is adjusted during sampling so that the sampled result can
    be looped.
    """

    amplitude: float
    frequency: float
    _class_name: ClassVar[str] = ""

    @staticmethod
    @abstractmethod
    def _sample_chunk(offset: np.ndarray, *, amplitude: np.float32, chunk_size: int, frequency: float) -> np.ndarray:
        """Compute a single chunk.

        Parameters
        ----------
        offset
            Index of the first element of the chunk (1-element array)
        amplitude
            Amplitude of the signal
        chunk_size
            Number of samples to produce
        frequency
            Rounded frequency, expressed as cycles per sample
        """

    def sample(self, n: int, sample_rate: float) -> da.Array:  # noqa: D102
        # Round target frequency to fit an integer number of waves into signal_heaps
        waves = max(1, round(n * self.frequency / sample_rate))
        frequency = waves * sample_rate / n
        logger.info(f"Rounded tone frequency to {frequency} Hz")

        # Index of the first element of each chunk
        offsets = da.arange(0, n, CHUNK_SIZE, chunks=1, dtype=np.int64)
        return _sample_helper(
            n,
            offsets,
            self._sample_chunk,
            amplitude=np.float32(self.amplitude),
            frequency=waves / n,
            meta=np.array((), np.float32),
        )

    def __str__(self) -> str:
        return f"{self._class_name}({self.amplitude}, {self.frequency})"


@dataclass
class CW(Periodic):
    """Continuous wave.

    To make the resulting signal periodic, the frequency is adjusted during
    sampling so that the sampled result can be looped.
    """

    _class_name: ClassVar[str] = "cw"

    @staticmethod
    def _sample_chunk(offset: np.ndarray, *, amplitude: np.float32, chunk_size: int, frequency: float) -> np.ndarray:
        # Compute the complex exponential. Because it is being regularly
        # sampled, it is possible to do this efficiently by repeated
        # doubling. This also makes it possible to keep most of the
        # computation in single precision without losing much precision
        # (experimentally the results seem to be off by less than 1e-6).
        scale = frequency * (2 * np.pi)
        out = np.empty(chunk_size, np.complex64)
        out[0] = np.exp(offset[0] * scale * 1j) * amplitude
        valid = 1
        while valid < chunk_size:
            # Rotate the segment [0, valid) by valid steps, giving the segment
            # [valid, 2 * value). It's slightly complicated to handle the case
            # where we have to truncate to chunk_size.
            add = min(valid, chunk_size - valid)
            rot = np.exp(valid * scale * 1j).astype(np.complex64)
            np.multiply(out[0:add], rot, out[valid : valid + add])
            valid += add
        return out.real


@dataclass
class Comb(Periodic):
    """Signal with periodic impulses.

    To make the resulting signal periodic, the frequency is adjusted during
    sampling so that the sampled result can be looped.
    """

    _class_name: ClassVar[str] = "comb"

    @staticmethod
    def _sample_chunk(offset: np.ndarray, *, amplitude: np.float32, chunk_size: int, frequency: float) -> np.ndarray:
        start = offset[0]
        stop = start + chunk_size
        start_cycle = math.ceil(start * frequency)
        stop_cycle = math.ceil(stop * frequency)
        # Rounding errors can make start_cycle/stop_cycle be off by 1. So we
        # include an extra cycle on each end to ensure we don't miss anything
        # at the edges, then trim again if necessary.
        cycles = np.arange(start_cycle - 1, stop_cycle + 1)
        indices = np.rint(cycles / np.float32(frequency)).astype(int)
        indices = indices[(start <= indices) & (indices < stop)]
        out = np.zeros(chunk_size, np.float32)
        out[indices - start] = amplitude
        return out


@dataclass
class Random(Signal):
    """Base class for randomly-generated signals.

    This base class is only suitable when the samples at different times
    are independent. The derived class must implement :meth:`_sample_chunk`.
    """

    entropy: int  #: entropy used to populate a :class:`np.random.SeedSequence`

    def __init__(self, entropy: int | None = None) -> None:
        self.entropy = entropy if entropy is not None else self._generate_entropy()

    def _generate_entropy(self) -> int:
        """Generate a random seed.

        This is split into a separate method so that it can be easily mocked.
        """
        # In general SeedSequence.entropy is not always an int, but it is for
        # this usage.
        return np.random.SeedSequence().entropy  # type: ignore

    @abstractmethod
    def _sample_chunk(self, seed_seq: Sequence[np.random.SeedSequence], *, chunk_size: int) -> np.ndarray:
        """Sample random values from a single chunk.

        Parameters
        ----------
        seed_seq
            A single-element list with the entropy to use to initialise a random generator
        chunk_size
            The number of elements to generate
        """

    def sample(self, n: int, sample_rate: float) -> da.Array:  # noqa: D102
        seed_seq = np.random.SeedSequence(self.entropy)
        return _sample_helper_random(n, seed_seq, self._sample_chunk, meta=np.array((), np.float32))


@dataclass
class WGN(Random):
    """White Gaussian Noise signal.

    Each sample in time is an independent Gaussian random variable with zero
    mean and a given standard deviation.

    In practice, the signal has a period equal to the value of `n` given to
    :meth:`sample`, which could lead to undesirable correlations.

    Parameters
    ----------
    std
        Standard deviation of the samples
    entropy
        If provided, used to seed the random number generator
    """

    std: float = 1.0  #: standard deviation

    def __init__(self, std: float, entropy: int | None = None) -> None:
        # __init__ is overridden to change the argument order
        super().__init__(entropy)
        self.std = std

    def _sample_chunk(self, seed_seq: Sequence[np.random.SeedSequence], *, chunk_size: int) -> np.ndarray:
        # The RNG is initialised every time this is called so that it will
        # produce the same results.
        rng = np.random.default_rng(seed_seq[0])
        return rng.standard_normal(size=chunk_size, dtype=np.float32) * self.std

    def __str__(self) -> str:
        return f"wgn({self.std}, {self.entropy})"


@dataclass
class Delay(Signal):
    """Delay another signal by an integer number of samples.

    Parameters
    ----------
    signal
        Underlying signal to delay
    delay
        Number of samples to delay the signal (may be negative)
    """

    signal: Signal
    delay: int

    def __post_init__(self) -> None:
        if self.signal.terminal:
            raise TerminalError(self.signal)

    def sample(self, n: int, sample_rate: float) -> da.Array:  # noqa: D102
        return da.roll(self.signal.sample(n, sample_rate), self.delay)

    def __str__(self) -> str:
        return f"delay({self.signal}, {self.delay})"


@dataclass
class Nodither(Signal):
    """Mark a signal expression as not needing dither.

    Parameters
    ----------
    signal
        Underlying signal
    """

    signal: Signal

    def __post_init__(self) -> None:
        if self.signal.terminal:
            raise TerminalError(self.signal)

    def sample(self, n: int, sample_rate: float) -> da.Array:  # noqa: D102
        return self.signal.sample(n, sample_rate)

    def __str__(self) -> str:
        return f"nodither({self.signal})"

    @property
    def terminal(self) -> bool:
        """Prevent this signal from being used in expressions."""
        return True


def _apply_operator(s: str, loc: int, tokens: pp.ParseResults) -> Signal:
    assert len(tokens) == 1
    tokens = tokens[0]  # infix_operator passes the expression with an extra nesting level
    op_map = {"*": operator.mul, "+": operator.add, "-": operator.sub}
    result = tokens[0]
    for i in range(1, len(tokens), 2):
        result = op_map[tokens[i]](result, tokens[i + 1])
    return result


def parse_signals(prog: str) -> list[Signal]:
    """Generate a set of signals from a domain-specific language.

    See :ref:`dsim-dsl` for a description of the language.
    """
    var_table = {}
    output = []

    def assign(s: str, loc: int, tokens: pp.ParseResults) -> None:
        var_table[tokens[0]] = tokens[1]

    def get_variable(s: str, loc: int, tokens: pp.ParseResults) -> None:
        try:
            return var_table[tokens[0]]
        except KeyError:
            raise pp.ParseFatalException("", loc, f"Unknown variable {tokens[0]!r}")

    lpar = pp.Suppress("(")
    rpar = pp.Suppress(")")
    comma = pp.Suppress(",")
    eq = pp.Suppress("=")
    semicolon = pp.Suppress(";")

    variable = pp.pyparsing_common.identifier("variable")
    real = pp.pyparsing_common.number
    integer = pp.pyparsing_common.integer
    signed_integer = pp.pyparsing_common.signed_integer
    expr = pp.Forward()
    # See https://pyparsing-docs.readthedocs.io/en/latest/HowToUsePyparsing.html#expression-subclasses
    # for an explanation of + versus - in these rules (it helps give more
    # useful errors). I've been conservative with the use of - i.e., there
    # may be some +'s that can still be changed to -'s.
    cw = pp.Keyword("cw") + lpar - real + comma - real + rpar
    cw.set_parse_action(lambda s, loc, tokens: CW(tokens[1], tokens[2]))
    comb = pp.Keyword("comb") + lpar - real + comma - real + rpar
    comb.set_parse_action(lambda s, loc, tokens: Comb(tokens[1], tokens[2]))
    wgn = pp.Keyword("wgn") + lpar - real + pp.Opt(comma - integer("entropy")) + rpar
    wgn.set_parse_action(lambda s, loc, tokens: WGN(tokens[1], tokens.get("entropy")))
    delay = pp.Keyword("delay") + lpar - expr + comma - signed_integer + rpar
    delay.set_parse_action(lambda s, loc, tokens: Delay(tokens[1], tokens[2]))
    nodither = pp.Keyword("nodither") + lpar - expr + rpar
    nodither.set_parse_action(lambda s, loc, tokens: Nodither(tokens[1]))
    variable_expr = variable.copy()
    variable_expr.set_parse_action(get_variable)
    real_expr = real.copy()
    real_expr.set_parse_action(lambda s, loc, tokens: Constant(float(tokens[0])))

    atom = real_expr | cw | comb | wgn | delay | nodither | variable_expr
    expr <<= pp.infix_notation(
        atom, [("*", 2, pp.OpAssoc.LEFT, _apply_operator), (pp.one_of("+ -"), 2, pp.OpAssoc.LEFT, _apply_operator)]
    )
    assignment = variable + eq - expr
    assignment.set_parse_action(assign)
    expr_statement = expr.copy()
    expr_statement.add_parse_action(lambda s, loc, tokens: output.append(tokens[0]))
    statement = (assignment | expr_statement) - semicolon
    program = statement[...]

    program.parse_string(prog, parse_all=True)
    return output


def format_signals(signals: Sequence[Signal]) -> str:
    """Inverse of :func:`parse_signals`.

    Currently object identity is not preserved, so if a simple signal is
    re-used multiple times (e.g., shared across output signals), it will
    be repeated in the output. This is subject to change.
    """
    return "; ".join(str(s) for s in signals) + ";"


def _dither_sample_chunk(seed_seq: Sequence[np.random.SeedSequence], *, chunk_size: int) -> np.ndarray:
    """Produce one chunk for :func:`dither`."""
    rng = np.random.default_rng(seed_seq[0])
    return rng.random(size=chunk_size, dtype=np.float32) - np.float32(0.5)


def make_dither(n_pols: int, n: int, entropy: int | None = None) -> xr.DataArray:
    """Create a set of dither signals to use with :func:`quantise`.

    The returned array has ``pol`` and ``data`` axes, and is backed by a Dask
    array.

    The implementation currently uses a uniform distribution, but that is
    subject to change.
    """
    seed_seqs = np.random.SeedSequence(entropy).spawn(n_pols)
    d = da.stack(
        [
            _sample_helper_random(n, seed_seq, _dither_sample_chunk, meta=np.array((), np.float32))
            for seed_seq in seed_seqs
        ]
    )
    return xr.DataArray(d, dims=["pol", "data"])


@numba.njit
def _clip(a, a_min, a_max):
    """Like np.clip, but for scalars.

    It's not working in numba: https://github.com/numba/numba/issues/3469.
    """
    if a < a_min:
        return a_min
    elif a > a_max:
        return a_max
    else:
        return a


@numba.njit(nogil=True)
def _quantise_chunk(chunk: np.ndarray, dither: np.ndarray, scale: np.float32) -> np.ndarray:
    out = np.empty_like(chunk, dtype=np.int32)
    for i in range(chunk.shape[0]):
        scaled = chunk[i] * scale + dither[i]
        out[i] = np.rint(_clip(scaled, -scale, scale))
    return out


def quantise(
    data: da.Array,
    bits: int,
    dither: da.Array,
) -> da.Array:
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
        Values to add to the data after scaling.
    """
    scale = np.float32(2 ** (bits - 1) - 1)
    return da.blockwise(_quantise_chunk, "i", data, "i", dither, "i", scale=scale, meta=np.array((), np.int32))


@numba.njit(nogil=True)
def _saturation_counts(data: np.ndarray, saturation_value: np.integer) -> np.ndarray:
    out = np.empty((data.shape[0], 1), np.uint64)
    for i in range(data.shape[0]):
        total = np.uint64(0)
        for j in range(data.shape[1]):
            total += np.uint64(data[i, j] >= saturation_value or data[i, j] <= -saturation_value)
        out[i, 0] = total
    return out


def saturation_counts(data: da.Array, saturation_value) -> da.Array:
    """Return an array indicating counts of saturated elements of ``data``.

    The count is taken along each row of ``data``.

    Elements are considered saturated if they exceed `saturation_value` in
    absolute value.
    """
    assert data.ndim == 2
    # Ensure the saturation value is already a numpy scalar
    saturation_value = data.dtype.type(saturation_value)
    block_sums = da.map_blocks(
        _saturation_counts,
        data,
        meta=np.array((), np.uint64),
        chunks=(data.chunks[0], (1,) * len(data.chunks[1])),
        saturation_value=saturation_value,
    )
    return da.sum(block_sums, axis=1)


@numba.njit(nogil=True)
def _packbits(data: np.ndarray, bits: int) -> np.ndarray:
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


def packbits(data: da.Array, bits: int) -> da.Array:
    """Pack integers into bytes.

    The least-significant `bits` bits of each integer in `data` is collected
    together in big-endian order, and returned as a sequence of bytes. The
    total number of bits must form a whole number of bytes.

    If the chunks in `data` are not be aligned on byte boundaries then a
    slower path is used.
    """
    assert data.ndim == 1
    if data.shape[0] * bits % BYTE_BITS:
        raise ValueError("Total number of bits is not a multiple of 8")
    if not all(c * bits % BYTE_BITS == 0 for c in data.chunks[0]):
        assert CHUNK_SIZE % BYTE_BITS == 0
        data = data.rechunk(CHUNK_SIZE)
    out_chunks = (tuple(c * bits // BYTE_BITS for c in data.chunks[0]),)
    return da.map_blocks(_packbits, data, dtype=np.uint8, chunks=out_chunks, bits=bits)


def sample(
    signals: Sequence[Signal],
    timestamp: int,
    period: int | None,
    sample_rate: float,
    sample_bits: int,
    out: xr.DataArray,
    out_saturated: xr.DataArray | None = None,
    saturation_group: int = 1,
    *,
    dither: bool | xr.DataArray = True,
    dither_seed: int | None = None,
) -> None:
    """Sample, quantise and pack a set of signals.

    The number of samples to generate is determined from the output array.

    Parameters
    ----------
    signals
        Signals to sample, one per polarisation
    timestamp
        Timestamp for the first element to return. The signal is rotated by
        this amount.
    period
        Number of samples after which to repeat. This must divide into the
        total number of samples to generate. If not specified, uses the
        total number of samples.
    sample_rate
        Passed to :meth:`Signal.sample`
    sample_bits
        Passed to :func:`quantise` and :func:`packbits`
    out
        Output array, with a dimension called ``pol`` (which must match the
        number of signals). The other dimensions are flattened.
    out_saturated
        Output array, with the same shape as ``out``, into which saturation
        counts are written.
    saturation_group
        Samples are taken in contiguous groups of this size and each element of
        out_saturated is a saturation count for one group. This must divide
        into the total number of samples.
    dither
        If true (default), add uniform random values in the range [-0.5, 0.5)
        after scaling to reduce artefacts. It may also be a :class:`xr.DataArray`
        with axes called ``pol`` (which must match the number of signals) and
        ``data`` (which must have length at least equal to ``period``).
    """
    n_pols = out.sizes["pol"]
    if len(signals) != n_pols:
        raise ValueError(f"Expected {n_pols} signals, received {len(signals)}")
    n = out.isel(pol=0).data.size * BYTE_BITS // sample_bits
    if period is None:
        period = n
    if n % period:
        raise ValueError(f"period {period} does not divide into total samples {n}")
    if n % saturation_group:
        raise ValueError(f"saturation_group (saturation_group) does not divide into total samples {n}")

    if dither is True:
        dither = make_dither(len(signals), period, entropy=dither_seed)
    elif dither is False:
        dither = xr.DataArray(da.zeros((n_pols, period), np.float32, chunks=CHUNK_SIZE), dims=["pol", "data"])
    else:
        if dither.sizes["pol"] != n_pols:
            raise ValueError(f"Expected {n_pols} dither signals, received {dither.sizes['pol']}")
        if dither.sizes["data"] < period:
            raise ValueError(f"Expected at least {period} dither samples, only found {dither.sizes['data']}")
        dither = dither.isel(data=np.s_[:period])

    in_arrays = []
    out_arrays = []
    for i, sig in enumerate(signals):
        data = sig.sample(period, sample_rate)
        if sig.terminal:
            sig_dither = da.zeros(period, np.float32, chunks=CHUNK_SIZE)
        else:
            sig_dither = dither.isel(pol=i).data
        data = quantise(data, sample_bits, sig_dither)
        data = da.roll(data, -timestamp)
        if period < n:
            data = da.tile(data, n // period)
        if out_saturated is not None:
            saturated = saturation_counts(data.reshape(-1, saturation_group), 2 ** (sample_bits - 1) - 1)
        data = packbits(data, sample_bits)
        in_arrays.append(data)
        out_arrays.append(out.isel(pol=i).data.ravel())
        if out_saturated is not None:
            in_arrays.append(saturated)
            out_arrays.append(out_saturated.isel(pol=i).data.ravel())
    # Compute all the pols together, so that common signals are only computed
    # once.
    da.store(in_arrays, out_arrays, lock=False)


class SignalService:
    """Compute signals in a separate process.

    The provided arrays must be backed by :class:`.SharedArray`, and each must
    have an xarray attribute called ``"shared_array"`` which holds the backing
    :class:`.SharedArray`.

    Parameters
    ----------
    arrays
        All the arrays that might be passed to :meth:`sample`.
    sample_bits
        Number of bits per sample for all queries.
    dither_seed
        Seed used to generate a fixed dither.
    """

    @dataclass
    class _Request:
        """Serialises a request from the main process to the service process."""

        signals: Sequence[Signal]
        timestamp: int
        period: int | None
        sample_rate: float
        out_idx: int  #: Index of the out array in the list of valid arrays
        out_saturated_idx: int | None  #: Index of the out saturated array in the list of valid arrays
        saturation_group: int

    @staticmethod
    def _run(
        array_schemas: list[dict],
        sample_bits: int,
        dither_seed: int | None,
        pipe: multiprocessing.connection.Connection,
    ) -> None:
        """Run the main service loop for the separate process.

        It receives a _Request on the pipe, and replies with either ``None``
        or an exception. When the main process wants to shut down, it will
        close the pipe.
        """
        # Avoid catching Ctrl-C meant for the parent
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))

        # Load the arrays by fetching the shared array out of the special attribute
        arrays = [
            xr.DataArray.from_dict({**schema, "data": schema["attrs"]["shared_array"].buffer})
            for schema in array_schemas
        ]

        # Generate and pre-compute the dither
        n_pols = arrays[0].sizes["pol"]
        n = arrays[0].isel(pol=0).data.size * BYTE_BITS // sample_bits
        dither = make_dither(n_pols, n, dither_seed)
        dither = dither.persist()  # Compute now and store results

        while True:
            try:
                req: SignalService._Request = pipe.recv()
            except EOFError:
                break  # Caller has shut down the pipe
            try:
                sample(
                    req.signals,
                    req.timestamp,
                    req.period,
                    req.sample_rate,
                    sample_bits,
                    arrays[req.out_idx],
                    out_saturated=arrays[req.out_saturated_idx] if req.out_saturated_idx is not None else None,
                    saturation_group=req.saturation_group,
                    dither=dither,
                )
            except Exception as exc:
                pipe.send(exc)
            else:
                pipe.send(None)

        # Not strictly necessary since the process is about to die anyway,
        # but might help prevent warnings about leaking file descriptors in
        # future.
        for array in arrays:
            array.attrs["shared_array"].close()

    def __init__(self, arrays: Sequence[xr.DataArray], sample_bits: int, dither_seed: int | None = None) -> None:
        self.arrays = arrays
        # These contain the `shared_array` attribute, which carries the
        # reference to the shared memory into the child process.
        array_schemas = [array.to_dict(data=False) for array in arrays]
        # The default "fork" method seems to cause problems with the unit
        # tests (NGC-637). Spawning is slower but ensures we share nothing
        # other than what we wish to share explicitly.
        ctx = multiprocessing.get_context("spawn")
        self._pipe, remote_pipe = ctx.Pipe()
        self._process = ctx.Process(
            target=self._run,
            args=(
                array_schemas,
                sample_bits,
                dither_seed,
                remote_pipe,
            ),
        )
        self._process.start()
        remote_pipe.close()  # Ensures the child holds the only reference

    async def stop(self) -> None:
        """Shut down the process."""
        self._pipe.close()
        await asyncio.get_running_loop().run_in_executor(None, self._process.join)

    def _make_request(self, request: "SignalService._Request") -> None:
        self._pipe.send(request)
        reply = self._pipe.recv()
        if reply is not None:
            raise reply

    def _array_index(self, out: xr.DataArray) -> int:
        for i, array in enumerate(self.arrays):
            # Object identity doesn't work well, I think because fetching one
            # xr.DataArray from a xr.DataSet creates a new object on the fly.
            # So we check if they're referencing the same memory in the same
            # way.
            if array.data.__array_interface__ == out.data.__array_interface__:
                return i
        raise ValueError("output was not registered with the constructor")

    async def sample(
        self,
        signals: Sequence[Signal],
        timestamp: int,
        period: int | None,
        sample_rate: float,
        out: xr.DataArray,
        out_saturated: xr.DataArray | None = None,
        saturation_group: int = 1,
    ) -> None:
        """Perform signal sampling in the remote process.

        `out` and `out_saturated` must each be one of the arrays passed to the
        constructor. Only the first `n` samples will be populated (and this
        will be taken as the period).
        """
        out_idx = self._array_index(out)
        out_saturated_idx = self._array_index(out_saturated) if out_saturated is not None else None
        loop = asyncio.get_running_loop()
        req = SignalService._Request(
            signals, timestamp, period, sample_rate, out_idx, out_saturated_idx, saturation_group
        )
        await loop.run_in_executor(None, self._make_request, req)

    async def __aenter__(self) -> "SignalService":
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.stop()
