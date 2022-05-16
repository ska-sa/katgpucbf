"""Synthesis of simulated signals."""

import asyncio
import logging
import multiprocessing.connection
import operator
import os
import signal
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import dask.array as da
import numba
import numpy as np
import pyparsing as pp
import xarray as xr

from .. import BYTE_BITS

logger = logging.getLogger(__name__)
#: Dask chunk size for sampling signals (must be a multiple of 8)
CHUNK_SIZE = 2**20


class Signal(ABC):
    """Abstract base class for signals.

    An instance is simply a real-valued function of time, for a single
    polarisation.
    """

    def _sample_helper(self, n: int, chunk_data: da.Array, sample_chunk: Callable, **kwargs) -> da.Array:
        """Help implement :meth:`sample` in subclasses.

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

    def sample(self, n: int, sample_rate: float) -> da.Array:  # noqa: D102
        return self.combine(self.a.sample(n, sample_rate), self.b.sample(n, sample_rate))

    def __str__(self) -> str:
        return f"({self.a} {self.op_name} {self.b})"


@dataclass
class CW(Signal):
    """Continuous wave.

    To make the resulting signal periodic, the frequency is adjusted during
    sampling so that the sampled result can be looped.
    """

    amplitude: float
    frequency: float

    @staticmethod
    def _sample_chunk(offset: np.int64, *, amplitude: np.float32, chunk_size: int, scale: float) -> np.ndarray:
        """Compute :math:`np.cos(np.arange(offset, n + offset) * scale) * amplitude` efficiently.

        The return value is single precision.
        """
        # Compute the complex exponential. Because it is being regularly
        # sampled, it is possible to do this efficiently by repeated
        # doubling. This also makes it possible to keep most of the
        # computation in single precision without losing much precision
        # (experimentally the results seem to be off by less than 1e-6).
        out = np.empty(chunk_size, np.complex64)
        out[0] = np.exp(offset * scale * 1j) * amplitude
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

    def sample(self, n: int, sample_rate: float) -> da.Array:  # noqa: D102
        # Round target frequency to fit an integer number of waves into signal_heaps
        waves = max(1, round(n * self.frequency / sample_rate))
        frequency = waves * sample_rate / n
        logger.info(f"Rounded tone frequency to {frequency} Hz")
        scale = self.frequency / sample_rate * 2 * np.pi

        # Index of the first element of each chunk
        offsets = da.arange(0, n, CHUNK_SIZE, chunks=1, dtype=np.int64)
        return self._sample_helper(
            n,
            offsets,
            self._sample_chunk,
            amplitude=np.float32(self.amplitude),
            scale=scale,
        )

    def __str__(self) -> str:
        return f"cw({self.amplitude}, {self.frequency})"


# mypy override is due to https://github.com/python/mypy/issues/5374
@dataclass  # type: ignore
class Random(Signal):
    """Base class for randomly-generated signals.

    This base class is only suitable when the samples at different times
    are independent. The derived class must implement :meth:`_sample_chunk`.
    """

    entropy: int  #: entropy used to populate a :class:`np.random.SeedSequence`

    def __init__(self, entropy: Optional[int] = None) -> None:
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
        n_chunks = (n + CHUNK_SIZE - 1) // CHUNK_SIZE
        seed_seqs = np.random.SeedSequence(self.entropy).spawn(n_chunks)
        # Chunk size of 1 so that we can map each SeedSequence to an output chunk
        seed_seqs_dask = da.from_array(np.array(seed_seqs, dtype=object), 1)
        return self._sample_helper(n, seed_seqs_dask, self._sample_chunk)


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

    std: float  #: standard deviation

    def __init__(self, std: float, entropy: Optional[int] = None) -> None:
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


class Dither(Random):
    """Random signal to add as part of quantisation, for dithering.

    Unlike other signals, this is scaled such that the quantisation bin size is
    1.0.

    The implementation currently uses a uniform distribution, but that is
    subject to change.
    """

    def _sample_chunk(self, seed_seq: Sequence[np.random.SeedSequence], *, chunk_size: int) -> np.ndarray:
        # The RNG is initialised every time this is called so that it will
        # produce the same results.
        rng = np.random.default_rng(seed_seq[0])
        return rng.random(size=chunk_size, dtype=np.float32) - np.float32(0.5)


def _apply_operator(s: str, loc: int, tokens: pp.ParseResults) -> Signal:
    assert len(tokens) == 1
    tokens = tokens[0]  # infix_operator passes the expression with an extra nesting level
    op_map = {"*": operator.mul, "+": operator.add, "-": operator.sub}
    result = tokens[0]
    for i in range(1, len(tokens), 2):
        result = op_map[tokens[i]](result, tokens[i + 1])
    return result


def parse_signals(prog: str) -> List[Signal]:
    """Generate a set of signals from a domain-specific language.

    The domain-specific language consists of statements terminated by
    semi-colons. Two types of statements are available:

    1. Assignments have the form :samp:`{var} = {expr}`. The :samp:`{var}`
       must be a valid ASCII Python identifier. Expressions are described
       later.

    2. Return values consist solely of an expression.

    An expression may consist of function calls, parentheses, the operators
    ``+``, ``-`` and ``*``, and previously-defined variables. The following
    functions are available (parameters must be integer or floating-point
    literals).

    cw(amplitude, frequency)
        See :class:`CW`.

    wgn(std [, entropy])
        See :class:`WGN`.
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
    # See https://pyparsing-docs.readthedocs.io/en/latest/HowToUsePyparsing.html#expression-subclasses
    # for an explanation of + versus - in these rules (it helps give more
    # useful errors). I've been conservative with the use of - i.e., there
    # may be some +'s that can still be changed to -'s.
    cw = pp.Keyword("cw") + lpar - real + comma - real + rpar
    cw.set_parse_action(lambda s, loc, tokens: CW(tokens[1], tokens[2]))
    wgn = pp.Keyword("wgn") + lpar - real + pp.Opt(comma - integer("entropy")) + rpar
    wgn.set_parse_action(lambda s, loc, tokens: WGN(tokens[1], tokens.get("entropy")))
    variable_expr = variable.copy()
    variable_expr.set_parse_action(get_variable)

    atom = cw | wgn | variable_expr
    expr = pp.infix_notation(
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


def quantise(data: da.Array, bits: int, dither: bool = True) -> da.Array:
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
    scaled = data * scale
    if dither:
        # TODO: should it be seeded in some controllable way?
        scaled += Dither().sample(scaled.size, 0)
    return da.rint(da.clip(scaled, -scale, scale)).astype(np.int32)


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


def packbits(data: da.Array, bits: int) -> da.Array:
    """Pack integers into bytes.

    The least-significant `bits` bits of each integer in `data` is collected
    together in big-endian order, and returned as a sequence of bytes. The
    total number of bits must form a whole number of bytes.

    The chunks in `data` must be aligned to multiples of 8.
    """
    assert data.ndim == 1
    if not all(c % BYTE_BITS == 0 for c in data.chunks[0]):
        raise ValueError("Chunks are not aligned to byte boundaries")
    out_chunks = (tuple(c * bits // BYTE_BITS for c in data.chunks[0]),)
    return da.map_blocks(_packbits, data, dtype=np.uint8, chunks=out_chunks, bits=bits)


def sample(signals: Sequence[Signal], timestamp: int, sample_rate: float, sample_bits: int, out: xr.DataArray) -> None:
    """Sample, quantise and pack a set of signals.

    The number of samples to generate is determined from the output array.

    Parameters
    ----------
    signals
        Signals to sample, one per polarisation
    timestamp
        Timestamp for the first element to return. The signal is rotated by
        this amount.
    sample_rate
        Passed to :meth:`Signal.sample`
    sample_bits
        Passed to :func:`quantise` and :func:`packbits`
    out
        Output array, with a dimension called ``pol`` (which must match the
        number of signals). The other dimensions are flattened.
    """
    if len(signals) != out.sizes["pol"]:
        raise ValueError(f"Expected {out.sizes['pol']} signals, received {len(signals)}")
    n = out.isel(pol=0).data.size * BYTE_BITS // sample_bits
    sampled = []
    for sig in signals:
        data = sig.sample(n, sample_rate)
        data = quantise(data, sample_bits)
        data = da.roll(data, -timestamp)
        data = packbits(data, sample_bits)
        sampled.append(data)
    # Compute all the pols together, so that common signals are only computed
    # once.
    da.store(sampled, [out.isel(pol=i).data.ravel() for i in range(len(signals))], lock=False)


class SignalService:
    """Compute signals in a separate process.

    This only works with the fork model of multiprocessing, as it depends on
    inheriting the arrays across the :meth:`os.fork`. Additionally, the arrays
    must be allocated in such a way that they are shared rather than
    copy-on-write, for example, using :mod:`multiprocessing.sharedctypes`.

    Parameters
    ----------
    arrays
        All the arrays that might be passed to :meth:`sample`.
    """

    @dataclass
    class _Request:
        """Serialises a request from the main process to the service process."""

        signals: Sequence[Signal]
        timestamp: int
        sample_rate: float
        sample_bits: int
        out_idx: int  #: Index of the array in the list of valid arrays

    @staticmethod
    def _run(
        arrays: Sequence[xr.DataArray],
        pipe: multiprocessing.connection.Connection,
        parent_pipe: multiprocessing.connection.Connection,
    ) -> None:
        """Run the main service loop for the separate process.

        It receives a _Request on the pipe, and replies with either ``None``
        or an exception. When the main process wants to shut down, it will
        close the pipe.
        """
        # Avoid catching Ctrl-C meant for the parent
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        # This runs in the child, so doesn't need to hold a handle to the
        # parent close. Closing it ensures that we get the EOFError when
        # the parent closes the pipe and we try to read.
        parent_pipe.close()
        os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
        while True:
            try:
                req: SignalService._Request = pipe.recv()
            except EOFError:
                return  # Caller has shut down the pipe
            try:
                sample(req.signals, req.timestamp, req.sample_rate, req.sample_bits, arrays[req.out_idx])
            except Exception as exc:
                pipe.send(exc)
            else:
                pipe.send(None)

    def __init__(self, arrays: Sequence[xr.DataArray]) -> None:
        self.arrays = arrays
        self._pipe, remote_pipe = multiprocessing.Pipe()
        self._process = multiprocessing.Process(target=self._run, args=(arrays, remote_pipe, self._pipe))
        self._process.start()

    async def stop(self) -> None:
        """Shut down the process."""
        self._pipe.close()
        await asyncio.get_running_loop().run_in_executor(None, self._process.join)

    def _make_request(self, request: "SignalService._Request") -> None:
        self._pipe.send(request)
        reply = self._pipe.recv()
        if reply is not None:
            raise reply

    async def sample(
        self, signals: Sequence[Signal], timestamp: int, sample_rate: float, sample_bits: int, out: xr.DataArray
    ) -> None:
        """Perform signal sampling in the remote process.

        `out` must be one of the arrays passed to the constructor.
        """
        for i, array in enumerate(self.arrays):
            # Object identity doesn't work well, I think because fetching one
            # xr.DataArray from a xr.DataSet creates a new object on the fly.
            # So we check if they're referencing the same memory in the same
            # way.
            if array.data.__array_interface__ == out.data.__array_interface__:
                out_idx = i
                break
        else:
            raise ValueError("output was not registered with the constructor")
        loop = asyncio.get_running_loop()
        req = SignalService._Request(signals, timestamp, sample_rate, sample_bits, out_idx)
        await loop.run_in_executor(None, self._make_request, req)

    async def __aenter__(self) -> "SignalService":
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.stop()
