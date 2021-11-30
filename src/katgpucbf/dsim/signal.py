"""Synthesis of simulated signals."""

import logging
import operator
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List

import numba
import numpy as np
import pyparsing as pp
from numpy.typing import ArrayLike

from .. import BYTE_BITS

logger = logging.getLogger(__name__)


class Signal(ABC):
    """Abstract base class for signals.

    An instance is simply a real-valued function of time, for a single
    polarisation.
    """

    @abstractmethod
    def sample(self, timestamp: int, n: int, sample_rate: float) -> np.ndarray:
        """Sample the signal at regular intervals.

        The returned values should be scaled to the range (-1, 1).

        Parameters
        ----------
        timestamp
            Time (in samples since the sync epoch) of the first returned sample.
        n
            Number of samples to generate
        sample_rate
            Frequency of samples (Hz)

        Returns
        -------
        samples
            Array of samples, float32.
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


@dataclass(frozen=True)
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
    combine: Callable[[np.ndarray, np.ndarray], np.ndarray]
    op_name: str

    def sample(self, timestamp: int, n: int, sample_rate: float) -> np.ndarray:  # noqa: D102
        # The ignore is due to https://github.com/python/mypy/issues/10711
        return self.combine(  # type: ignore
            self.a.sample(timestamp, n, sample_rate), self.b.sample(timestamp, n, sample_rate)
        )

    def __str__(self) -> str:
        return f"({self.a} {self.op_name} {self.b})"


@dataclass(frozen=True)
class CW(Signal):
    """Continuous wave."""

    amplitude: float
    frequency: float

    def sample(self, timestamp: int, n: int, sample_rate: float) -> np.ndarray:  # noqa: D102
        # Round target frequency to fit an integer number of waves into signal_heaps
        waves = max(1, round(n * self.frequency / sample_rate))
        frequency = waves * sample_rate / n
        logger.info(f"Rounded tone frequency to {frequency} Hz")
        # Compute the complex exponential. Because it is being regularly
        # sampled, it is possible to do this efficiently by repeated
        # doubling. This also makes it possible to keep most of the
        # computation in single precision without losing much precision
        # (experimentally the results seem to be off by less than 1e-6).
        scale = self.frequency / sample_rate * 2 * np.pi
        cplex = np.empty(n, np.complex64)
        cplex[0] = np.exp(timestamp * scale * 1j) * self.amplitude
        valid = 1
        while valid < n:
            # Rotate the segment [0, valid) by valid steps, giving the segment
            # [valid, 2 * value). It's slightly complicated to handle the case
            # where we have to truncate to n.
            add = min(valid, n - valid)
            rot = np.exp(valid * scale * 1j).astype(np.complex64)
            np.multiply(cplex[0:add], rot, cplex[valid : valid + add])
            valid += add
        # The result is copied to make it compact (otherwise the imaginary
        # parts don't get freed).
        return cplex.real.copy()

    def __str__(self) -> str:
        return f"cw({self.amplitude}, {self.frequency})"


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
    functions are available (parameters must be floating-point literals).

    cw(amplitude, frequency)
        See :class:`CW`.
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
    cw = pp.Keyword("cw") + lpar - real + comma - real + rpar
    cw.set_parse_action(lambda s, loc, tokens: CW(tokens[1], tokens[2]))
    variable_expr = variable.copy()
    variable_expr.set_parse_action(get_variable)

    atom = cw | variable_expr
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
