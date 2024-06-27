# noqa: D104

################################################################################
# Copyright (c) 2020-2024, National Research Foundation (SARAO)
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

import copy
from typing import Mapping, Self

import numpy as np
import prometheus_client
from numpy.typing import NDArray

from katgpucbf import BYTE_BITS, DIG_SAMPLE_BITS


class PromDiff:
    """Collects Prometheus metrics before and after test code, and provides differences.

    Typical usage is::

        with PromDiff(namespace=METRIC_NAMESPACE) as prom_diff:
            ...  # Do stuff that increments counters
        prom_diff.diff(name, labels)

    In some cases one may wish to make many queries with the same labels.
    In this case, default labels can be passed to the constructor::

        with PromDiff(namespace=METRIC_NAMESPACE, labels={"stream": stream_name}) as prom_diff:
            ...  # Do stuff that increments counters
        # Metrics `name1` and `name2` both have the same label
        prom_diff.diff(name1)
        prom_diff.diff(name2)

    Alternatively, one can create a new :class:`PromDiff` that holds the same
    data but has additional default labels:

        with PromDiff(namespace=METRIC_NAMESPACE) as prom_diff:
            ...  # Do stuff that increments counters
        labelled_diff = diff.with_labels(labels)
        labelled_diff.diff(name1)
        labelled_diff.diff(name2)

    Labels are cumulative: more labels can be passed to :meth:`value` or
    :meth:`diff` and they augment the default labels.

    Parameters
    ----------
    registry
        Prometheus metric registry
    namespace
        Namespace to prepend to metric names
    labels
        Default labels to use in queries
    """

    def __init__(
        self,
        *,
        registry: prometheus_client.CollectorRegistry = prometheus_client.REGISTRY,
        namespace: str | None = None,
        labels: Mapping[str, str] | None = None,
    ) -> None:
        self._registry = registry
        self._before: list[prometheus_client.samples.Sample] = []
        self._after: list[prometheus_client.samples.Sample] = []
        self._prefix = namespace + "_" if namespace is not None else ""
        self._labels = dict(labels) if labels is not None else {}

    def __enter__(self) -> Self:
        self._before = [s for metric in self._registry.collect() for s in metric.samples]
        return self

    def __exit__(self, *args) -> None:
        self._after = [s for metric in self._registry.collect() for s in metric.samples]

    def _get_labels(self, labels: Mapping[str, str] | None) -> dict[str, str]:
        """Compute effective labels by combining default and per-call labels."""
        if labels is None:
            return self._labels
        else:
            return {**self._labels, **labels}

    def _get_value(
        self, samples: list[prometheus_client.samples.Sample], name: str, labels: dict[str, str]
    ) -> float | None:
        for s in samples:
            if s.name == self._prefix + name and s.labels == labels:
                return s.value
        return None

    def value(self, name: str, labels: Mapping[str, str] | None = None) -> float:
        """Return the value of the metric at the end of the context manager protocol.

        If it is not found, raises an :exc:`AssertionError`.
        """
        value = self._get_value(self._after, name, self._get_labels(labels))
        assert value is not None, f"Metric {name}{labels} does not exist"
        return value

    def diff(self, name: str, labels: Mapping[str, str] | None = None) -> float:
        """Return the increase in the metric during the context manager protocol.

        If the metric did not exist at the start and end, raises an
        :exc:`AssertionError`.
        """
        effective_labels = self._get_labels(labels)
        before = self._get_value(self._before, name, effective_labels)
        after = self._get_value(self._after, name, effective_labels)
        assert before is not None, f"Metric {name}{labels} did not exist at start"
        assert after is not None, f"Metric {name}{labels} did not exist at start"
        return after - before

    def with_labels(self, labels: Mapping[str, str] | None = None) -> Self:
        """Create a new instance holding the same data but with extra default labels.

        The `labels` are used to update the current default labels rather than
        completely replacing them.
        """
        ret = copy.copy(self)  # Note: shallow copy
        ret._labels = self._get_labels(labels)
        return ret


def unpackbits(data: NDArray[np.uint8], sample_bits: int = DIG_SAMPLE_BITS) -> np.ndarray:
    """Unpack a bit-packed array of signed big-endian integers.

    Typically `sample_bits` will be DIG_SAMPLE_BITS, but can be up to 64. The
    dtype of the result depends on the number of bits.
    """
    dtype: np.dtype
    if sample_bits <= 16:
        dtype = np.dtype(np.int16)
    elif sample_bits <= 32:
        dtype = np.dtype(np.int32)
    elif sample_bits <= 64:
        dtype = np.dtype(np.int64)
    else:
        raise ValueError("sample_bits is too large")
    if data.size * BYTE_BITS % sample_bits:
        raise ValueError("data does not contain a whole number of samples")
    bits = np.unpackbits(data).reshape(-1, sample_bits)
    # Replicate the high (sign) bit
    extra = np.tile(bits[:, 0:1], (1, 8 * dtype.itemsize - sample_bits))
    combined = np.hstack([extra, bits])
    packed = np.packbits(combined)
    return packed.view(dtype.newbyteorder(">")).astype(dtype)


def packbits(data: np.ndarray, sample_bits: int = DIG_SAMPLE_BITS) -> NDArray[np.uint8]:
    """Pack a bit-packed array of signed big-endian integers.

    This is the inverse of :meth:`unpackbits`, but it also handles
    multidimensional inputs, packing along the last dimension.
    """
    assert data.dtype.kind == "i"
    if sample_bits > 8 * data.dtype.itemsize:
        raise ValueError("sample_bits is too large")
    if sample_bits * data.shape[-1] % BYTE_BITS:
        raise ValueError("packing the last axis of data does not produce a whole number of bytes")
    # Force to big endian. Also force C-contiguous since .view requires it
    data = np.require(data, dtype=data.dtype.newbyteorder(">"), requirements="C")
    bits = np.unpackbits(data.view(np.uint8), axis=-1).reshape(data.shape + (8 * data.dtype.itemsize,))
    # Strip off the unused bits and flatten out the extra axis
    bits = bits[..., -sample_bits:]
    bits = bits.reshape(bits.shape[:-2] + (-1,))
    return np.packbits(bits, axis=-1)


def unpack_complex(data: np.ndarray) -> np.ndarray:
    """Unpack array of Gaussian integers to complex dtype.

    The dtype of `data` must be a type returned by :func:`.gaussian_dtype`.
    """
    if data.dtype.itemsize == 1:
        # It's 4-bit packed. We assume that >> will sign extend on signed types
        real = data.view(np.int8) >> 4
        imag = data.view(np.int8) << 4 >> 4
    else:
        real = data["real"]
        imag = data["imag"]
    return real.astype(np.float32) + 1j * imag.astype(np.float32)
