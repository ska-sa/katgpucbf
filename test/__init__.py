# noqa: D104

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

from typing import Any

import aiokatcp
import numpy as np
import prometheus_client
from numpy.typing import NDArray

from katgpucbf import BYTE_BITS, DIG_SAMPLE_BITS


class PromDiff:
    """Collects Prometheus metrics before and after test code, and provides differences.

    Typical usage is::

        with PromDiff(namespace=METRIC_NAMESPACE) as diff:
            ...  # Do stuff that increments counters
        diff.get_sample_diff(name, labels)

    Parameters
    ----------
    registry
        Prometheus metric registry
    namespace
        Namespace to prepend to metric names
    """

    def __init__(
        self,
        *,
        registry: prometheus_client.CollectorRegistry = prometheus_client.REGISTRY,
        namespace: str | None = None,
    ) -> None:
        self._registry = registry
        self._before: list[prometheus_client.samples.Sample] = []
        self._after: list[prometheus_client.samples.Sample] = []
        self._prefix = namespace + "_" if namespace is not None else ""

    def __enter__(self) -> "PromDiff":
        self._before = [s for metric in self._registry.collect() for s in metric.samples]
        return self

    def __exit__(self, *args) -> None:
        self._after = [s for metric in self._registry.collect() for s in metric.samples]

    def _get_value(
        self, samples: list[prometheus_client.samples.Sample], name: str, labels: dict[str, str]
    ) -> float | None:
        for s in samples:
            if s.name == self._prefix + name and s.labels == labels:
                return s.value
        return None

    def get_sample_value(self, name: str, labels: dict[str, str] | None = None) -> float | None:
        """Return the value of the metric at the end of the context manager protocol.

        If it is not found, returns ``None``.
        """
        if labels is None:
            labels = {}
        return self._get_value(self._after, name, labels)

    def get_sample_diff(self, name: str, labels: dict[str, str] | None = None) -> float | None:
        """Return the increase in the metric during the context manager protocol.

        If it is not found, returns ``None``. If it was not found on entry,
        raises an exception.
        """
        if labels is None:
            labels = {}
        before = self._get_value(self._before, name, labels)
        after = self._get_value(self._after, name, labels)
        if before is not None and after is not None:
            return after - before
        elif after is not None:
            raise ValueError(f"Metric {name}{labels} did not exist at start")
        else:
            return after


async def get_sensor(client: aiokatcp.Client, name: str) -> Any:
    """Get the value of a sensor.

    .. todo:

       This should probably be implemented in aiokatcp.
    """
    # This is not a complete list of sensor types. Extend as necessary.
    sensor_types = {
        b"integer": int,
        b"float": float,
        b"boolean": bool,
        b"discrete": str,  # Gets the string name of the enum
        b"string": str,  # Allows passing through arbitrary values even if not UTF-8
    }

    _reply, informs = await client.request("sensor-list", name)
    assert len(informs) == 1
    sensor_type = sensor_types.get(informs[0].arguments[3], bytes)
    _reply, informs = await client.request("sensor-value", name)
    assert len(informs) == 1
    assert informs[0].arguments[3] in {b"nominal", b"warn", b"error"}
    return aiokatcp.decode(sensor_type, informs[0].arguments[4])


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
