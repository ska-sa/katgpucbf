################################################################################
# Copyright (c) 2023, National Research Foundation (SARAO)
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

from dataclasses import dataclass
from typing import Awaitable, Callable, Sequence, TypeVar

import numpy as np
from numpy.typing import ArrayLike

_T = TypeVar("_T", covariant=True)


def _entropy(a: ArrayLike, axis: int | tuple[int, ...] | None = None) -> np.floating:
    array = np.asarray(a)
    return np.sum(-array * np.log(array), axis=axis)


@dataclass
class NoisySearchResult:
    low: int  #: Lower bound for the new element (as an index into the original array)
    high: int  #: Upper bound for the new element (as an index into the original array)
    comparisons: int  #: Number of comparisons made
    confidence: float  #: Probability that the interval [low, high) contains the new element


async def noisy_search(
    items: Sequence[_T],
    noise: ArrayLike,
    tolerance: float,
    compare: Callable[[_T], Awaitable[bool]],
    *,
    max_interval: int = 1,
    max_comparisons: int | None = None
) -> NoisySearchResult:
    """
    Perform a binary search with a noisy comparison function.

    It uses a series of comparisons to determine the position at which a new
    element should be inserted into a sorted list. The comparisons may
    probabilistically return the incorrect result.

    The implementation is loosely based on [Ben-Or08]_. It is a much simplified
    version that just uses the basic Bayesian update, without any of the
    recursive tricks the paper uses to bound the theoretical expected running
    time.

    .. [Ben-Or08] Ben Or and Hassidim. The Bayesian Learner is Optimal for Noisy
       Binary Search (and Pretty Good for Quantum as Well). 49th Symposium on
       Foundations of Computer Science, pp. 221-230. 2008.

    Parameters
    ----------
    items
        Existing elements.
    noise
        Probability that comparison will return the incorrect result.
        Alternatively it may be a matrix with `n` rows and `n + 1` columns,
        where `n` is the length of `items`. The entry in row `i`, column `j`
        is the probability that passing element `i` to `compare` will return
        true if the correct position is immediately before element `j`. It
        should thus be greater than 0.5 when `i` >= `j` and less than 0.5
        otherwise.
    tolerance
        Maximum probability that this function may return an incorrect result
        (with a uniform prior).
    compare
        Comparison function. It is passed an existing element, and should
        mostly return true if the new element comes before it and false
        otherwise.
    max_interval
        Maximum width of the returned interval (in indices)
    max_comparisons
        A limit on the number of comparisons. If this number of comparisons is
        reached, a confidence interval wider than `max_interval` may be
        returned.

    Returns
    -------
    """
    n = len(items)
    if np.isscalar(noise):
        real_noise: float
        # np.isscalar is annotated with TypeGuard and it loses all the
        # original type information, so mypy
        real_noise = float(noise)  # type: ignore
        assert 0.0 <= real_noise < 0.5
        yes_scale = np.tri(n, n + 1) * (1 - 2 * real_noise) + real_noise
    else:
        yes_scale = np.asarray(noise)
    no_scale = 1 - yes_scale
    assert 0 < tolerance < 0.5
    a = np.full(n + 1, 1 / (n + 1))  # Uniform prior
    entropy = np.empty(n)
    comparisons = 0
    while True:
        # Determine the current confidence interval
        csum = np.cumsum(a)
        low = int(np.searchsorted(csum, 0.5 * tolerance)) - 1
        high = int(np.searchsorted(csum, 1 - 0.5 * tolerance))
        if high - low <= max_interval or (max_comparisons is not None and comparisons >= max_comparisons):
            confidence = a[high] - (a[low] if low >= 0 else 0.0)
            return NoisySearchResult(low=low, high=high, comparisons=comparisons, confidence=float(confidence))

        # Determine query point that gives the lowest expected entropy afterwards
        yes = a[np.newaxis, :] * yes_scale
        yes /= np.sum(yes, axis=1, keepdims=True)
        no = a[np.newaxis, :] * no_scale
        no /= np.sum(no, axis=1, keepdims=True)
        entropy = _entropy(yes, axis=1) * csum[:n] + _entropy(no, axis=1) * (1 - csum[:n])
        i = int(np.argmin(entropy))
        comparisons += 1
        if await compare(items[i]):
            a *= yes_scale[i]
        else:
            a *= no_scale[i]
        # Normalise
        a /= np.sum(a)
