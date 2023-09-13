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
from typing import Awaitable, Callable, Generic, Sequence, TypeVar

import numpy as np
from numpy.typing import NDArray

_T = TypeVar("_T", covariant=True)


def _entropy(a: NDArray[float], axis: int | tuple[int, ...] | None = None) -> np.float_:
    return np.sum(-a * np.log(a), axis=axis)


@dataclass
class NoisySearchResult(Generic[_T]):
    position: int  #: The position of the element before which the new element is to be inserted
    queries: int  #: Number of queries made
    confidence: float  #: Probability that the answer is correct


async def noisy_search(
    items: Sequence[_T], noise: float | NDArray[float], tolerance: float, compare: Callable[[_T], Awaitable[bool]]
) -> NoisySearchResult[_T]:
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
        (with a uniform prior)
    compare
        Comparison function. It is passed an existing element, and should
        mostly return true if the new element comes before it and false
        otherwise.

    Returns
    -------
    """
    n = len(items)
    if np.isscalar(noise):
        assert 0 <= noise < 0.5
        yes_scale = np.tri(n, n + 1) * (1 - 2 * noise) + noise
    else:
        yes_scale = noise
    no_scale = 1 - yes_scale
    assert 0 < tolerance < 0.5
    a = np.full(n + 1, 1 / (n + 1))  # Uniform prior
    entropy = np.empty(n)
    queries = 0
    while True:
        maxi = np.argmax(a)
        if a[maxi] >= 1 - tolerance:
            return NoisySearchResult(position=maxi, queries=queries, confidence=float(a[maxi]))

        # Determine query point that gives the lowest expected entropy afterwards
        csum = np.cumsum(a)[:n]
        yes = a[np.newaxis, :] * yes_scale
        yes /= np.sum(yes, axis=1, keepdims=True)
        no = a[np.newaxis, :] * no_scale
        no /= np.sum(no, axis=1, keepdims=True)
        entropy = _entropy(yes, axis=1) * csum + _entropy(no, axis=1) * (1 - csum)
        i = np.argmin(entropy)
        queries += 1
        if await compare(items[i]):
            a *= yes_scale[i]
        else:
            a *= no_scale[i]
        # Normalise
        a /= np.sum(a)
