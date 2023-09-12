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

_T = TypeVar("_T", covariant=True)


@dataclass
class NoisySearchResult(Generic[_T]):
    position: int  #: The position of the element before which the new element is to be inserted
    queries: int  #: Number of queries made
    confidence: float  #: Probability that the answer is correct


async def noisy_search(
    items: Sequence[_T], noise: float, tolerance: float, compare: Callable[[_T], Awaitable[bool]]
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
        Existing elements. It is particularly efficient to pass an instance
        of :class:`range`.
    noise
        Probability that comparison will return the incorrect result
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
    assert 0 <= noise < 0.5
    assert 0 < tolerance < 0.5
    n = len(items)
    a = [1.0 / (n + 1)] * (n + 1)  # Uniform prior
    queries = 0
    while True:
        for i, v in enumerate(a):
            if v >= 1 - tolerance:
                return NoisySearchResult(position=i, queries=queries, confidence=v)

        t = 0.0
        prev_t = 0.0
        for i in range(n + 1):
            prev_t = t
            t += a[i]
            if t > 0.5:
                break
        # The paper has an asymmetric condition, but we pick the closest
        # partition point.
        if i > 0 and 0.5 - prev_t < t - 0.5:
            i -= 1
            t = prev_t
        queries += 1
        if await compare(items[i]):
            norm = (1 - noise) * t + noise * (1 - t)
            scale_left = (1 - noise) / norm
            scale_right = noise / norm
        else:
            norm = noise * t + (1 - noise) * (1 - t)
            scale_left = noise / norm
            scale_right = (1 - noise) / norm
        for j in range(i + 1):
            a[j] *= scale_left
        for j in range(i + 1, n + 1):
            a[j] *= scale_right
