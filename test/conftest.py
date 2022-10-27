################################################################################
# Copyright (c) 2020-2022, National Research Foundation (SARAO)
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

"""Common test plugin for all katgpucbf tests.

It adds a new pytest mark. Its use is best demonstrated by example:

.. code:: python

    @pytest.mark.combinations(
        "a, b",
        [1, 2, 3],
        ["cat", "dog", "meerkat", "possum"]
    )
    def my_test(a, b):
        ...

This will run :func:`!my_test` multiple times, with `a` spanning the values
``[1, 2, 3]`` and `b` spanning the values "cat", "dog", "meerkat", and
"possum". By default, it will only run 4 out of the 12 possible combinations
(enough to ensure that each individual value gets tested). By passing a
command-line option :option:`!--all-combinations` to pytest one can instead
run all 12 possible combinations.

Additional, a ``filter`` kwarg can be passed with a predicate function that can
decide whether any particular combination should be considered for testing. It
is given a dictionary mapping names to values, and returns true if that
combination is a candidate.
"""

import itertools
from dataclasses import dataclass
from typing import Any

import pytest
import spead2

pytest_plugins = ["katsdpsigproc.pytest_plugin"]


def pytest_configure(config) -> None:
    """Register custom markers."""
    config.addinivalue_line("markers", "combinations(names, *values): test combinations of values")


def pytest_addoption(parser) -> None:
    """Register new command-line options."""
    group = parser.getgroup("combinations")
    group.addoption("--all-combinations", action="store_true", help="Test the full Cartesian product of parameters")


@dataclass
class _CombinationsCandidate:
    indices: tuple[int, ...]  # Index into the list of options for each position
    values: tuple  # Actual value, organised by position
    by_name: dict[str, Any]  # Lookup by argument name (for filtering)


def pytest_generate_tests(metafunc) -> None:
    """Apply "combinations" marker."""
    all_combinations = metafunc.config.option.all_combinations
    for marker in metafunc.definition.iter_markers("combinations"):
        if isinstance(marker.args[0], (tuple, list)):
            names = list(marker.args[0])
        else:
            names = [name.strip() for name in marker.args[0].split(",") if name.strip()]
        values = marker.args[1:]
        if len(names) != len(values):
            pytest.fail(
                f"{metafunc.definition.nodeid}: "
                f'in "combinations" the number of names ({len(names)}):\n'
                f"  {names}\n"
                f"must be equal to the number of values ({len(values)}):\n"
                f"  {values}",
                pytrace=False,
            )
        if not names:
            continue  # Nothing to do if there are zero names
        candidates = []
        filter = marker.kwargs.get("filter", lambda combo: True)
        for indices in itertools.product(*(range(len(value_list)) for value_list in values)):
            candidate = _CombinationsCandidate(
                indices=indices,
                values=tuple(value_list[i] for value_list, i in zip(values, indices)),
                by_name={name: value_list[i] for name, value_list, i in zip(names, values, indices)},
            )
            if filter(candidate.by_name):
                candidates.append(candidate)
        if all_combinations:
            combos = candidates
        else:
            # Repeatly take the candidate with the least coverage, until every
            # individual value is tested at least once. Tie-breaking is
            # selected to favour the last elements: if the lists contain tests
            # of increasing complexity, then this ensures that most complex
            # combination gets tested.
            cover = [[0] * len(value_list) for value_list in values]
            combos = []
            while any(0 in c for c in cover):
                best: _CombinationsCandidate | None = None
                best_score = 0
                for candidate in candidates:
                    score = sum(c[idx] for c, idx in zip(cover, candidate.indices))
                    if best is None or score <= best_score:
                        best = candidate
                        best_score = score
                if not best:
                    raise RuntimeError("Filter is too strict: not all values can be tested")
                combos.append(best)
                candidates.remove(best)
                for c, idx in zip(cover, best.indices):
                    c[idx] += 1
            # First combo will have last element of each list. Make that the
            # last test rather than the first.
            combos.reverse()
        metafunc.parametrize(names, [combo.values for combo in combos])


@pytest.fixture
def mock_recv_streams(mocker, n_src_streams: int) -> list[spead2.InprocQueue]:
    """Mock out :func:`katgpucbf.recv.add_reader` to use in-process queues.

    Returns
    -------
    queues
        A list of in-process queue to use for sending data. The number of queues
        in the list is determined by ``n_src_streams``.
    """
    queues = [spead2.InprocQueue() for _ in range(n_src_streams)]
    queue_iter = iter(queues)  # Each call to add_reader gets the next queue

    def add_reader(
        stream: spead2.recv.ChunkRingStream,
        *,
        src: str | list[tuple[str, int]],
        interface: str | None,
        ibv: bool,
        comp_vector: int,
        buffer: int,
    ) -> None:
        """Mock implementation of :func:`katgpucbf.recv.add_reader`."""
        queue = next(queue_iter)
        stream.add_inproc_reader(queue)

    mocker.patch("katgpucbf.recv.add_reader", autospec=True, side_effect=add_reader)
    return queues
