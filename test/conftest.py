################################################################################
# Copyright (c) 2020-2021, National Research Foundation (SARAO)
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
"""

import pytest

pytest_plugins = ["katsdpsigproc.pytest_plugin"]


def pytest_configure(config) -> None:
    """Register custom markers."""
    config.addinivalue_line("markers", "combinations(names, *values): test combinations of values")


def pytest_addoption(parser) -> None:
    """Register new command-line options."""
    group = parser.getgroup("combinations")
    group.addoption("--all-combinations", action="store_true", help="Test the full Cartesian product of parameters")


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
        if all_combinations:
            for name, value_list in zip(names, values):
                metafunc.parametrize(name, value_list)
        else:
            # Determine the total number of combinations to test, which will be
            # the longest of the value lists.
            n = max(len(value_list) for value_list in values)
            combos = []
            for i in range(n):
                if i == n - 1:
                    # Ensure that the last test uses the last item from each
                    # list. If the lists contain tests of increasing
                    # complexity, then this ensures that most complex
                    # combination gets tested.
                    combo = tuple(value_list[-1] for value_list in values)
                else:
                    combo = tuple(value_list[i % len(value_list)] for value_list in values)
                combos.append(combo)
            metafunc.parametrize(names, combos)
