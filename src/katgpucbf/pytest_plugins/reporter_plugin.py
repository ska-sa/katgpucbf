################################################################################
# Copyright (c) 2022-2026, National Research Foundation (SARAO)
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

"""Wrap the functionality of :mod:`katgpucbf.pytest_plugins.reporter` as a pytest plugin."""

import inspect
import time
from collections.abc import Generator

import matplotlib.style
import pytest
import pytest_check

from .reporter import Reporter

pytest_plugins = ["pytest_check"]
pdf_report_data_key = pytest.StashKey[dict]()


@pytest.fixture(autouse=True, scope="function")
def pdf_report(request, monkeypatch) -> Reporter:
    """Fixture for logging steps in a test."""
    reporter = Reporter(request.node.stash[pdf_report_data_key], raw_data=request.config.getini("raw_data"))
    orig_log_failure = pytest_check.check_log.log_failure
    orig_stack = inspect.stack

    def stack():
        # The real log_failure function constructs a backtrace, and inserting
        # our wrapper into the call stack messes that up. We need to have it
        # skip an extra level for each wrapper we're injecting.
        return orig_stack()[2:]

    def log_failure(msg="", check_str="", tb=None):
        __tracebackhide__ = True
        if check_str:
            reporter.failure(f"Failed assertion: {msg}: {check_str}")
        else:
            reporter.failure(f"Failed assertion: {msg}")
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(inspect, "stack", stack)
            return orig_log_failure(msg, check_str, tb)

    # Patch the central point where pytest-check logs failures so that we can
    # insert them into the test procedure.
    monkeypatch.setattr(pytest_check.check_log, "log_failure", log_failure)
    # context_manager uses `from .check_log import log_failure` so we have to
    # patch it under that name.
    monkeypatch.setattr(pytest_check.context_manager, "log_failure", log_failure)
    return reporter


def pytest_addoption(parser: pytest.Parser, pluginmanager: pytest.PytestPluginManager) -> None:
    """Add the inifile options."""
    parser.addini("raw_data", "Include raw data for figures", "bool", False)


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line("markers", "requirements(reqs): indicate which system engineering requirements are tested")
    config.addinivalue_line("markers", "name(name): human-readable name for the test")


@pytest.hookimpl(wrapper=True)
def pytest_runtest_setup(item) -> Generator[None, None, None]:
    """Set up the user property for passing data to the report generator."""
    blurb = inspect.getdoc(item.function)
    if blurb is None:
        raise AssertionError(f"Test {item.name} has no docstring")
    reqs: list[str] = []
    for marker in item.iter_markers("requirements"):
        if isinstance(marker.args[0], tuple | list):
            reqs.extend(marker.args[0])
        else:
            reqs.extend(name.strip() for name in marker.args[0].split(",") if name.strip())
    data = [{"$msg_type": "test_info", "blurb": blurb, "test_start": time.time(), "requirements": reqs}]
    name_marker = item.get_closest_marker("name")
    if name_marker is not None:
        data[0]["test_name"] = name_marker.args[0]
    item.user_properties.append(("pdf_report_data", data))
    item.stash[pdf_report_data_key] = data
    yield


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item) -> Generator[None, None, None]:
    """Update the test_start field when the test is actually started.

    This gives a more accurate start time than the one recorded by
    :func:`pytest_runtest_setup`, which is the time at which setup
    started.
    """
    item.stash[pdf_report_data_key][0]["test_start"] = time.time()
    yield


@pytest.fixture(autouse=True)
def matplotlib_report_style() -> Generator[None, None, None]:
    """Set the style of all matplotlib plots."""
    with (
        matplotlib.style.context("ggplot"),
        matplotlib.rc_context(
            {
                # Serif fonts better match the rest of the document
                "font.family": "serif",
                "font.serif": ["Liberation Serif"],
                # A lot of the graphs are noisy and a narrower linewidth makes
                # the detail easier to see.
                "lines.linewidth": 0.3,
            }
        ),
    ):
        yield
