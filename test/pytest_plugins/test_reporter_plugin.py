################################################################################
# Copyright (c) 2026, National Research Foundation (SARAO)
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

"""Reporter plugin unit tests."""

import json

import pytest


@pytest.fixture(autouse=True)
def setup_pytester(pytester: pytest.Pytester) -> None:
    """Set up pytester with conftest, ini, and test files."""
    pytester.makeconftest(
        """
        pytest_plugins = [
            "katgpucbf.pytest_plugins.reporter_plugin",
        ]
        """
    )
    pytester.makeini(
        """
        [pytest]
        addopts = --report-log=report.json
        """
    )
    pytester.copy_example("demo.py")


def _list_test_report_messages(report_data: list[dict], message_type: str) -> list[dict]:
    """List all messages of a given type in the report data."""
    messages = []
    for entry in report_data:
        if entry.get("$report_type") == "TestReport":
            # Check user_properties for pdf_report_data
            user_props = entry.get("user_properties", [])
            for prop_name, prop_value in user_props:
                if prop_name == "pdf_report_data" and isinstance(prop_value, list):
                    for msg in prop_value:
                        if msg.get("$msg_type") == message_type:
                            messages.append(msg)
    return messages


def test_slow_fixture_updates_timestamp(pytester: pytest.Pytester) -> None:
    """Test that the timestamp is updated when the slow fixture is used."""
    result = pytester.runpytest("demo.py::test_passes")
    assert result.ret == 0
    result.assert_outcomes(passed=1)
    assert len(result.errlines) == 0

    # Read and parse the report.json file
    report_file = pytester.path / "report.json"
    assert report_file.exists(), "report.json file should exist"

    with open(report_file, encoding="utf-8") as f:
        report_data = [json.loads(line) for line in f]

    found_duration = False
    for entry in report_data:
        if entry.get("$report_type") == "TestReport":
            # Check user_properties for pdf_report_data
            duration = entry.get("duration")
            if entry.get("when") == "call":
                continue
            assert isinstance(duration, float)
            assert duration > 0.5
            found_duration = True
            break
    assert found_duration, "Duration should be present in the report"


def test_failure(pytester: pytest.Pytester) -> None:
    """Test that the test failure is reported correctly."""
    result = pytester.runpytest("demo.py::test_assert_failure")
    result.assert_outcomes(failed=1)
    assert len(result.errlines) == 0
    assert "1 == 2" in result.outlines[-2]


def test_figure_creates_binary_figure_in_report(pytester: pytest.Pytester) -> None:
    """Test that the figure test is reported correctly."""
    result = pytester.runpytest("demo.py::test_figure_plot")
    result.assert_outcomes(passed=1)
    assert len(result.errlines) == 0

    # Read and parse the report.json file
    report_file = pytester.path / "report.json"
    assert report_file.exists(), "report.json file should exist"

    with open(report_file, encoding="utf-8") as f:
        report_data = [json.loads(line) for line in f]

    # Find test report entries and check for binary_figure data
    found_binary_figure = False
    for msg in _list_test_report_messages(report_data, "step"):
        items = msg.get("items", [])
        for item in items:
            if item.get("$msg_type") == "binary_figure":
                assert "content" in item, "binary_figure should have content field"
                assert "type" in item, "binary_figure should have type field"
                assert item["type"] == "pdf", "binary_figure type should be pdf"
                assert len(item["content"]) > 0, "binary_figure content should not be empty"
                found_binary_figure = True

    assert found_binary_figure, "binary_figure data should be present in the report"


def test_check_test_is_reported_correctly(pytester: pytest.Pytester) -> None:
    """Test that the check test is reported correctly."""
    result = pytester.runpytest("demo.py::test_check_with_failures")
    result.assert_outcomes(failed=1)
    assert len(result.errlines) == 0

    # Read and parse the report.json file
    report_file = pytester.path / "report.json"
    assert report_file.exists(), "report.json file should exist"

    with open(report_file, encoding="utf-8") as f:
        report_data = [json.loads(line) for line in f]

    # Find test report entries and check for step details
    found_bad_things_step = False
    found_good_things_step = False
    bad_things_failures = []

    for msg in _list_test_report_messages(report_data, "step"):
        step_message = msg.get("message", "")
        items = msg.get("items", [])
        if step_message == "Expect some bad things":
            found_bad_things_step = True
            for item in items:
                if item.get("$msg_type") == "failure":
                    bad_things_failures.append(item.get("message", ""))
        if step_message == "Expect some good things":
            found_good_things_step = True

    assert any("(1 * 3) == 2" in line for line in result.outlines), "'1 * 3 == 2' should be present in the report"
    assert any("check with msg" in line for line in result.outlines), "'check with msg' should be present in the report"
    assert found_bad_things_step, "Step 'Expect some bad things' should be present in the report"
    assert len(bad_things_failures) > 0, "Step 'Expect some bad things' should have failure items"
    assert found_good_things_step, "Step 'Expect some good things' should be present in the report"


def test_marked_xfail_is_not_reported_as_failed(pytester: pytest.Pytester) -> None:
    """Test that the xfail test is reported correctly."""
    result = pytester.runpytest("demo.py::test_xfail")
    result.assert_outcomes(xfailed=1)
    assert len(result.errlines) == 0
