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

"""Example tests helpful in developing the reporting framework."""

import json
import re
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def pytestini_content() -> str:
    """The pytest.ini content for the demo tests."""
    return """
        [pytest]
        asyncio_default_fixture_loop_scope = function
        tester = Test
        master_controller_host = localhost
        master_controller_port = 5001
        prometheus_url = http://localhost:9090
        product_name = test_cbf
        interface = lo
        interface_gbps = 90
        use_ibv = false
        cores =
        default_antennas = 8
        max_antennas = 8
        wideband_channels = 1024
        narrowband_channels = 32768
        narrowband_decimation = 8
        vlbi_decimation = 8
        bands = l
        beams = 4
        raw_data = false
        array_dir =
        """


@pytest.fixture(scope="function")
def setup_pytester(pytester: pytest.Pytester, pytestini_content: str) -> pytest.Pytester:
    """Set up pytester with conftest, ini, and test files."""
    # Reset report values by deleting the report file if it exists
    report_file = pytester.path / "report.json"
    if report_file.exists():
        report_file.unlink()

    proj_root_path = Path(__file__).resolve().parent.parent.parent.parent
    qualification_conftest = proj_root_path / "qualification" / "conftest.py"
    with open(qualification_conftest, encoding="utf-8") as f:
        conftest_content = f.read()
    # Replace relative imports with absolute imports for pytester
    conftest_content = re.sub(r"from \.(\w+) import", r"from qualification.\1 import", conftest_content)

    demo_test_content = open(proj_root_path / "qualification" / "demo" / "demo.py", encoding="utf-8").read()
    pytester.makeconftest(conftest_content)
    pytester.makeini(pytestini_content)
    pytester.makepyfile(demo_test_content)
    return pytester


def test_slow_fixture_updates_timestamp(setup_pytester: pytest.Pytester) -> None:
    """Test that the timestamp is updated when the slow fixture is used."""
    result = setup_pytester.runpytest("--image-override=::", "--report-log=report.json", "-k test_passes")
    assert result.ret == 0
    result.assert_outcomes(passed=1, failed=0, errors=0, skipped=0, xpassed=0, xfailed=0)
    # read the report.json file and confirm test
    assert result.duration > 0
    assert len(result.errlines) == 0


def test_failure(setup_pytester: pytest.Pytester) -> None:
    """Test that the test failure is reported correctly."""
    result = setup_pytester.runpytest("--image-override=::", "--report-log=report.json", "-k test_assert_failure")
    result.assert_outcomes(passed=0, failed=1, errors=0, skipped=0, xpassed=0, xfailed=0)
    # read the report.json file and confirm test
    assert result.duration > 0
    assert "1 == 2" in result.outlines[-2]


def test_figure_creates_binary_figure_in_report(setup_pytester: pytest.Pytester) -> None:
    """Test that the figure test is reported correctly."""
    result = setup_pytester.runpytest("--image-override=::", "--report-log=report.json", "-k test_figure_plot")
    result.assert_outcomes(passed=1, failed=0, errors=0, skipped=0, xpassed=0, xfailed=0)
    assert result.duration > 0
    assert len(result.errlines) == 0

    # Read and parse the report.json file
    report_file = setup_pytester.path / "report.json"
    assert report_file.exists(), "report.json file should exist"

    with open(report_file, encoding="utf-8") as f:
        report_data = [json.loads(line) for line in f]

    # Find test report entries and check for binary_figure data
    found_binary_figure = False
    for entry in report_data:
        if entry.get("$report_type") == "TestReport":
            # Check user_properties for pdf_report_data
            user_props = entry.get("user_properties", [])
            for prop_name, prop_value in user_props:
                if prop_name == "pdf_report_data" and isinstance(prop_value, list):
                    # Look through the data for binary_figure entries
                    for msg in prop_value:
                        if msg.get("$msg_type") == "step":
                            items = msg.get("items", [])
                            for item in items:
                                if item.get("$msg_type") == "binary_figure":
                                    assert "content" in item, "binary_figure should have content field"
                                    assert "type" in item, "binary_figure should have type field"
                                    assert item["type"] == "pdf", "binary_figure type should be pdf"
                                    assert len(item["content"]) > 0, "binary_figure content should not be empty"
                                    found_binary_figure = True

    assert found_binary_figure, "binary_figure data should be present in the report"


@pytest.mark.xfail(reason="Need to fix the check plugin for pytester usage")
def test_check_test_is_reported_correctly(setup_pytester: pytest.Pytester) -> None:
    """Test that the check test is reported correctly."""
    result = setup_pytester.runpytest("--image-override=::", "--report-log=report.json", "-k test_check_with_failures")
    result.assert_outcomes(passed=0, failed=1, errors=0, skipped=0, xpassed=0, xfailed=0)
    assert result.duration > 0
    assert len(result.errlines) == 0

    # Read and parse the report.json file
    report_file = setup_pytester.path / "report.json"
    assert report_file.exists(), "report.json file should exist"

    with open(report_file, encoding="utf-8") as f:
        report_data = [json.loads(line) for line in f]

    # Find test report entries and check for step details
    found_bad_things_step = False
    found_good_things_step = False
    bad_things_failures = []

    # TODO: Simplify and fix check plugin for pytester usage
    for entry in report_data:
        if entry.get("$report_type") == "TestReport":
            # Check user_properties for pdf_report_data
            user_props = entry.get("user_properties", [])
            for prop_name, prop_value in user_props:
                if prop_name == "pdf_report_data" and isinstance(prop_value, list):
                    # Look through the data for step entries
                    for msg in prop_value:
                        if msg.get("$msg_type") == "step":
                            step_message = msg.get("message", "")
                            items = msg.get("items", [])

                            if step_message == "Expect some bad things":
                                found_bad_things_step = True
                                # Collect failure items from this step
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


def test_marked_xfail_is_not_reported_as_failed(setup_pytester: pytest.Pytester) -> None:
    """Test that the xfail test is reported correctly."""
    result = setup_pytester.runpytest("--image-override=::", "--report-log=report.json", "-k test_xfail")
    result.assert_outcomes(passed=0, failed=0, errors=0, skipped=0, xpassed=0, xfailed=1)
    assert result.duration > 0
    assert len(result.errlines) == 0


def _extract_array_path_from_output(outlines: list[str], pytester_path: Path) -> Path | None:
    """Extract the array path from pytest output lines that contain 'Arrays written to'."""
    for line in outlines:
        if "Arrays written to" in line:
            # Extract the path after "Arrays written to "
            prefix = "Arrays written to "
            idx = line.find(prefix)
            if idx != -1:
                array_path_str = line[idx + len(prefix) :].strip()
                # The path might be relative to pytester.path or absolute
                array_path = Path(array_path_str)
                if not array_path.is_absolute():
                    array_path = pytester_path / array_path
                return array_path
    return None


def test_failed_np_assertion_dumps_arrays(setup_pytester: pytest.Pytester) -> None:
    """Test that the numpy fail test is reported correctly."""
    result = setup_pytester.runpytest("--image-override=::", "--report-log=report.json", "-k test_numpy_fails")
    result.assert_outcomes(passed=0, failed=1, errors=0, skipped=0, xpassed=0, xfailed=0)
    assert result.duration > 0
    assert result.parseoutcomes()["failed"] == 1
    assert any("Arrays written to" in line for line in result.outlines), (
        "Arrays written to should be present in the report"
    )

    # Extract the path from the output
    array_path = _extract_array_path_from_output(result.outlines, setup_pytester.path)

    assert array_path is not None, "Could not find array path in output"
    assert array_path.exists(), f"Array file should exist at {array_path}"

    # Open and verify the file
    with np.load(array_path) as data:
        assert "ACTUAL" in data, "Array file should contain ACTUAL array"
        assert "DESIRED" in data, "Array file should contain DESIRED array"


def test_failed_np_assertion_dumps_arrays_and_unwraps_approx(setup_pytester: pytest.Pytester) -> None:
    """Test that the numpy fail approx test is reported correctly."""
    result = setup_pytester.runpytest(
        "--image-override=::", "--report-log=report.json", "-k test_numpy_with_approx_fails"
    )
    result.assert_outcomes(passed=0, failed=1, errors=0, skipped=0, xpassed=0, xfailed=0)
    assert result.duration > 0
    assert result.parseoutcomes()["failed"] == 1
    assert any("Arrays written to" in line for line in result.outlines), (
        "Arrays written to should be present in the report"
    )
    # Extract the path from the output
    array_path = _extract_array_path_from_output(result.outlines, setup_pytester.path)

    assert array_path is not None, "Could not find array path in output"
    assert array_path.exists(), f"Array file should exist at {array_path}"

    # Open and verify the file
    with np.load(array_path) as data:
        assert "ACTUAL" in data, "Array file should contain ACTUAL array"
        assert "DESIRED" in data, "Array file should contain DESIRED array"
