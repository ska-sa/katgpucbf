################################################################################
# Copyright (c) 2022-2025, National Research Foundation (SARAO)
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

from pathlib import Path

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

    qualification_conftest = Path(__file__).resolve().parent.parent / "conftest.py"
    with open(qualification_conftest, encoding="utf-8") as f:
        conftest_content = f.read()
    # Replace relative imports with absolute imports for pytester
    conftest_content = conftest_content.replace("from .cbf import", "from qualification.cbf import")
    conftest_content = conftest_content.replace("from .recv import", "from qualification.recv import")
    conftest_content = conftest_content.replace("from .reporter import", "from qualification.reporter import")
    demo_test_content = open(Path(__file__).resolve().parent / "demotests.py", encoding="utf-8").read()
    pytester.makeconftest(conftest_content)
    pytester.makeini(pytestini_content)
    pytester.makepyfile(demo_test_content)
    return pytester


def test_slow_fixture_updates_timestamp(setup_pytester: pytest.Pytester) -> None:
    """Test that the timestamp is updated when the slow fixture is used."""
    result = setup_pytester.runpytest("--image-override=::", "--report-log=report.json", "-k test_passes")
    print(result.stdout.str())
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


def test_check_test_is_reported_correctly(setup_pytester: pytest.Pytester) -> None:
    """Test that the check test is reported correctly."""
    result = setup_pytester.runpytest("--image-override=::", "--report-log=report.json", "-k test_check_with_failures")
    result.assert_outcomes(passed=0, failed=1, errors=0, skipped=0, xpassed=0, xfailed=0)
    assert result.duration > 0
    assert len(result.errlines) == 0


def test_marked_xfail_is_not_reported_as_failed(setup_pytester: pytest.Pytester) -> None:
    """Test that the xfail test is reported correctly."""
    result = setup_pytester.runpytest("--image-override=::", "--report-log=report.json", "-k test_xfail")
    result.assert_outcomes(passed=0, failed=0, errors=0, skipped=0, xpassed=0, xfailed=1)
    assert result.duration > 0
    assert len(result.errlines) == 0


def test_failed_np_assertion_dumps_arrays(setup_pytester: pytest.Pytester) -> None:
    """Test that the numpy fail test is reported correctly."""
    result = setup_pytester.runpytest("--image-override=::", "--report-log=report.json", "-k test_numpy_fails")
    result.assert_outcomes(passed=0, failed=1, errors=0, skipped=0, xpassed=0, xfailed=0)
    assert result.duration > 0
    assert result.parseoutcomes()["failed"] == 1


def test_failed_np_assertion_dumps_arrays_and_unwraps_approx(setup_pytester: pytest.Pytester) -> None:
    """Test that the numpy fail approx test is reported correctly."""
    result = setup_pytester.runpytest(
        "--image-override=::", "--report-log=report.json", "-k test_numpy_with_approx_fails"
    )
    result.assert_outcomes(passed=0, failed=1, errors=0, skipped=0, xpassed=0, xfailed=0)
    assert result.duration > 0
