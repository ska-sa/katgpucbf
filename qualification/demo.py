################################################################################
# Copyright (c) 2022, National Research Foundation (SARAO)
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

import matplotlib.figure
import pytest

from .reporter import Reporter


@pytest.mark.requirements("DEMO-000")
def test_passes(pdf_report: Reporter) -> None:
    r"""Pass the test.

    Here is some maths: :math:`e^{\pi j} + 1 = 0`.

    Verification method
    -------------------
    Don't actually test anything.
    """
    pdf_report.step("Do things")
    pdf_report.detail("Thing implementation detail 1")
    pdf_report.detail("Thing implementation detail 2")


def test_assert_failure(pdf_report: Reporter) -> None:
    """Always fail."""
    pdf_report.step("Test something bad")
    pdf_report.detail("Check that 1 = 2")
    assert 1 == 2


def test_figure(pdf_report: Reporter) -> None:
    """Plot a figure."""
    fig = matplotlib.figure.Figure()
    ax = fig.subplots()
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_title("Caption")
    ax.plot([1, 3, 4], [2, 6, 3], label="A line")
    ax.legend()

    pdf_report.step("Show a figure")
    pdf_report.figure(fig)


def test_expect(pdf_report: Reporter, expect) -> None:
    """Use ``expect`` and observe failures."""
    pdf_report.step("Expect some bad things")
    expect(1 == 2)
    expect(3 == 4)
    pdf_report.step("Expect some good things")
    expect(1 == 1)
