#!/usr/bin/env python3

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

"""Generate a PDF based on the intermediate JSON output."""
import argparse
import json
import os
import pathlib
import re
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Literal, Optional, Tuple, Union

import pytest
from docutils.core import publish_parts
from dotenv import dotenv_values
from pylatex import (
    Command,
    Document,
    LongTable,
    MiniPage,
    MultiColumn,
    NoEscape,
    Section,
    Subsection,
    Subsubsection,
    TextColor,
)
from pylatex.base_classes import Environment
from pylatex.labelref import Hyperref
from pylatex.package import Package
from pylatex.utils import bold

RESOURCE_PATH = pathlib.Path(__file__).parent
UNKNOWN = "unknown"


class LstListing(Environment):
    """A class to wrap LaTeX's lstlisting environment."""

    packages = [Package("listings")]
    escape = False
    content_separator = "\n"


def readable_duration(duration: float) -> str:
    """Change a duration in seconds to something human-readable."""
    if duration < 60:
        return f"{duration:.3f} s"
    else:
        seconds = duration % 60
        duration /= 60
        if duration < 60:
            return f"{int(duration)} m {seconds:.3f} s"
        else:
            minutes = int(duration % 60)
            duration /= 60
            return f"{int(duration)} h {minutes} m {seconds:.3f} s"


def rst2latex(text: str) -> str:
    """Turn a section of ReStructured Text (like a docstring) into LaTeX."""
    return publish_parts(source=text, writer_name="latex")["body"]


@dataclass
class Detail:
    """A message logged by ``pdf_report.detail``."""

    message: str
    timestamp: float


@dataclass
class Figure:
    """A figure created by ``pdf_report.figure`` or ``pdf_report.raw_figure``."""

    code: str


@dataclass
class Step:
    """A step created by ``pdf_report.step``."""

    message: str
    details: List[Union[Detail, Figure]] = field(default_factory=list)


@dataclass
class Result:
    """A single test execution.

    This combines the setup, call and teardown phases, which appear as separate
    lines in the report.json file.
    """

    nodeid: str
    name: str
    blurb: str
    steps: List[Step] = field(default_factory=list)
    outcome: Literal["passed", "failed", "skipped"] = "failed"
    failure_messages: List[str] = field(default_factory=list)
    start_time: Optional[float] = None
    duration: float = 0.0
    config: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Interface:
    """A network interface."""

    name: str
    driver: str
    firmware_version: str
    version: str


@dataclass(frozen=True)
class GPU:
    """Information about a GPU."""

    number: int
    model_name: str
    driver: str
    vbios: str


@dataclass(frozen=True)
class Host:
    """Configuration information for a host."""

    kernel: str
    cpus: Tuple[str, ...]
    interfaces: Tuple[Interface, ...]
    gpus: Tuple[GPU, ...]
    ram: Optional[int]
    product_name: str = UNKNOWN
    board_name: str = UNKNOWN
    bios_version: str = UNKNOWN


@dataclass
class ConfigParam:
    """A configuration parameter describing software / hardware used in the test run."""

    name: str
    value: str


@dataclass
class TestConfiguration:
    """Global configuration of the test."""

    params: List[ConfigParam] = field(default_factory=list)
    hosts: Dict[Host, List[str]] = field(default_factory=lambda: defaultdict(list))  # Hostnames for each config


def _parse_detail(detail: dict) -> Union[Detail, Figure]:
    """Parse a single detail from the details of a step."""
    msg_type = detail["$msg_type"]
    if msg_type == "detail":
        return Detail(detail["message"], detail["timestamp"])
    elif msg_type == "figure":
        return Figure(detail["code"])
    else:
        raise ValueError(f"Do not know how to parse detail $msg_type of {msg_type!r}")


def _parse_report_data(result: Result, msg: dict) -> None:
    """Parse a single entry in the ``pdf_report_data`` user property."""
    assert isinstance(msg, dict)
    msg_type = msg["$msg_type"]
    if msg_type == "step":
        details = [_parse_detail(detail) for detail in msg["details"]]
        result.steps.append(Step(msg["message"], details))
    elif msg_type == "test_info":
        if not result.blurb:
            result.blurb = msg["blurb"]
        if result.start_time is None:
            result.start_time = float(msg["test_start"])
    elif msg_type == "config":
        msg = dict(msg)  # Avoid mutating the caller's copy
        msg.pop("$msg_type")  # We don't need this.
        result.config.update(msg)
    else:
        raise ValueError(f"Do not know how to parse $msg_type of {msg_type!r}")


def _parse_interface(labels: dict) -> Interface:
    """Parse a single network interface from ``node_ethtool_info`` labels."""
    return Interface(
        name=labels["device"],
        driver=labels.get("driver", UNKNOWN),
        firmware_version=labels.get("firmware_version", UNKNOWN),
        version=labels.get("version", UNKNOWN),
    )


def _parse_gpu(labels: dict) -> GPU:
    """Parse a single GPU from ``dcgm_exporter_info`` labels."""
    return GPU(
        number=int(labels["gpu"]),
        model_name=labels["modelName"],
        driver=labels.get("DCGM_FI_DRIVER_VERSION", UNKNOWN),
        vbios=labels.get("DCGM_FI_DEV_VBIOS_VERSION", UNKNOWN),
    )


def _parse_host(msg: dict) -> Tuple[str, Host]:
    """Parse a single host configuration line."""
    kernel = UNKNOWN
    cpus: Dict[int, str] = {}  # Indexed by package
    interfaces = []
    gpus = []
    kwargs = {}
    ram: Optional[int] = None

    for metric in msg["config"]:
        labels = metric["metric"]
        value = float(metric["value"][1])
        metric_name = labels["__name__"]
        if metric_name == "node_dmi_info":
            for label, label_value in labels.items():
                if label in {"product_name", "board_name", "bios_version"}:
                    kwargs[label] = label_value
        elif metric_name == "node_cpu_info" and "model_name" in labels:
            # The check for model_name is because older versions of
            # node-exporter didn't provide it.
            cpus[int(labels["package"])] = labels["model_name"]
        elif metric_name == "node_memory_MemTotal_bytes":
            ram = int(value)
        elif metric_name == "node_ethtool_info":
            interfaces.append(_parse_interface(labels))
        elif metric_name == "dcgm_exporter_info":
            gpus.append(_parse_gpu(labels))
        elif metric_name == "node_uname_info":
            kernel = " ".join([labels["sysname"], labels["release"], labels["version"]])
    interfaces.sort(key=lambda interface: interface.name)
    gpus.sort(key=lambda gpu: gpu.number)
    host = Host(
        kernel=kernel,
        cpus=tuple(model_name for _, model_name in sorted(cpus.items())),
        interfaces=tuple(interfaces),
        gpus=tuple(gpus),
        ram=ram,
        **kwargs,
    )
    return msg["hostname"], host


def parse(input_data: List[dict]) -> Tuple[TestConfiguration, List[Result]]:
    """Parse the data written by pytest-reportlog.

    Parameters
    ----------
    input_data
        A Python list, which should have been loaded from pytest-reportlog's
        JSON output.

    Returns
    -------
    Lists of :class:`ConfigParam` and :class:`Result` objects representing the
    test configuration and the results of all the tests logged in the JSON input.
    """
    test_configuration = TestConfiguration()
    results: List[Result] = []
    for line in input_data:
        if line["$report_type"] == "TestConfiguration":
            # Tabulate this somehow. It'll need to modify the return.
            line.pop("$report_type")
            for config_param_name, config_param_value in line.items():
                test_configuration.params.append(ConfigParam(config_param_name, config_param_value))
            continue  # We've removed $report_type, which would break later code
        elif line["$report_type"] == "HostConfiguration":
            hostname, host = _parse_host(line)
            test_configuration.hosts[host].append(hostname)

        if line["$report_type"] != "TestReport":
            continue
        # _from_json is an "experimental" function.
        report = pytest.TestReport._from_json(line)
        nodeid = report.nodeid
        if not results or results[-1].nodeid != nodeid:
            # It's the first time we've seen this nodeid (otherwise we merge
            # with the existing Result).
            results.append(Result(nodeid, report.location[2], ""))
        result = results[-1]
        # The teardown phase has all the log messages, so we ignore the setup and call phases.
        if report.when == "teardown":
            for prop in report.user_properties:
                if prop[0] == "pdf_report_data":
                    for msg in prop[1][:]:
                        assert isinstance(msg, dict)
                        _parse_report_data(result, msg)
        # If teardown fails, the whole test should be seen as failing
        if report.outcome != "passed" or report.when == "call":
            result.outcome = report.outcome
        # The test duration will be the sum of setup, call and teardown.
        result.duration += report.duration
        if report.longrepr is not None:
            # Strip out ANSI color escape sequences
            failure_message = re.sub("\x1b\\[.*?m", "", report.longreprtext)
            result.failure_messages.append(failure_message)
    return test_configuration, results


def fix_test_name(test_name: str) -> str:
    """Change a test's name from a pytest one to a more human-friendly one."""
    return " ".join([word.capitalize() for word in test_name.split("_") if word != "test"])


def _doc_test_configuration(doc: Document, test_configuration: TestConfiguration):
    with doc.create(Section("Test Configuration")) as config_section:
        with config_section.create(LongTable(r"|r|l|")) as config_table:
            config_table.add_hline()
            config_table.add_row([bold("Configuration Parameter"), bold("Value")])
            config_table.add_hline()
            for config_param in test_configuration.params:
                config_table.add_row([config_param.name, config_param.value])
                config_table.add_hline()
            config_table.add_row(["Some other software version", "x.y.ZZ"])
            config_table.add_hline()

    with doc.create(Section("Hosts")) as hosts_section:
        for host, hostnames in test_configuration.hosts.items():
            with hosts_section.create(LongTable(r"|r|l|")) as host_table:
                host_table.add_hline()
                for hostname in sorted(hostnames):
                    host_table.add_row([MultiColumn(2, align="|l|", data=hostname)])
                host_table.add_hline()
                host_table.add_row([MultiColumn(2, align="|c|", data=bold("System"))])
                host_table.add_hline()
                assert len(host.cpus) <= 1, "Need to extend to handle multiple CPUs"
                host_table.add_row("CPU", host.cpus[0] if host.cpus else UNKNOWN)
                if host.ram is not None:
                    host_table.add_row("RAM", f"{host.ram / 2**30:.3f} GiB")
                else:
                    host_table.add_row("RAM", "unknown")
                host_table.add_row("Product name", host.product_name)
                host_table.add_row("Board name", host.board_name)
                host_table.add_row("BIOS version", host.bios_version)
                host_table.add_row("Kernel", host.kernel)
                host_table.add_hline()
                for interface in host.interfaces:
                    host_table.add_row([MultiColumn(2, align="|c|", data=bold(interface.name))])
                    host_table.add_hline()
                    host_table.add_row("Driver", interface.driver + " " + interface.version)
                    host_table.add_row("Firmware", interface.firmware_version)
                    host_table.add_hline()
                for gpu in host.gpus:
                    host_table.add_row([MultiColumn(2, align="|c|", data=bold(f"GPU {gpu.number}"))])
                    host_table.add_hline()
                    host_table.add_row("Model", gpu.model_name)
                    host_table.add_row("Driver", gpu.driver)
                    host_table.add_row("VBIOS", gpu.vbios)
                    host_table.add_hline()


def document_from_json(input_data: Union[str, list]) -> Document:
    """Take a test result and generate a :class:`pylatex.Document` for a report.

    Parameters
    ----------
    input_data
        Either the list of parsed JSON entries, or a path to the report file
        generated by :samp:`pytest --report-log={filename}`.

    Returns
    -------
    doc
        A document
    """
    if not isinstance(input_data, list):
        result_list = []
        with open(input_data) as fp:
            for line in fp:
                result_list.append(json.loads(line))
    else:
        result_list = input_data

    test_configuration, results = parse(result_list)

    # Get information from the .env file, such as tester's name, which shouldn't
    # really be in git. Allow environment to override
    config = {**dotenv_values(), **os.environ}

    doc = Document(
        document_options=["11pt", "english", "twoside"],
        inputenc=None,  # katdoc inputs inputenc with specific options, so prevent a clash
    )
    today = date.today()  # TODO: should store inside the JSON
    doc.set_variable("theAuthor", config.get("TESTER_NAME", "Unknown"))
    doc.set_variable("docDate", today.strftime("%d %B %Y"))
    doc.preamble.append(NoEscape((RESOURCE_PATH / "preamble.tex").read_text()))
    doc.append(Command("title", "Integration Test Report"))
    doc.append(Command("makekatdocbeginning"))

    _doc_test_configuration(doc, test_configuration)

    with doc.create(Section("Result Summary")) as summary_section:
        with summary_section.create(LongTable(r"|r|l|r|")) as summary_table:
            total_duration: float = 0.0
            passed = 0
            failed = 0
            summary_table.add_hline()
            summary_table.add_row(
                [
                    bold("Test"),
                    bold("Result"),
                    bold("Duration"),
                ]
            )
            summary_table.add_hline()
            for result in results:
                summary_table.add_row(
                    # PyLatex strips (among other things) underscores from label names, apparently they're dangerous.
                    # See: https://jeltef.github.io/PyLaTeX/latest/pylatex/pylatex.labelref.html#pylatex.labelref.Marker
                    [
                        Hyperref(f"subsec:{result.name.replace('_', '')}", fix_test_name(result.name)),
                        TextColor("green" if result.outcome == "passed" else "red", result.outcome),
                        readable_duration(result.duration),
                    ]
                )
                summary_table.add_hline()
                total_duration += result.duration
                if result.outcome == "passed":
                    passed += 1
                else:
                    failed += 1
        summary_section.append(f"{len(results)} tests run, with {passed} passing and {failed} failing.\n")
        summary_section.append(f"Total test duration: {readable_duration(total_duration)}")

    with doc.create(Section("Detailed Test Results")) as section:
        for result in results:
            with section.create(Subsection(fix_test_name(result.name), label=result.name)):
                section.append(NoEscape(rst2latex(result.blurb) + "\n\n"))
                section.append("Outcome: ")
                section.append(TextColor("green" if result.outcome == "passed" else "red", result.outcome.upper()))
                section.append(Command("hspace", "1cm"))
                assert result.start_time is not None
                section.append(f"Test start time: {datetime.fromtimestamp(float(result.start_time)).strftime('%T')}")
                section.append(Command("hspace", "1cm"))
                section.append(f"Duration: {readable_duration(result.duration)} seconds\n")

                with section.create(LongTable(r"|l|p{0.4\linewidth}|")) as config_table:
                    config_table.add_hline()
                    config_table.add_row((MultiColumn(2, align="|c|", data=bold("Test Configuration")),))
                    config_table.add_hline()
                    for key, value in result.config.items():
                        config_table.add_row([key.capitalize(), value])
                        config_table.add_hline()

                with section.create(Subsubsection("Procedure", label=False)) as procedure:
                    with section.create(LongTable(r"|l|p{0.7\linewidth}|")) as procedure_table:
                        for step in result.steps:
                            procedure_table.add_hline()
                            procedure_table.add_row((MultiColumn(2, align="|l|", data=bold(step.message)),))
                            procedure_table.add_hline()
                            for detail in step.details:
                                if isinstance(detail, Detail):
                                    procedure_table.add_row(
                                        [
                                            f"{readable_duration(detail.timestamp - result.start_time)}",
                                            detail.message,
                                        ]
                                    )
                                elif isinstance(detail, Figure):
                                    mp = MiniPage(width=NoEscape(r"\textwidth"))
                                    mp.append(NoEscape(r"\center"))
                                    mp.append(NoEscape(detail.code))
                                    procedure_table.add_row((MultiColumn(2, align="|c|", data=mp),))
                                procedure_table.add_hline()

                    if result.failure_messages:
                        with procedure.create(LstListing()) as failure_message:
                            for message in result.failure_messages:
                                failure_message.append(message)

    return doc


def main():
    """Convert a JSON report to a PDF."""
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Output of pytest --report-log=...")
    parser.add_argument("pdf", help="PDF file to write")
    args = parser.parse_args()
    doc = document_from_json(args.input)
    if args.pdf.endswith(".pdf"):
        args.pdf = args.pdf[:-4]  # Strip .pdf suffix, because generate_pdf appends it
    with tempfile.NamedTemporaryFile(mode="w", prefix="latexmkrc") as latexmkrc:
        # TODO: latexmk uses Perl, which has different string parsing to
        # Python. If the path contains both a single quote and a special
        # symbol it will not produce a valid Perl string.
        parent_dir = str(RESOURCE_PATH)
        latexmkrc.write(f"ensure_path('TEXINPUTS', {parent_dir!r})\n")
        latexmkrc.flush()
        doc.generate_pdf(args.pdf, compiler="latexmk", compiler_args=["--pdf", "-r", latexmkrc.name])


if __name__ == "__main__":
    main()
