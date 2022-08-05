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
import logging
import os
import pathlib
import re
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Literal, Mapping, Optional, Sequence, Set, Tuple, Union
from uuid import UUID

import docutils.writers.latex2e
import pytest
from docutils.core import publish_parts
from pylatex import (
    Command,
    Document,
    LongTable,
    MiniPage,
    MultiColumn,
    NoEscape,
    Section,
    SmallText,
    Subsection,
    Subsubsection,
    TextColor,
)
from pylatex.base_classes import Container, Environment
from pylatex.base_classes.latex_object import LatexObject
from pylatex.labelref import Hyperref, Label, Marker
from pylatex.lists import Description
from pylatex.package import Package
from pylatex.utils import bold, escape_latex

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


class DocumentClass(docutils.writers.latex2e.DocumentClass):
    """Override docutils to demote all sections by two levels."""

    def __init__(self) -> None:
        super().__init__("article", False)
        del self.sections[:2]


class Translator(docutils.writers.latex2e.LaTeXTranslator):
    """Override docutils to use :class:`DocumentClass`."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.d_class = DocumentClass()


def rst2latex(text: str) -> str:
    """Turn a section of ReStructured Text (like a docstring) into LaTeX."""
    writer = docutils.writers.latex2e.Writer()
    writer.translator_class = Translator
    return publish_parts(source=text, writer=writer)["body"]


@dataclass
class Detail:
    """A message logged by ``pdf_report.detail``."""

    message: str
    timestamp: float


@dataclass
class Figure:
    """A figure created by ``pdf_report.figure`` or ``pdf_report.raw_figure``."""

    code: str

    def __post_init__(self) -> None:
        # Matplotlib relies on commands with @ in them, and uses
        # \makeatletter to change the category code. However, because of the
        # way longtable and multicolumn interact, the content is scanned
        # before the category code can be changed. Fix it by using \csname,
        # which is more robust.
        self.code = re.sub(r"\\(pgfsys@[a-z@]+)", r"\\csname \1\\endcsname", self.code)


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
    name: str  # Name by which the test is known to pytest (including parameters)
    text_name: str  # Name shown in the document
    blurb: str
    requirements: List[str] = field(default_factory=list)
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
    bus_id: str


@dataclass(frozen=True)
class GPU:
    """Information about a GPU."""

    number: int
    model_name: str
    driver: str
    vbios: str
    bus_id: str
    ram: int


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
class Task:
    """A single Mesos task within a correlator."""

    host: str
    version: str  # Docker image
    git_version: str


@dataclass
class CorrelatorConfiguration:
    """Configuration information for a correlator instance."""

    description: str
    uuid: UUID
    tasks: Dict[str, Task]


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
    correlators: List[CorrelatorConfiguration] = field(default_factory=list)


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
        if not result.text_name:
            if "test_name" in msg:
                # Replace the Python identifier, but keep any parameters
                result.text_name = msg["test_name"]
                if "[" in result.name:
                    result.text_name += result.name[result.name.index("[") :]
            else:
                result.text_name = fix_test_name(result.name)
        result.requirements = msg["requirements"]
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
        bus_id=labels.get("bus_info", UNKNOWN),
    )


def _parse_gpu(labels: dict, fb_total: float) -> GPU:
    """Parse a single GPU from the DCGM_FI_DEV_FB_TOTAL metric."""
    return GPU(
        number=int(labels["gpu"]),
        model_name=labels["modelName"],
        driver=labels.get("DCGM_FI_DRIVER_VERSION", UNKNOWN),
        vbios=labels.get("DCGM_FI_DEV_VBIOS_VERSION", UNKNOWN),
        bus_id=labels.get("DCGM_FI_DEV_PCI_BUSID", UNKNOWN),
        # Documentation says "MB", but using 10**6 leads to a "12GB" GPU only having 11.2 GiB
        ram=round(fb_total * 1024**2),
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
        elif metric_name == "DCGM_FI_DEV_FB_TOTAL":
            gpus.append(_parse_gpu(labels, value))
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


def _parse_task(msg: dict) -> Task:
    return Task(host=msg["host"], version=msg["version"], git_version=msg["git_version"])


def _parse_correlator_configuration(msg: dict) -> CorrelatorConfiguration:
    tasks = {name: _parse_task(value) for (name, value) in msg["tasks"].items()}
    return CorrelatorConfiguration(description=msg["description"], uuid=UUID(msg["uuid"]), tasks=tasks)


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
        elif line["$report_type"] == "CorrelatorConfiguration":
            test_configuration.correlators.append(_parse_correlator_configuration(line))

        if line["$report_type"] != "TestReport":
            continue
        # _from_json is an "experimental" function.
        report = pytest.TestReport._from_json(line)
        nodeid = report.nodeid
        if not results or results[-1].nodeid != nodeid:
            # It's the first time we've seen this nodeid (otherwise we merge
            # with the existing Result).
            results.append(Result(nodeid, report.location[2], "", ""))
        result = results[-1]
        # The teardown phase has all the log messages, so we ignore the setup and call phases.
        if report.when == "teardown":
            for prop in report.user_properties:
                if prop[0] == "pdf_report_data":
                    assert isinstance(prop[1], list)
                    for msg in prop[1]:
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


def collapse_versions(versions: Set[str]) -> str:
    """Reduce a set of versions to a single string.

    This is appropriate to use when it is expected that there will be at most one.
    """
    if len(versions) == 0:
        return "N/A"
    elif len(versions) == 1:
        return next(iter(versions))
    else:
        return "multiple versions"


def get_requirement_tex(name: str) -> str:
    """Get the LaTeX code snippet describing a requirement.

    If the requirement is unknown, it logs an error and returns filler text.
    """
    path = RESOURCE_PATH / "requirements_text" / f"{name}.tex"
    try:
        return path.read_text()
    except FileNotFoundError:
        logging.error("Text for requirement %s not found", name)
        return rf"\textcolor{{red}}{{Text for requirement {name} not found.}}"


def _format_image(image: str) -> LatexObject:
    """Format a Docker image name for a table.

    These tends to be a very long string, which needs to be wrapped to avoid
    going off the page.
    """
    # The `\strut`s ensure a bit more space around this row.
    image = escape_latex(image)
    image = image.replace("@sha256:", r"@sha256:\\ ")  # Break into two lines
    image = r"\begin{Bflushleft}[t]\strut " + image + r"\strut\end{Bflushleft}"
    return SmallText(NoEscape(image))


def _doc_test_configuration_global(section: Container, test_configuration: TestConfiguration) -> None:
    # Identify versions running on the correlator. Ideally it should be unique
    images = set()
    git_versions = set()
    product_controller_image = "unknown"
    product_controller_git_version = "unknown"
    for correlator in test_configuration.correlators:
        for task_name, task in correlator.tasks.items():
            if task_name == "product_controller":
                product_controller_image = task.version
                product_controller_git_version = task.git_version
            else:
                images.add(task.version)
                git_versions.add(task.git_version)

    with section.create(LongTable(r"|r|l|")) as config_table:
        config_table.add_hline()
        config_table.add_row([MultiColumn(2, align="|c|", data=bold("Test suite"))])
        config_table.add_hline()
        for config_param in test_configuration.params:
            config_table.add_row([config_param.name, config_param.value])
            config_table.add_hline()

        config_table.add_row([MultiColumn(2, align="|c|", data=bold("Correlator"))])
        config_table.add_hline()
        image = collapse_versions(images)
        config_table.add_row("Image", _format_image(image))
        config_table.add_row("Version", collapse_versions(git_versions))
        config_table.add_hline()

        config_table.add_row([MultiColumn(2, align="|c|", data=bold("Product Controller"))])
        config_table.add_hline()
        config_table.add_row("Image", _format_image(product_controller_image))
        config_table.add_row("Version", product_controller_git_version)
        config_table.add_hline()


def _doc_hosts(section: Container, hosts: Mapping[Host, Sequence[str]]) -> None:
    for host, hostnames in hosts.items():
        with section.create(LongTable(r"|r|l|")) as host_table:
            host_table.add_hline()
            for hostname in sorted(hostnames):
                label = Label(Marker(hostname, prefix="host"))
                host_table.add_row([MultiColumn(2, align="|l|", data=[label, hostname])])
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
                host_table.add_row("Bus ID", interface.bus_id)
                host_table.add_hline()
            for gpu in host.gpus:
                host_table.add_row([MultiColumn(2, align="|c|", data=bold(f"GPU {gpu.number}"))])
                host_table.add_hline()
                host_table.add_row("Model", gpu.model_name)
                host_table.add_row("Driver", gpu.driver)
                host_table.add_row("VBIOS", gpu.vbios)
                host_table.add_row("Bus ID", gpu.bus_id)
                host_table.add_row("RAM", f"{gpu.ram / 2**30:.0f} GiB")
                host_table.add_hline()


def _doc_correlators(section: Container, correlators: Sequence[CorrelatorConfiguration]) -> None:
    patterns = [
        ("Product controller", re.compile("product_controller")),
        ("DSim {i}", re.compile(r"sim\.dsim(\d+)\.\d+\.0")),
        ("F-engine {i}", re.compile(r"f\.antenna_channelised_voltage\.(\d+)")),
        ("XB-engine {i}", re.compile(r"xb\.baseline_correlation_products\.(\d+)")),
    ]
    for i, correlator in enumerate(correlators, start=1):
        with section.create(Subsection(f"Configuration {i}")) as subsec:
            subsec.append(Label(Marker(str(correlator.uuid), prefix="correlator")))
            subsec.append(correlator.description)
            with subsec.create(LongTable(r"|l|l|")) as table:
                table.add_hline()
                for name, pattern in patterns:
                    tasks = []
                    for task_name, task in correlator.tasks.items():
                        if match := pattern.fullmatch(task_name):
                            try:
                                tasks.append((int(match.group(1)), task))
                            except IndexError:  # Pattern has no group 1
                                tasks.append((0, task))
                    tasks.sort()
                    for idx, task in tasks:
                        table.add_row(
                            name.format(i=idx),
                            Hyperref(Marker(task.host, prefix="host"), task.host),
                        )
                    table.add_hline()


def _doc_test_configuration(doc: Document, test_configuration: TestConfiguration) -> None:
    with doc.create(Section("Test Configuration")) as config_section:
        _doc_test_configuration_global(config_section, test_configuration)
    with doc.create(Section("Hosts")) as hosts_section:
        _doc_hosts(hosts_section, test_configuration.hosts)
    with doc.create(Section("Correlators")) as correlators_section:
        _doc_correlators(correlators_section, test_configuration.correlators)


def _doc_result_summary(section: Container, results: Sequence[Result]) -> None:
    with section.create(LongTable(r"|r|l|r|")) as summary_table:
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
                [
                    Hyperref(Marker(result.name, prefix="subsec"), result.text_name),
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
    section.append(f"{len(results)} tests run, with {passed} passing and {failed} failing.\n")
    section.append(f"Total test duration: {readable_duration(total_duration)}")


def _doc_requirements(section: Container, requirements: Sequence[str]) -> None:
    if requirements:
        with section.create(Description()) as desc:
            for requirement in requirements:
                desc.add_item(requirement, NoEscape(get_requirement_tex(requirement)))
    else:
        section.append("None")


def _doc_outcome(section: Container, test_configuration: TestConfiguration, result: Result) -> None:
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
            if key == "correlator":
                # Turn the raw UUID into a section reference
                for i, correlator in enumerate(test_configuration.correlators, start=1):
                    if correlator.uuid == UUID(value):
                        marker = Marker(value, prefix="correlator")
                        value = Hyperref(marker, f"Configuration {i}")
                        break
            config_table.add_row([key.capitalize(), value])
            config_table.add_hline()


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

    doc = Document(
        document_options=["11pt", "english", "twoside"],
        inputenc=None,  # katdoc inputs inputenc with specific options, so prevent a clash
    )
    today = date.today()  # TODO: should store inside the JSON
    doc.set_variable("theAuthor", "DSP Team")
    doc.set_variable("docDate", today.strftime("%d %B %Y"))
    doc.preamble.append(NoEscape((RESOURCE_PATH / "preamble.tex").read_text()))
    doc.append(Command("title", "Integration Test Report"))
    doc.append(Command("makekatdocbeginning"))

    _doc_test_configuration(doc, test_configuration)

    with doc.create(Section("Result Summary")) as summary_section:
        _doc_result_summary(summary_section, results)

    with doc.create(Section("Detailed Test Results")) as section:
        for result in results:
            with section.create(Subsection(result.text_name, label=result.name)):
                section.append(NoEscape(rst2latex(result.blurb) + "\n\n"))
                with section.create(Subsubsection("Requirements Verified")) as requirements_sec:
                    _doc_requirements(requirements_sec, result.requirements)
                with section.create(Subsubsection("Results")) as results_sec:
                    _doc_outcome(results_sec, test_configuration, result)
                with section.create(Subsubsection("Procedure", label=False)) as procedure:
                    with section.create(LongTable(r"|l|p{0.7\linewidth}|")) as procedure_table:
                        assert result.start_time is not None
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
    logging.basicConfig()
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
        # Give TeX an order of magnitude more memory than usual. This is needed
        # to handle the reams of text generated for plots.
        os.environ.setdefault("buf_size", "2000000")
        os.environ.setdefault("extra_mem_top", "50000000")
        doc.generate_pdf(args.pdf, compiler="latexmk", compiler_args=["--pdf", "-r", latexmkrc.name])


if __name__ == "__main__":
    main()
