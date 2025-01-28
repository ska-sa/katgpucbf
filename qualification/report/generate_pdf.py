#!/usr/bin/env python3

################################################################################
# Copyright (c) 2022-2024, National Research Foundation (SARAO)
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
import base64
import copy
import itertools
import json
import logging
import os
import pathlib
import re
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Literal, Union
from uuid import UUID

import docutils.writers.latex2e
import pytest
from docutils.core import publish_parts
from pylatex import (
    Command,
    Document,
    Enumerate,
    LongTable,
    MiniPage,
    MultiColumn,
    MultiRow,
    NoEscape,
    Section,
    SmallText,
    StandAloneGraphic,
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

DEFAULT_REPORT_DOC_ID = "E1200-0000-005-dev"
DEFAULT_PROCEDURE_DOC_ID = "E1200-0000-004"
RESOURCE_PATH = pathlib.Path(__file__).parent
UNKNOWN = "unknown"
GROUP_NAMES = {
    "general": "General tests",
    "antenna_channelised_voltage": "Antenna channelised voltage tests",
    "baseline_correlation_products": "Baseline correlation products tests",
    "tied_array_channelised_voltage": "Tied-array channelised voltage tests",
    "demo": "Report demonstration tests",
}


def str_to_bool(s: str) -> bool:
    """Convert a string containing "True" or "False" to boolean."""
    match s:
        case "True":
            return True
        case "False":
            return False
        case _:
            raise ValueError(f"Expected 'True' or 'False', received {s!r}")


def cbf_sort_key(config: dict) -> tuple:
    """Generate a sort key to sort CBFs."""
    # It seems at least some of these get stringified, so ensure they are numeric
    return (
        float(config["bandwidth"]),
        int(config["channels"]),
        int(config["antennas"]),
        int(config["beams"]),
        float(config["integration_time"]),
        int(config["dsims"]),
        int(config["narrowband_decimation"]),
        str_to_bool(config.get("narrowband_vlbi", "False")),
    )


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


def outcome_color(outcome: str) -> str:
    """Get LaTeX color to use for a given outcome."""
    if outcome == "passed":
        return "green"
    elif outcome == "failed":
        return "red"
    else:
        return "black"


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
    # If these are not specifically set, docutils issues deprecation warnings
    # indicating that the defaults will change. These settings are the values
    # that will become the default in future.
    overrides = {
        "use_latex_citations": True,
        "legacy_column_widths": False,
    }
    return publish_parts(source=text, writer=writer, settings_overrides=overrides)["body"]


@dataclass
class Detail:
    """A message logged by ``pdf_report.detail``."""

    message: str
    timestamp: float


@dataclass
class Failure:
    """A failure logged by ``pdf_report.failure``."""

    message: str
    timestamp: float


@dataclass
class RawFigure:
    """A figure created by ``pdf_report.raw_figure``."""

    code: str


@dataclass
class Figure:
    """A figure created by ``pdf_report.figure``."""

    content: bytes
    type: str


Item = Union[Detail, Failure, Figure, RawFigure]  # Any information provided inside a ``pdf_report.step``


@dataclass
class Step:
    """A step created by ``pdf_report.step``."""

    message: str
    items: list[Item] = field(default_factory=list)


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
    cpus: tuple[str, ...]
    interfaces: tuple[Interface, ...]
    gpus: tuple[GPU, ...]
    ram: int | None
    product_name: str = UNKNOWN
    board_name: str = UNKNOWN
    bios_version: str = UNKNOWN


@dataclass
class Task:
    """A single Mesos task within a CBF."""

    host: str
    version: str  # Docker image
    git_version: str
    interfaces: dict[str, str]


@dataclass
class CBFConfiguration:
    """Configuration information for a CBF instance."""

    mode_config: dict
    uuid: UUID
    tasks: dict[str, Task]  # Empty if the CBF failed
    error: str | None  # None if the CBF started successfully
    index: int = -1  # Position in the list of CBFs
    title: str = ""  # Text to use for section name, links etc in the document

    def mode_str(self, *, expand: bool = False) -> str:
        """Generate a description string for the CBF.

        The :attr:`mode_config` is expected to be in the format of, e.g.:

        .. code:: python
            {
                "antennas": "4",
                "channels": "8192",
                "bandwidth": "856",
                "band": "L",
                "integration_time": "0.5",
                "dsims": "4",
                "beams": "4",
                "narrowband_decimation": "8",
                "narrowband_vlbi": "False"
            }

        If `expand` is True, it will return a 'long description' as a sentence:

        - 4 antennas, 8192 channels, 4 beams, L-band, 0.5s integrations, 4 dsims

        If `expand` is False, it will return a MeerKAT config mode string of:

        - bc8n856M8k

        Parameters
        ----------
        config
            A dictionary in the format described above.
        expand
            To indicate whether a full-sentence description should be returned
            or just the mode string should be generated.

        Returns
        -------
        Short or long description of CBF mode.
        """
        narrowband_decimation = int(self.mode_config["narrowband_decimation"])
        # Older report files won't contain the narrowband_vlbi key
        narrowband_vlbi = str_to_bool(self.mode_config.get("narrowband_vlbi", "False"))
        if expand:
            # Long description required
            parts = [
                f'{self.mode_config["antennas"]} antennas',
                f'{self.mode_config["channels"]} channels',
                f'{self.mode_config["beams"]} beams',
                f'{self.mode_config["band"]}-band',
                f'{self.mode_config["integration_time"]}s integrations',
                f'{self.mode_config["dsims"]} dsims',
            ]
            if narrowband_decimation > 1:
                parts.append(f"1/{narrowband_decimation} narrowband")
                if narrowband_vlbi:
                    parts[-1] += " (VLBI)"
            config_mode = ", ".join(parts) + "."
        else:
            antpols = int(self.mode_config["antennas"]) * 2
            chans = int(self.mode_config["channels"]) // 1000
            mode = "bc" if int(self.mode_config["beams"]) > 0 else "c"
            if narrowband_decimation > 1 and narrowband_vlbi:
                mode += "v"
            config_mode = f'{mode}{antpols}n{self.mode_config["bandwidth"]}M{chans}k'

        return config_mode

    @property
    def marker(self) -> Marker:
        """Marker for linking to this CBF."""
        return Marker(str(self.uuid), prefix="cbf")

    @property
    def link(self) -> Hyperref:
        """Hyperlink to this CBF."""
        return Hyperref(self.marker, self.title)


@dataclass
class TestConfiguration:
    """Global configuration of the test."""

    params: dict[str, str] = field(default_factory=dict)
    hosts: dict[Host, list[str]] = field(default_factory=lambda: defaultdict(list))  # Hostnames for each config
    cbfs: list[CBFConfiguration] = field(default_factory=list)
    cbf_index: dict[UUID, CBFConfiguration] = field(default_factory=dict)


@dataclass
class Result:
    """A single test execution.

    This combines the setup, call and teardown phases, which appear as separate
    lines in the report.json file.
    """

    nodeid: str
    filename: str
    lineno: int | None
    name: str  # Name by which the test is known to pytest (including parameters)
    text_name: str = ""  # Name shown in the document (excluding parameters)
    parameters: str = ""  # Parameter part of `name`
    subtitle: str = ""  # Title of the report within the ResultSet
    blurb: str = ""
    requirements: list[str] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)
    outcome: Literal["passed", "failed", "skipped", "xfail"] = "failed"
    xfail_reason: str | None = None
    failure_messages: list[str] = field(default_factory=list)
    start_time: float | None = None
    duration: float = 0.0
    config: dict = field(default_factory=dict)
    cbf: CBFConfiguration | None = None

    @property
    def base_nodeid(self) -> str:
        """Get the nodeid with parameters removed."""
        pos = self.nodeid.find("[")
        if pos == -1:
            return self.nodeid
        else:
            return self.nodeid[:pos]

    @property
    def marker(self) -> Marker:
        """Marker for linking to this result."""
        return Marker(self.nodeid, "ssubsec")

    @property
    def link(self) -> Hyperref:
        """Hyperlink to this result (including only the subtitle)."""
        return Hyperref(self.marker, self.subtitle)

    @property
    def group(self) -> str:
        """Top-level directory of the test."""
        return self.filename.split("/")[0]


class ResultSet:
    """A collection of results with the same :attr:`Result.base_nodeid`."""

    def __init__(self, results: Sequence[Result]) -> None:
        self.results = list(results)
        assert self.results, "results must not be empty"
        self.base_nodeid = results[0].base_nodeid
        self.text_name = results[0].text_name
        self.blurb = results[0].blurb
        self.duration = 0.0
        self.outcome_counts = {"passed": 0, "failed": 0, "skipped": 0, "xfail": 0}
        self.requirements: list[str] = []
        self.group = self.results[0].group
        for result in self.results:
            assert result.base_nodeid == self.base_nodeid, "base_nodeids do not match"
            self.text_name = self._merge(self.text_name, result.text_name, "text names")
            self.blurb = self._merge(self.blurb, result.blurb, "blurbs")
            self.duration += result.duration
            self.outcome_counts[result.outcome] += 1
            for requirement in result.requirements:
                if requirement not in self.requirements:
                    self.requirements.append(requirement)

    def _merge(self, a: str, b: str, name: str) -> str:
        """Combine a property from two results.

        If either is blank, the other is used. If both are non-blank and they
        don't match, raise a :exc:`ValueError`.
        """
        if a != b and a and b:
            raise ValueError(f"Inconsistent {name} for tests in ResultSet {self.base_nodeid}")
        return a or b

    @property
    def marker(self) -> Marker:
        """Marker for linking to this ResultSet."""
        return Marker(self.base_nodeid, prefix="subsec")

    @property
    def link(self) -> Hyperref:
        """Hyperlink to this ResultSet."""
        return Hyperref(self.marker, self.text_name)

    @property
    def qualified_link(self) -> Hyperref:
        """Hyperlink to this ResultSet, with a fully-qualified name."""
        return Hyperref(self.marker, GROUP_NAMES.get(self.group, self.group) + " / " + self.text_name)


def _parse_item(item: dict) -> Item:
    """Parse a single item from the items of a step."""
    msg_type = item["$msg_type"]
    if msg_type == "detail":
        return Detail(item["message"], item["timestamp"])
    elif msg_type == "failure":
        return Failure(item["message"], item["timestamp"])
    elif msg_type == "figure":
        return RawFigure(item["code"])
    elif msg_type == "binary_figure":
        return Figure(base64.standard_b64decode(item["content"]), item["type"])
    else:
        raise ValueError(f"Do not know how to parse item $msg_type of {msg_type!r}")


def _parse_report_data(result: Result, msg: dict) -> None:
    """Parse a single entry in the ``pdf_report_data`` user property."""
    assert isinstance(msg, dict)
    msg_type = msg["$msg_type"]
    if msg_type == "step":
        items = [_parse_item(item) for item in msg["items"]]
        result.steps.append(Step(msg["message"], items))
    elif msg_type == "test_info":
        if not result.blurb:
            result.blurb = msg["blurb"]
        if result.start_time is None:
            result.start_time = float(msg["test_start"])
        if not result.text_name:
            if "[" in result.name:
                pos = result.name.index("[")
                base_name = result.name[:pos]
                result.parameters = result.name[pos + 1 : -1]
            else:
                result.parameters = ""
                base_name = result.name
            if "test_name" in msg:
                # Replace the Python identifier, but keep any parameters
                result.text_name = msg["test_name"]
            else:
                result.text_name = fix_test_name(base_name)
        result.requirements = msg["requirements"]
    elif msg_type == "config":
        msg = dict(msg)  # Avoid mutating the caller's copy
        msg.pop("$msg_type")  # We don't need this.
        if "correlator" in msg:  # Backwards compatibility: handle this old name
            msg["cbf"] = msg.pop("correlator")
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


def _parse_host(msg: dict) -> tuple[str, Host]:
    """Parse a single host configuration line."""
    kernel = UNKNOWN
    cpus: dict[int, str] = {}  # Indexed by package
    interfaces = []
    gpus = []
    kwargs = {}
    ram: int | None = None

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
            # Round ram to the nearest MiB. For unknown reasons, some machines
            # have a few KiB more or less RAM available than others even when
            # the hardware is identical, and this rounding helps coalesce
            # them in the report.
            ram = round(value / 2**20) * 2**20
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
    return Task(
        host=msg["host"],
        interfaces=msg.get("interfaces", {}),
        version=msg["version"],
        git_version=msg["git_version"],
    )


def _parse_cbf_configuration(msg: dict) -> CBFConfiguration:
    tasks = {name: _parse_task(value) for (name, value) in msg.get("tasks", {}).items()}
    mode_config = msg["mode_config"].copy()
    mode_config.setdefault("beams", "0")  # For backwards compatibility with old inputs
    error = msg.get("error")
    return CBFConfiguration(mode_config=mode_config, uuid=UUID(msg["uuid"]), tasks=tasks, error=error)


def _fix_result_cbfs(test_configuration: TestConfiguration, results: Iterable[Result]) -> None:
    """Populate the 'cbf' field in the results by matching up the UUIDs."""
    for result in results:
        try:
            result.cbf = test_configuration.cbf_index[UUID(result.config["cbf"])]
        except KeyError:
            result.cbf = None


def parse(input_data: list[dict]) -> tuple[TestConfiguration, list[Result]]:
    """Parse the data written by pytest-reportlog.

    Parameters
    ----------
    input_data
        A Python list, which should have been loaded from pytest-reportlog's
        JSON output.

    Returns
    -------
    :class:`TestConfiguration` and :class:`Result` objects representing the
    test configuration and the results of all the tests logged in the JSON input.
    """
    test_configuration = TestConfiguration()
    results: list[Result] = []
    for line in input_data:
        if line["$report_type"] == "TestConfiguration":
            test_configuration.params = dict(line)
            test_configuration.params.pop("$report_type")
        elif line["$report_type"] == "HostConfiguration":
            hostname, host = _parse_host(line)
            test_configuration.hosts[host].append(hostname)
        elif line["$report_type"] in ["CorrelatorConfiguration", "CBFConfiguration"]:
            # "CorrelatorConfiguration" is for backwards compatibility
            cbf = _parse_cbf_configuration(line)
            test_configuration.cbfs.append(cbf)
            test_configuration.cbf_index[cbf.uuid] = cbf

        if line["$report_type"] != "TestReport":
            continue
        # _from_json is an "experimental" function. It also mutates its argument,
        # so we have to copy it to avoid altering input_data.
        report = pytest.TestReport._from_json(copy.deepcopy(line))
        nodeid = report.nodeid
        if not results or results[-1].nodeid != nodeid:
            # It's the first time we've seen this nodeid (otherwise we merge
            # with the existing Result).
            results.append(Result(nodeid, report.location[0], report.location[1], report.location[2]))
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
            if report.outcome == "skipped" and getattr(report, "wasxfail", None) is not None:
                result.outcome = "xfail"
                result.xfail_reason = report.wasxfail
        # The test duration will be the sum of setup, call and teardown.
        result.duration += report.duration
        if report.longrepr is not None:
            # Strip out ANSI color escape sequences
            failure_message = re.sub("\x1b\\[.*?m", "", report.longreprtext)
            result.failure_messages.append(failure_message)

    _fix_result_cbfs(test_configuration, results)
    return test_configuration, results


def sort_results(test_configuration: TestConfiguration, results: list[Result]) -> None:
    """Sort CBFs and results (in-place)."""
    test_configuration.cbfs.sort(key=lambda cbf: cbf_sort_key(cbf.mode_config))
    for i, cbf in enumerate(test_configuration.cbfs):
        cbf.index = i
        cbf.title = f"Configuration {i + 1}: {cbf.mode_str()}"

    def result_key(result: Result) -> tuple:
        try:
            cbf_index = test_configuration.cbf_index[UUID(result.config["cbf"])].index
        except KeyError:
            cbf_index = -1
        # This is not guaranteed to be a unique key, if there are parameters
        # that don't affect the choice of correlator. In that case, we just
        # stick with the original order chosen by pytest (sorting is stable
        # in Python).
        return result.filename, result.lineno, result.base_nodeid, cbf_index

    results.sort(key=result_key)


def collate_results(results: Iterable[Result]) -> list[ResultSet]:
    """Group Results into ResultSets.

    The :attr:`Result.subtitle` attribute is also populated here.
    """
    result_lists: dict[str, list[Result]] = {}
    for result in results:
        result_lists.setdefault(result.base_nodeid, []).append(result)
    out = []
    for result_list in result_lists.values():
        result_set = ResultSet(result_list)
        # Check whether the CBF titles alone give unique subtitles
        cbf_titles = {result.cbf.title for result in result_list if result.cbf is not None}
        if len(cbf_titles) == len(result_list):
            for result in result_list:
                assert result.cbf is not None
                result.subtitle = result.cbf.title
        else:
            # Have to use parameters to disambiguate
            for result in result_list:
                parts = [result.cbf.title] if result.cbf is not None else []
                if result.parameters:
                    parts.append(result.parameters)
                if not parts:
                    parts = ["Result"]  # Handle case of unparametrised test with no CBF
                result.subtitle = " ".join(parts)
        out.append(result_set)
    return out


def extract_requirements_verified(result_sets: Iterable[ResultSet]) -> dict[str, list[ResultSet]]:
    """Extract a matrix of requirements verified from the list of results.

    Parameters
    ----------
    result_sets
        Result sets as generated by :func:`collate_results`.

    Returns
    -------
        A dictionary with the requirements as keys and list of the ResultSets which
        verify that requirement as the values.
    """
    requirements_verified = defaultdict(list)
    for result_set in result_sets:
        for requirement in result_set.requirements:
            requirements_verified[requirement].append(result_set)
    return requirements_verified


def fix_test_name(test_name: str) -> str:
    """Change a test's name from a pytest one to a more human-friendly one."""
    return " ".join([word.capitalize() for word in test_name.split("_") if word != "test"])


def collapse_versions(versions: set[str]) -> str:
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


def _doc_preamble(doc: Document, doc_id: str, make_report: bool) -> None:
    """Set up the preamble of the document."""
    doc.packages.add(Package("hyperref", "colorlinks,linkcolor=blue"))
    today = date.today()  # TODO: should store inside the JSON
    doc.set_variable("theAuthor", "DSP Team")
    doc.set_variable("docDate", today.strftime("%d %B %Y"))
    doc.set_variable("docId", doc_id)
    doc.set_variable("docType", f"Qualification Test {'Report' if make_report else 'Procedure'}")
    doc.preamble.append(NoEscape((RESOURCE_PATH / "preamble.tex").read_text()))
    doc.append(Command("title", f"MeerKAT Extension CBF Qualification Test {'Report' if make_report else 'Procedure'}"))
    doc.append(Command("makekatdocbeginning"))


def _doc_test_configuration_global(section: Container, test_configuration: TestConfiguration) -> None:
    # Identify versions running on the CBF. Ideally it should be unique
    images = set()
    git_versions = set()
    product_controller_image = "unknown"
    product_controller_git_version = "unknown"
    for cbf in test_configuration.cbfs:
        for task_name, task in cbf.tasks.items():
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
        for config_key, config_value in test_configuration.params.items():
            config_table.add_row([config_key, config_value])
            config_table.add_hline()

        config_table.add_row([MultiColumn(2, align="|c|", data=bold("CBF"))])
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


def _doc_cbfs(section: Container, cbfs: Sequence[CBFConfiguration]) -> None:
    patterns = [
        ("Product controller", re.compile("product_controller")),
        ("DSim {i}", re.compile(r"sim\.dsim(\d+)\.\d+\.0")),
        ("F-engine {i}", re.compile(r"f\.(?:wideband-)?antenna-channelised-voltage\.(\d+)")),
        ("XB-engine {i}", re.compile(r"xb\.baseline-correlation-products\.(\d+)")),
        ("WB XB-engine {i}", re.compile(r"xb\.wideband-baseline-correlation-products\.(\d+)")),
    ]
    for cbf in cbfs:
        with section.create(Subsection(cbf.title)) as subsec:
            subsec.append(Label(Marker(str(cbf.uuid), prefix="cbf")))
            subsec.append(cbf.mode_str(expand=True))
            if cbf.error is None:
                with subsec.create(LongTable(r"|l|l|l|")) as table:
                    table.add_hline()
                    for name, pattern in patterns:
                        tasks = []
                        for task_name, task in cbf.tasks.items():
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
                                ", ".join(task.interfaces.values()),
                            )
                        if tasks:
                            table.add_hline()
            else:
                subsec.append("\n\n")
                subsec.append(TextColor("red", "CBF failed to start"))
                with subsec.create(LstListing()) as failure_message:
                    failure_message.append(cbf.error)


def _doc_requirements_verified(doc: Document, requirements_verified: dict[str, list[ResultSet]]) -> None:
    with doc.create(Section("Requirements Verified")) as req_section:
        with req_section.create(LongTable(r"|l|l|")) as table:
            table.add_hline()
            table.add_row([bold("Requirement"), bold("Tests verifying requirement")])
            table.add_hline()
            for req, result_sets in sorted(requirements_verified.items(), key=lambda item: item[0]):
                table.add_row(MultiRow(len(result_sets), data=req), result_sets[0].qualified_link)
                for result_set in result_sets[1:]:
                    table.add_hline(2)
                    table.add_row("", result_set.qualified_link)
                table.add_hline()


def _doc_test_configuration(doc: Document, test_configuration: TestConfiguration) -> None:
    with doc.create(Section("Test Configuration")) as config_section:
        _doc_test_configuration_global(config_section, test_configuration)
    with doc.create(Section("Hosts")) as hosts_section:
        _doc_hosts(hosts_section, test_configuration.hosts)
    with doc.create(Section("CBFs")) as cbfs_section:
        _doc_cbfs(cbfs_section, test_configuration.cbfs)


def _doc_result_summary(section: Container, result_sets: Sequence[ResultSet]) -> None:
    for key, group in itertools.groupby(result_sets, lambda result_set: result_set.group):
        with section.create(Subsection(GROUP_NAMES.get(key, key))) as group_section:
            with group_section.create(LongTable(r"|l|r|c|c|c|")) as summary_table:
                total_duration: float = 0.0
                passed = 0
                failed = 0
                skipped = 0
                total = 0
                summary_table.add_hline()
                summary_table.add_row(
                    [
                        bold("Test"),
                        bold("Duration"),
                        bold("Passed"),
                        bold("Failed"),
                        bold("Skipped"),
                    ]
                )
                summary_table.add_hline()
                for result_set in group:
                    set_passed = result_set.outcome_counts["passed"]
                    set_failed = result_set.outcome_counts["failed"] + result_set.outcome_counts["xfail"]
                    set_skipped = result_set.outcome_counts["skipped"]
                    summary_table.add_row(
                        [
                            Hyperref(Marker(result_set.base_nodeid, prefix="subsec"), result_set.text_name),
                            readable_duration(result_set.duration),
                            str(set_passed) if set_passed else "",
                            TextColor("red", str(set_failed)) if set_failed else "",
                            str(set_skipped) if set_skipped else "",
                        ]
                    )
                    summary_table.add_hline()
                    total_duration += result_set.duration
                    passed += set_passed
                    failed += set_failed
                    skipped += set_skipped
                    total += len(result_set.results)
            group_section.append(f"{total} tests run, with {passed} passing, {failed} failing and {skipped} skipped.\n")
            section.append(f"Total test duration: {readable_duration(total_duration)}")


def _doc_requirements(section: Container, requirements: Sequence[str]) -> None:
    if requirements:
        with section.create(Description()) as desc:
            for requirement in requirements:
                desc.add_item(requirement, NoEscape(get_requirement_tex(requirement)))
    else:
        section.append("None")


def _doc_outcomes(section: Container, result_set: ResultSet) -> None:
    with section.create(LongTable("|r|r|l|l|")) as table:
        table.add_hline()
        table.add_row([bold("Start time"), bold("Duration"), bold("Configuration"), bold("Outcome")])
        table.add_hline()
        for result in result_set.results:
            if result.start_time is not None:
                start_time = datetime.fromtimestamp(float(result.start_time), timezone.utc).strftime("%T")
            else:
                start_time = ""
            table.add_row(
                [
                    start_time,
                    readable_duration(result.duration),
                    result.link,
                    TextColor(outcome_color(result.outcome), result.outcome.upper()),
                ]
            )
            table.add_hline()
        table.add_row([bold("Total"), bold(readable_duration(result_set.duration)), "", ""])
        table.add_hline()


def _doc_outcome(section: Container, result: Result) -> None:
    section.append("Outcome: ")
    section.append(TextColor(outcome_color(result.outcome), result.outcome.upper()))
    if result.xfail_reason:
        section.append(NoEscape(r"\ "))
        section.append(f"({result.xfail_reason})")
    section.append(Command("hspace", "1cm"))
    if result.start_time is not None:
        section.append(
            f"Test start time: {datetime.fromtimestamp(float(result.start_time), timezone.utc).strftime('%F %T')}"
        )
        section.append(Command("hspace", "1cm"))
    section.append(f"Duration: {readable_duration(result.duration)}\n")

    with section.create(LongTable(r"|l|p{0.4\linewidth}|")) as config_table:
        config_table.add_hline()
        config_table.add_row((MultiColumn(2, align="|c|", data=bold("Test Configuration")),))
        config_table.add_hline()
        for key, value in result.config.items():
            if key == "cbf" and result.cbf is not None:
                key = "CBF"
                value = result.cbf.link
            else:
                key = key.capitalize()
            config_table.add_row([key, value])
            config_table.add_hline()


def _doc_result(section: Container, result: Result, tmp_dir: pathlib.Path, figure_ids: Iterator[int]) -> None:
    _doc_outcome(section, result)
    with section.create(LongTable(r"|l|p{0.7\linewidth}|")) as procedure_table:
        procedure_table.add_hline()
        for step in result.steps:
            assert result.start_time is not None
            procedure_table.add_row((MultiColumn(2, align="|l|", data=bold(step.message)),))
            procedure_table.add_hline()
            for item in step.items:
                if isinstance(item, Detail):
                    procedure_table.add_row(
                        [
                            f"{readable_duration(item.timestamp - result.start_time)}",
                            item.message,
                        ]
                    )
                elif isinstance(item, Failure):
                    # add_row doesn't seem to have a way to put
                    # more than one sequential command into a
                    # cell. Just construct the cell by hand.
                    # Without the Bflushleft there is (for some
                    # reason) a lot of extra vertical space.
                    cell = r"\color{red}\begin{Bflushleft}\begin{lstlisting}"
                    cell += "\n" + item.message + "\n"
                    cell += r"\end{lstlisting}\end{Bflushleft}"
                    procedure_table.add_row(
                        [
                            f"{readable_duration(item.timestamp - result.start_time)}",
                            NoEscape(cell),
                        ]
                    )
                elif isinstance(item, RawFigure):
                    filename = f"figure{next(figure_ids):04}.tex"
                    (tmp_dir / filename).write_text(item.code)
                    mp = MiniPage(width=NoEscape(r"\textwidth"))
                    mp.append(NoEscape(r"\center"))
                    mp.append(NoEscape(r"\input{" + filename + "}"))
                    procedure_table.add_row((MultiColumn(2, align="|c|", data=mp),))
                elif isinstance(item, Figure):
                    filename = f"figure{next(figure_ids):04}.{item.type}"
                    (tmp_dir / filename).write_bytes(item.content)
                    mp = MiniPage(width=NoEscape(r"\textwidth"))
                    mp.append(NoEscape(r"\center"))
                    # pagebox=artbox is needed because the images generated
                    # by matplotlib lack a CropBox and without it, graphicx
                    # will not fall back to anything else to determine the size.
                    mp.append(StandAloneGraphic(filename, "pagebox=artbox"))
                    procedure_table.add_row((MultiColumn(2, align="|c|", data=mp),))
                procedure_table.add_hline()

    if result.cbf is not None and result.cbf.error is not None:
        # In this case we'll already have details about the failure under
        # the heading for the CBF, so don't repeat the whole thing here.
        section.append(TextColor("red", f"CBF failed to start: {result.cbf.error.splitlines()[-1]}"))
    elif result.failure_messages:
        with section.create(LstListing()) as failure_message:
            for message in result.failure_messages:
                failure_message.append(message)


def _doc_result_set(
    section: Container, result_set: ResultSet, tmp_dir: pathlib.Path, figure_ids: Iterator[int], make_report: bool
) -> None:
    section.append(NoEscape(rst2latex(result_set.blurb) + "\n\n"))
    with section.create(Subsubsection("Requirements Verified")) as requirements_sec:
        _doc_requirements(requirements_sec, result_set.requirements)

    if make_report:
        with section.create(Subsubsection("Results")) as outcomes_sec:
            _doc_outcomes(outcomes_sec, result_set)
        for result in result_set.results:
            with section.create(Subsubsection(result.subtitle, label=result.marker.name)) as result_sec:
                _doc_result(result_sec, result, tmp_dir, figure_ids)
    else:
        with section.create(Subsubsection("Procedure")) as procedure:
            canonical_steps = [step.message for step in result_set.results[0].steps]
            with procedure.create(Enumerate()) as test_steps:
                for step in canonical_steps:
                    test_steps.add_item(step)
            # Check whether all results in the set have the same steps.
            all_same = True
            for result in result_set.results:
                steps = [step.message for step in result.steps]
                if steps != canonical_steps:
                    all_same = False
            if not all_same:
                procedure.append("Note: this procedure is an example. The exact steps will vary between modes.")


def document_from_list(result_list: list, doc_id: str, tmp_dir: pathlib.Path, *, make_report=True) -> Document:
    """Take a test result and generate a :class:`pylatex.Document` for a report.

    Parameters
    ----------
    result_list
        A list containing the test results.
    doc_id
        The document number that will be embedded into the PDF output.
    tmp_dir
        Temporary directory in which to write ancillary files.
    make_report
        Make a report rather than a procedure if true, a procedure rather than
        a report if false.

    Returns
    -------
    doc
        A PyLatex document
    """
    test_configuration, results = parse(result_list)
    sort_results(test_configuration, results)
    result_sets = collate_results(results)
    requirements_verified = extract_requirements_verified(result_sets)

    doc = Document(
        document_options=["11pt", "english", "twoside"],
        inputenc=None,  # katdoc inputs inputenc with specific options, so prevent a clash
    )
    _doc_preamble(doc, doc_id, make_report)
    _doc_requirements_verified(doc, requirements_verified)

    if make_report:
        _doc_test_configuration(doc, test_configuration)

        with doc.create(Section("Result Summary")) as summary_section:
            _doc_result_summary(summary_section, result_sets)

    figure_ids = itertools.count(1)
    for key, group in itertools.groupby(result_sets, lambda result_set: result_set.group):
        with doc.create(Section(GROUP_NAMES.get(key, key))) as section:
            for result_set in group:
                with section.create(Subsection(result_set.text_name, label=result_set.marker.name)) as result_set_sec:
                    _doc_result_set(result_set_sec, result_set, tmp_dir, figure_ids, make_report)

    return doc


def test_image_commit(result_list: list) -> str:
    """Extract the git commit ID of the test image from the test results.

    Parameters
    ----------
    result_list
        A list containing the test results.

    Returns
    -------
    commit_id : str
        The git commit ID of the katgpucbf image.
    """
    test_configuration = parse(result_list)[0]
    git_versions = set()
    for cbf in test_configuration.cbfs:
        for task_name, task in cbf.tasks.items():
            if task_name != "product_controller":
                git_versions.add(task.git_version)
    return collapse_versions(git_versions)


def list_from_json(input_data: str) -> list:
    """Take a test result and generate a list.

    Parameters
    ----------
    input_data
        A path to the report file generated by :samp:`pytest --report-log={filename}`.

    Returns
    -------
    result_list : list
        A list containing the test results.
    """
    result_list = []
    with open(input_data) as fp:
        for line in fp:
            result_list.append(json.loads(line))
    return result_list


def main():
    """Convert a JSON report to a PDF."""
    logging.basicConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Output of pytest --report-log=...")
    parser.add_argument("pdf", help="PDF file to write")
    parser.add_argument("-c", "--commit-id", action="store_true", help="Output commit ID of katgpucbf image")
    parser.add_argument(
        "--generate-procedure-doc",
        action="store_false",
        dest="make_report",
        help="Generate a procedure document, rather than the default report",
    )
    parser.add_argument(
        "--doc-id",
        help="Document number to write to the resulting PDF",
        type=str,
    )
    args = parser.parse_args()
    if args.doc_id is None:
        args.doc_id = DEFAULT_REPORT_DOC_ID if args.make_report else DEFAULT_PROCEDURE_DOC_ID
    result_list = list_from_json(args.input)
    if args.commit_id:
        print(test_image_commit(result_list))
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = pathlib.Path(tmp_dir)
        doc = document_from_list(result_list, args.doc_id, make_report=args.make_report, tmp_dir=tmp_dir)
        if args.pdf.endswith(".pdf"):
            args.pdf = args.pdf[:-4]  # Strip .pdf suffix, because generate_pdf appends it
        latexmkrc = tmp_dir / "latexmkrc"
        # TODO: latexmk uses Perl, which has different string parsing to
        # Python. If the path contains both a single quote and a special
        # symbol it will not produce a valid Perl string.
        parent_dir = str(RESOURCE_PATH)
        latexmkrc.write_text(f"ensure_path('TEXINPUTS', {parent_dir!r}, {str(tmp_dir)!r})\n")
        # Give TeX an order of magnitude more memory than usual. This is needed
        # to handle the reams of text generated for plots.
        os.environ.setdefault("buf_size", "2000000")
        os.environ.setdefault("extra_mem_top", "50000000")
        doc.generate_pdf(args.pdf, compiler="latexmk", compiler_args=["--pdf", "-r", str(latexmkrc)])


if __name__ == "__main__":
    main()
