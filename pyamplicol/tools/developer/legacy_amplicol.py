#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Run the pinned Fortran AmpliCol as a developer-only physics oracle."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import tempfile
import tomllib
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / "dependencies" / "release-lock.toml"
DEFAULT_REPOSITORY = ROOT / "dependencies" / "checkouts" / "legacy-amplicol"
DEFAULT_FIXTURE = ROOT / "tests" / "fixtures" / "reference" / "physics-v1.json"

_PDG_BY_NAME = {
    "g": 21,
    "d": 1,
    "u": 2,
    "s": 3,
    "c": 4,
    "b": 5,
    "t": 6,
    "d~": -1,
    "u~": -2,
    "s~": -3,
    "c~": -4,
    "b~": -5,
    "t~": -6,
    "a": 22,
    "z": 23,
    "w+": 24,
    "w-": -24,
    "e+": -11,
    "e-": 11,
    "mu+": -13,
    "mu-": 13,
    "ta+": -15,
    "ta-": 15,
    "ve": 12,
    "ve~": -12,
    "vm": 14,
    "vm~": -14,
    "vt": 16,
    "vt~": -16,
    "h": 25,
}
_VALUE_RE = re.compile(
    r"^AMPICOL_COLOR_PROBE_VALUE\s+\S+\s+\d+\s+\d+\s+([+\-0-9.Ee]+)$",
    re.MULTILINE,
)
_COMPONENT_RE = re.compile(
    r"^AMPICOL_COLOR_PROBE_COMPONENTS\s+"
    r"([+\-0-9.Ee]+)\s+([+\-0-9.Ee]+)\s+([+\-0-9.Ee]+)$",
    re.MULTILINE,
)
_INTEGER_LABELS = {
    "currents": "AMPICOL_COLOR_PROBE_CURRENTS",
    "vertices": "AMPICOL_COLOR_PROBE_VERTICES",
    "amplitudes": "AMPICOL_COLOR_PROBE_AMPLITUDES",
    "color_orders": "AMPICOL_COLOR_PROBE_COLOR_ORDERS",
}


class LegacyOracleError(RuntimeError):
    """The independent Fortran oracle could not be prepared or evaluated."""


@dataclass(frozen=True)
class ProcessEntry:
    group: int
    integral: int
    process_pdgs: tuple[int, ...]
    color_order: tuple[int, ...]


@dataclass(frozen=True)
class ProbeResult:
    value: float
    components: tuple[float, float, float]
    currents: int
    vertices: int
    amplitudes: int
    color_orders: int


def _run(
    command: Sequence[str],
    *,
    cwd: Path,
    capture: bool = True,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=capture,
    )
    if completed.returncode != 0:
        detail = "\n".join(
            part for part in (completed.stdout, completed.stderr) if part
        )
        raise LegacyOracleError(
            f"command exited with {completed.returncode}: {' '.join(command)}"
            + (f"\n{detail}" if detail else "")
        )
    return completed


def _release_lock() -> dict[str, Any]:
    with LOCK.open("rb") as stream:
        return tomllib.load(stream)


def expected_revision() -> str:
    return str(_release_lock()["legacy_amplicol"]["revision"])


def managed_patches() -> tuple[Path, ...]:
    lock = _release_lock()
    return tuple(
        ROOT / "dependencies" / str(entry["path"])
        for entry in lock["patches"]
        if entry["dependency"] == "legacy-amplicol"
    )


def validate_checkout(repository: Path) -> None:
    repository = repository.resolve()
    if not (repository / ".git").exists():
        raise LegacyOracleError(
            f"legacy AmpliCol checkout is absent: {repository}; run `just dev-install`"
        )
    revision = _run(["git", "rev-parse", "HEAD"], cwd=repository).stdout.strip()
    if revision != expected_revision():
        raise LegacyOracleError(
            f"legacy AmpliCol is at {revision}, expected {expected_revision()}"
        )
    for patch in managed_patches():
        completed = subprocess.run(
            ["git", "apply", "--reverse", "--check", str(patch)],
            cwd=repository,
            text=True,
            capture_output=True,
        )
        if completed.returncode != 0:
            raise LegacyOracleError(
                f"managed legacy patch is not applied: {patch.name}"
            )


def build_color_probe(repository: Path, *, jobs: int) -> None:
    validate_checkout(repository)
    _run(
        ["make", f"-j{max(1, jobs)}", "amplicol_color_probe"],
        cwd=repository,
        capture=False,
    )


def parse_process_file(path: Path) -> tuple[ProcessEntry, ...]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise LegacyOracleError(f"empty process file: {path}")
    try:
        n_external, n_unique = (int(value) for value in lines[0].split())
    except (ValueError, TypeError) as error:
        raise LegacyOracleError(f"invalid process header in {path}") from error
    cursor = 1 + n_unique

    def next_nonempty() -> str:
        nonlocal cursor
        while cursor < len(lines) and not lines[cursor].strip():
            cursor += 1
        if cursor >= len(lines):
            raise LegacyOracleError(f"truncated process file: {path}")
        value = lines[cursor].strip()
        cursor += 1
        return value

    try:
        n_groups = int(next_nonempty())
        entries: list[ProcessEntry] = []
        for _ in range(n_groups):
            header = [int(value) for value in next_nonempty().split()]
            if len(header) != 3 + n_external:
                raise ValueError("invalid group header width")
            group, n_integrals, _max_channels = header[:3]
            for integral in range(1, n_integrals + 1):
                tokens = next_nonempty().split()
                n_channels = int(tokens[0])
                process_start = 1 + n_channels
                process_end = process_start + n_external
                order_end = process_end + n_external
                if len(tokens) != order_end + 1:
                    raise ValueError("invalid process row width")
                entries.append(
                    ProcessEntry(
                        group=group,
                        integral=integral,
                        process_pdgs=tuple(
                            int(value) for value in tokens[process_start:process_end]
                        ),
                        color_order=tuple(
                            int(value) for value in tokens[process_end:order_end]
                        ),
                    )
                )
    except ValueError as error:
        raise LegacyOracleError(f"invalid process file {path}: {error}") from error
    return tuple(entries)


def process_pdgs(process: str) -> tuple[int, ...]:
    parts = process.lower().replace("bar", "~").split(">")
    if len(parts) != 2:
        raise LegacyOracleError(f"invalid concrete process: {process!r}")
    names = (*parts[0].split(), *parts[1].split())
    try:
        return tuple(_PDG_BY_NAME[name] for name in names)
    except KeyError as error:
        raise LegacyOracleError(
            f"legacy built-in-SM oracle does not recognize {error.args[0]!r}"
        ) from error


def select_process_entry(entries: Sequence[ProcessEntry], process: str) -> ProcessEntry:
    matches = matching_process_entries(entries, process)
    if not matches:
        raise LegacyOracleError(f"no Fortran process row matches {process!r}")
    return matches[0]


def matching_process_entries(
    entries: Sequence[ProcessEntry], process: str
) -> tuple[ProcessEntry, ...]:
    """Prefer the exact requested external order before multiset fallbacks."""

    wanted = process_pdgs(process)
    wanted_multiset = sorted(wanted)
    matches = tuple(
        entry for entry in entries if sorted(entry.process_pdgs) == wanted_multiset
    )
    return tuple(entry for entry in matches if entry.process_pdgs == wanted) + tuple(
        entry for entry in matches if entry.process_pdgs != wanted
    )


def _permutation(
    source_pdgs: Sequence[int], target_pdgs: Sequence[int]
) -> tuple[int, ...]:
    positions: dict[int, deque[int]] = defaultdict(deque)
    for index, pdg in enumerate(source_pdgs):
        positions[int(pdg)].append(index)
    result: list[int] = []
    for pdg in target_pdgs:
        queue = positions[int(pdg)]
        if not queue:
            raise LegacyOracleError(
                "cannot map external ordering "
                f"{tuple(source_pdgs)} to {tuple(target_pdgs)}"
            )
        result.append(queue.popleft())
    if any(positions.values()):
        raise LegacyOracleError(
            f"cannot map external ordering {tuple(source_pdgs)} to {tuple(target_pdgs)}"
        )
    return tuple(result)


def _parse_probe_output(output: str) -> ProbeResult:
    value_match = _VALUE_RE.search(output)
    component_match = _COMPONENT_RE.search(output)
    if value_match is None or component_match is None:
        raise LegacyOracleError(
            "Fortran color probe did not emit matrix-element values"
        )
    integers: dict[str, int] = {}
    for name, label in _INTEGER_LABELS.items():
        match = re.search(rf"^{label}\s+(\d+)$", output, re.MULTILINE)
        if match is None:
            raise LegacyOracleError(f"Fortran color probe did not emit {label}")
        integers[name] = int(match.group(1))
    return ProbeResult(
        value=float(value_match.group(1)),
        components=tuple(float(value) for value in component_match.groups()),
        **integers,
    )


def run_color_probe(
    repository: Path,
    *,
    process_file: Path,
    entry: ProcessEntry,
    source_pdgs: Sequence[int],
    momenta: Sequence[Sequence[float]],
    color_accuracy: str,
    helicities: Sequence[int] | None = None,
) -> ProbeResult:
    permutation = _permutation(source_pdgs, entry.process_pdgs)
    ordered_momenta = [momenta[index] for index in permutation]
    ordered_helicities = (
        [int(helicities[index]) for index in permutation]
        if helicities is not None
        else []
    )
    # The pinned Fortran probe copies this path into a fixed 80-character
    # buffer, so use the deliberately short system-temporary spelling.
    with tempfile.TemporaryDirectory(prefix="pac-", dir="/tmp") as raw:
        momentum_path = Path(raw) / "momenta.dat"
        momentum_path.write_text(
            "\n".join(
                " ".join(format(float(component), ".17g") for component in vector)
                for vector in ordered_momenta
            )
            + "\n",
            encoding="utf-8",
        )
        command = [
            str((repository / "amplicol_color_probe").resolve()),
            "1",
            str(entry.group),
            str(entry.integral),
            color_accuracy,
            str(process_file.resolve()),
            str(momentum_path),
            *(str(value) for value in ordered_helicities),
        ]
        completed = _run(command, cwd=Path(raw))
    return _parse_probe_output(completed.stdout + "\n" + completed.stderr)


def _resolved_case_value(cases: Mapping[str, Any], case: Mapping[str, Any]) -> Any:
    return (
        cases[str(case["resolved_from"])]["resolved"]
        if "resolved_from" in case
        else case["resolved"]
    )


def _momenta_case_value(cases: Mapping[str, Any], case: Mapping[str, Any]) -> Any:
    return (
        cases[str(case["momenta_from"])]["momenta"]
        if "momenta_from" in case
        else case["momenta"]
    )


def _helicities(identifier: str) -> tuple[int, ...]:
    if not identifier.startswith("h:"):
        raise LegacyOracleError(f"invalid physical helicity ID: {identifier!r}")
    return tuple(int(value) for value in identifier[2:].split(","))


def _close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=1.0e-8, abs_tol=1.0e-15)


def verify_fixture(
    repository: Path,
    fixture_path: Path,
    *,
    jobs: int,
    build: bool,
) -> dict[str, Any]:
    repository = repository.resolve()
    validate_checkout(repository)
    if build:
        build_color_probe(repository, jobs=jobs)
    elif not (repository / "amplicol_color_probe").is_file():
        raise LegacyOracleError("amplicol_color_probe has not been built")

    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    cases: dict[str, Any] = fixture["cases"]
    selected = {
        name: case
        for name, case in cases.items()
        if name.startswith("builtin_sm_")
        and case["process"]
        in {
            "d d~ > z",
            "d d~ > z g",
        }
    }
    report_cases: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="pyamplicol-fortran-process-") as raw:
        work = Path(raw)
        for process in sorted({case["process"] for case in selected.values()}):
            _run(
                [
                    sys.executable,
                    str(repository / "process_list.py"),
                    "--serial",
                    process,
                ],
                cwd=work,
            )
            process_file = work / "processes.txt"
            entries = parse_process_file(process_file)
            matching_entries = matching_process_entries(entries, process)
            if not matching_entries:
                raise LegacyOracleError(f"no Fortran process row matches {process!r}")
            entry = matching_entries[0]
            source_pdgs = process_pdgs(process)

            for name, case in selected.items():
                if case["process"] != process:
                    continue
                momenta = _momenta_case_value(cases, case)[0]
                summed = run_color_probe(
                    repository,
                    process_file=process_file,
                    entry=entry,
                    source_pdgs=source_pdgs,
                    momenta=momenta,
                    color_accuracy=case["color_accuracy"],
                )
                expected_total = float(case["total"])
                if not _close(summed.value, expected_total):
                    raise LegacyOracleError(
                        f"{name}: Fortran total {summed.value:.17g} does not match "
                        f"fixture {expected_total:.17g}"
                    )

                maximum_relative_difference = 0.0
                resolved = _resolved_case_value(cases, case)
                for helicity_id, colors in resolved.items():
                    if len(colors) != 1:
                        raise LegacyOracleError(
                            f"{name}: low-multiplicity oracle expects one "
                            "color component"
                        )
                    expected = float(next(iter(colors.values())))
                    probe = run_color_probe(
                        repository,
                        process_file=process_file,
                        entry=entry,
                        source_pdgs=source_pdgs,
                        momenta=momenta,
                        color_accuracy=case["color_accuracy"],
                        helicities=_helicities(helicity_id),
                    )
                    if not _close(probe.value, expected):
                        raise LegacyOracleError(
                            f"{name} {helicity_id}: Fortran value "
                            f"{probe.value:.17g} does not match fixture {expected:.17g}"
                        )
                    denominator = max(abs(expected), 1.0e-300)
                    maximum_relative_difference = max(
                        maximum_relative_difference,
                        abs(probe.value - expected) / denominator,
                    )

                report_cases[name] = {
                    "process": process,
                    "color_accuracy": case["color_accuracy"],
                    "total": summed.value,
                    "max_resolved_relative_difference": maximum_relative_difference,
                    "fortran_process_entry": {
                        **asdict(entry),
                        "exact_external_order": entry.process_pdgs == source_pdgs,
                        "matching_row_count": len(matching_entries),
                        "source_to_row_permutation": _permutation(
                            source_pdgs, entry.process_pdgs
                        ),
                    },
                    "fortran_topology": {
                        key: value
                        for key, value in asdict(summed).items()
                        if key not in {"value", "components"}
                    },
                }

    return {
        "schema_version": 1,
        "oracle": "legacy-fortran-amplicol-color-probe",
        "revision": expected_revision(),
        "fixture": fixture_path.name,
        "cases": report_cases,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=DEFAULT_REPOSITORY)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--jobs", type=int, default=5)
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = verify_fixture(
        args.repository,
        args.fixture,
        jobs=max(1, args.jobs),
        build=not args.no_build,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except LegacyOracleError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
