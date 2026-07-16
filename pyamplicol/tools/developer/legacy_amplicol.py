#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Run the pinned Fortran AmpliCol as a developer-only physics oracle."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import importlib
import json
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

_DEVELOPER_DIRECTORY = Path(__file__).resolve().parent
if not __package__ and str(_DEVELOPER_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(_DEVELOPER_DIRECTORY))
if TYPE_CHECKING:
    from tools.developer.legacy_oracle import checkout as _checkout
    from tools.developer.legacy_oracle import evidence as _evidence
    from tools.developer.legacy_oracle import model as _model
    from tools.developer.legacy_oracle import probe as _probe
    from tools.developer.legacy_oracle import processes as _processes
else:
    _ORACLE_PACKAGE = (
        f"{__package__}.legacy_oracle" if __package__ else "legacy_oracle"
    )
    _checkout = importlib.import_module(f"{_ORACLE_PACKAGE}.checkout")
    _evidence = importlib.import_module(f"{_ORACLE_PACKAGE}.evidence")
    _model = importlib.import_module(f"{_ORACLE_PACKAGE}.model")
    _probe = importlib.import_module(f"{_ORACLE_PACKAGE}.probe")
    _processes = importlib.import_module(f"{_ORACLE_PACKAGE}.processes")

ROOT = _model.ROOT
LOCK = _model.LOCK
DEFAULT_REPOSITORY = _model.DEFAULT_REPOSITORY
DEFAULT_FIXTURE = _model.DEFAULT_FIXTURE
FORTRAN_RTOL = _model.FORTRAN_RTOL
FORTRAN_ATOL = _model.FORTRAN_ATOL
FORTRAN_EVIDENCE_RTOL_DECIMAL = _model.FORTRAN_EVIDENCE_RTOL_DECIMAL
FORTRAN_EVIDENCE_ATOL_DECIMAL = _model.FORTRAN_EVIDENCE_ATOL_DECIMAL
FORTRAN_CERTIFIED_DECIMAL_DIGITS = _model.FORTRAN_CERTIFIED_DECIMAL_DIGITS
_V2_ONLY_LEGACY_PATCHES = _model.V2_ONLY_LEGACY_PATCHES
_MAX_SUPPORTED_QUARK_LINES = _model.MAX_SUPPORTED_QUARK_LINES

LegacyOracleError = _model.LegacyOracleError
ProcessEntry = _model.ProcessEntry
LcRowPartition = _model.LcRowPartition
CompilerProvenance = _model.CompilerProvenance
ProbeResult = _model.ProbeResult


def _run(
    command: Sequence[str],
    *,
    cwd: Path,
    capture: bool = True,
) -> subprocess.CompletedProcess[str]:
    return _checkout._run(
        command,
        cwd=cwd,
        capture=capture,
        subprocess_module=subprocess,
    )


def _release_lock() -> dict[str, Any]:
    return _checkout._release_lock(lock_path=LOCK)


def expected_revision() -> str:
    return _checkout.expected_revision(lock=_release_lock())


def checkout_url() -> str:
    return _checkout.checkout_url(lock=_release_lock())


def managed_patches() -> tuple[Path, ...]:
    return _checkout.managed_patches(root=ROOT, lock=_release_lock())


def managed_patch_metadata(
    *,
    fixture_schema_version: int | None = None,
) -> tuple[dict[str, str], ...]:
    return _checkout.managed_patch_metadata(
        fixture_schema_version=fixture_schema_version,
        root=ROOT,
        lock=_release_lock(),
    )


def prepare_checkout(repository: Path) -> None:
    _checkout.prepare_checkout(
        repository,
        run=_run,
        validate=validate_checkout,
        url=checkout_url,
        revision=expected_revision,
        patches=managed_patches,
    )


def _managed_patch_paths(patches: Sequence[Path]) -> tuple[str, ...]:
    return _checkout._managed_patch_paths(patches)


def _validate_exact_patch_state(repository: Path, patches: Sequence[Path]) -> None:
    _checkout._validate_exact_patch_state(
        repository,
        patches,
        subprocess_module=subprocess,
    )


def validate_checkout(repository: Path) -> None:
    _checkout.validate_checkout(
        repository,
        run=_run,
        revision=expected_revision,
        patches=managed_patches,
        validate_exact=_validate_exact_patch_state,
        subprocess_module=subprocess,
    )


def build_color_probe(repository: Path, *, jobs: int) -> None:
    validate_checkout(repository)
    _run(
        ["make", f"-j{max(1, jobs)}", "amplicol_color_probe"],
        cwd=repository,
        capture=False,
    )


def _make_database_variable(database: str, name: str) -> str:
    return _checkout._make_database_variable(database, name)


def _compiler_provenance(repository: Path) -> CompilerProvenance:
    return _checkout._compiler_provenance(
        repository,
        run=_run,
        shutil_module=shutil,
    )


_PDG_BY_NAME = _processes._PDG_BY_NAME
EXPECTED_FORTRAN_PROCESS_ROWS = _processes.EXPECTED_FORTRAN_PROCESS_ROWS
parse_process_file = _processes.parse_process_file
process_pdgs = _processes.process_pdgs
_normalized_process_expression = _processes._normalized_process_expression
_validate_supported_quark_line_scope = _processes._validate_supported_quark_line_scope
expected_process_entry = _processes.expected_process_entry
select_process_entry = _processes.select_process_entry
_select_declared_process_entry = _processes._select_declared_process_entry
matching_process_entries = _processes.matching_process_entries
matching_process_entries_for_pdgs = _processes.matching_process_entries_for_pdgs
_permutation = _processes._permutation
_concrete_process_id = _processes._concrete_process_id
_ordered_leg_ids = _processes._ordered_leg_ids

_VALUE_RE = _probe._VALUE_RE
_COMPONENT_RE = _probe._COMPONENT_RE
_LC_ROW_VALUE_RE = _probe._LC_ROW_VALUE_RE
_LC_ROW_PERMUTATION_RE = _probe._LC_ROW_PERMUTATION_RE
_LC_ROW_SUM_RE = _probe._LC_ROW_SUM_RE
_INTEGER_LABELS = _probe._INTEGER_LABELS
_parse_probe_output = _probe._parse_probe_output
_probe_number = _probe._probe_number
_ordered_binary64_momenta = _probe._ordered_binary64_momenta
_binary64_input_sha256 = _probe._binary64_input_sha256


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
    ordered_momenta = _ordered_binary64_momenta(
        source_pdgs,
        entry.process_pdgs,
        momenta,
    )
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


_resolved_case_value = _probe._resolved_case_value
_momenta_case_value = _probe._momenta_case_value
_helicities = _probe._helicities
_close = _probe._close


def _verify_fixture_v1(
    repository: Path,
    fixture_path: Path,
    fixture: Mapping[str, Any],
) -> dict[str, Any]:
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
            source_pdgs = process_pdgs(process)
            entry, matching_entries = _select_declared_process_entry(
                entries,
                generated_process=process,
                wanted_pdgs=source_pdgs,
            )

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
                        if key
                        not in {
                            "value",
                            "components",
                            "lc_row_partitions",
                            "lc_partition_sum",
                            "value_decimal",
                            "component_decimals",
                            "lc_partition_sum_decimal",
                        }
                    },
                }

    return {
        "schema_version": 1,
        "oracle": "legacy-fortran-amplicol-color-probe",
        "revision": expected_revision(),
        "fixture": fixture_path.name,
        "fixture_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
        "patches": managed_patch_metadata(fixture_schema_version=1),
        "tolerances": {
            "relative": FORTRAN_RTOL,
            "absolute": FORTRAN_ATOL,
        },
        "cases": report_cases,
    }


_decimal_value = _probe._decimal_value
_canonical_decimal = _probe._canonical_decimal
_probe_value_decimal = _probe._probe_value_decimal
_lc_row_value_decimal = _probe._lc_row_value_decimal
_decimal_close = _probe._decimal_close
_canonical_physics_output_sha256 = _probe._canonical_physics_output_sha256
_resolved_single_lc_row = _probe._resolved_single_lc_row

_reference_fixture_v2_module = _evidence._reference_fixture_v2_module
_load_fixture = _evidence.load_fixture
_fortran_evidence_set_id = _evidence._fortran_evidence_set_id
_stable_command_description = _evidence._stable_command_description
_evidence_id = _evidence._evidence_id


def _fortran_dependency_ids(fixture: Mapping[str, Any]) -> tuple[str, ...]:
    return _evidence._fortran_dependency_ids(
        fixture,
        revision=expected_revision(),
    )


def _oracle_content_sha256() -> str:
    return _evidence._oracle_content_sha256(
        lock=_release_lock(),
        patches=managed_patch_metadata(),
        revision=expected_revision(),
    )


def _verify_fixture_v2(
    repository: Path,
    fixture_path: Path,
    fixture: Mapping[str, Any],
) -> dict[str, Any]:
    reference = _reference_fixture_v2_module()
    reference._validate_wire(
        fixture,
        "reference-physics-v2.schema.json",
        str(fixture_path),
    )
    reference._validate_physics_case_hashes(fixture)

    models = {str(model["id"]): model for model in fixture["models"]}
    processes = {str(process["id"]): process for process in fixture["processes"]}
    points = {str(point["id"]): point for point in fixture["points"]}
    selected_cases = tuple(
        case
        for case in fixture["cases"]
        if models[str(case["model_id"])]["source_kind"] == "built-in-sm"
    )
    if not selected_cases:
        raise LegacyOracleError("fixture-v2 contains no built-in-SM cases")

    compiler = _compiler_provenance(repository)
    evidence_set_id = _fortran_evidence_set_id(fixture)
    records: list[dict[str, Any]] = []
    cases_by_process: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for case in selected_cases:
        process = processes[str(case["process_id"])]
        _validate_supported_quark_line_scope(
            tuple(int(value) for value in process["external_pdgs"]),
            context=f"case {case['id']}",
        )
        concrete_id = _concrete_process_id(
            processes,
            str(case["process_id"]),
        )
        cases_by_process[concrete_id].append(case)

    with tempfile.TemporaryDirectory(prefix="pyamplicol-fortran-process-") as raw:
        work = Path(raw)
        ordered_process_ids = sorted(
            cases_by_process,
            key=lambda process_id: (
                str(processes[process_id]["expression"]),
                process_id,
            ),
        )
        for concrete_id in ordered_process_ids:
            concrete_process = processes[concrete_id]
            generation_expression = str(concrete_process["expression"])
            _run(
                [
                    sys.executable,
                    str(repository / "process_list.py"),
                    "--serial",
                    generation_expression,
                ],
                cwd=work,
            )
            process_file = work / "processes.txt"
            entries = parse_process_file(process_file)

            ordered_cases = sorted(
                cases_by_process[concrete_id],
                key=lambda item: item["id"],
            )
            for case in ordered_cases:
                case_id = str(case["id"])
                process = processes[str(case["process_id"])]
                expression = str(process["expression"])
                source_pdgs = tuple(int(value) for value in process["external_pdgs"])
                source_leg_ids = tuple(
                    str(value) for value in process["external_leg_ids"]
                )
                entry, matching_entries = _select_declared_process_entry(
                    entries,
                    generated_process=generation_expression,
                    wanted_pdgs=source_pdgs,
                )
                permutation = _permutation(source_pdgs, entry.process_pdgs)
                row_leg_ids = tuple(source_leg_ids[index] for index in permutation)
                ordered_color_legs = _ordered_leg_ids(
                    row_leg_ids,
                    entry.color_order,
                    context=f"Fortran process row for {expression}",
                )
                color_accuracy = str(case["color_accuracy"])
                helicities = tuple(case["axes"]["helicities"])
                colors = tuple(case["axes"]["colors"])
                observations = {
                    str(observation["point_id"]): observation
                    for observation in case["observations"]
                }
                for point_id_value in case["point_ids"]:
                    point_id = str(point_id_value)
                    point = points[point_id]
                    observation = observations[point_id]
                    observation_certified_digits = int(
                        observation["certified_decimal_digits"]
                    )
                    if observation_certified_digits > FORTRAN_CERTIFIED_DECIMAL_DIGITS:
                        raise LegacyOracleError(
                            f"{case_id}/{point_id}: observation certifies "
                            f"{observation_certified_digits} digits, beyond the "
                            f"binary64 Fortran oracle's "
                            f"{FORTRAN_CERTIFIED_DECIMAL_DIGITS}-digit scope"
                        )
                    expected_evidence_id = _evidence_id(case_id, point_id)
                    if expected_evidence_id not in observation["evidence_refs"]:
                        raise LegacyOracleError(
                            f"{case_id}/{point_id} does not reference generated "
                            f"evidence {expected_evidence_id}"
                        )
                    momenta = tuple(
                        tuple(float(component) for component in row)
                        for row in point["momenta"]
                    )
                    binary64_input_sha256 = _binary64_input_sha256(
                        source_pdgs,
                        entry.process_pdgs,
                        momenta,
                    )
                    summed = run_color_probe(
                        repository,
                        process_file=process_file,
                        entry=entry,
                        source_pdgs=source_pdgs,
                        momenta=momenta,
                        color_accuracy=color_accuracy,
                    )
                    observed_total = _probe_value_decimal(summed)
                    expected_total = Decimal(str(observation["total"]))
                    if not _decimal_close(observed_total, expected_total):
                        raise LegacyOracleError(
                            f"{case_id}/{point_id}: Fortran total {observed_total} "
                            f"does not match fixture {expected_total}"
                        )

                    multi_flow_lc = color_accuracy == "lc" and len(colors) > 1
                    coverage = "helicity-aggregate" if multi_flow_lc else "resolved"
                    resolved_values: list[list[str]] = []
                    helicity_totals: list[str] = []
                    helicity_results: list[ProbeResult] = []
                    helicity_sum = Decimal(0)
                    for row_index, helicity in enumerate(helicities):
                        expected_row = tuple(
                            Decimal(str(value))
                            for value in observation["values"][row_index]
                        )
                        probe = run_color_probe(
                            repository,
                            process_file=process_file,
                            entry=entry,
                            source_pdgs=source_pdgs,
                            momenta=momenta,
                            color_accuracy=color_accuracy,
                            helicities=tuple(
                                int(value) for value in helicity["values"]
                            ),
                        )
                        context = f"{case_id}/{point_id} {helicity['id']}"
                        aggregate = _probe_value_decimal(probe)
                        helicity_sum += aggregate
                        if multi_flow_lc:
                            expected_aggregate = sum(expected_row, Decimal(0))
                            if bool(helicity["structural_zero"]) and aggregate != 0:
                                raise LegacyOracleError(
                                    f"{context}: Fortran did not preserve structural "
                                    f"zero aggregate ({aggregate})"
                                )
                            if not _decimal_close(aggregate, expected_aggregate):
                                raise LegacyOracleError(
                                    f"{context}: Fortran helicity aggregate "
                                    f"{aggregate} does not match fixture sum "
                                    f"{expected_aggregate}"
                                )
                            helicity_totals.append(_canonical_decimal(aggregate))
                            helicity_results.append(probe)
                            continue

                        if color_accuracy == "lc":
                            observed_row = _resolved_single_lc_row(
                                probe,
                                colors=colors,
                                source_to_row_permutation=permutation,
                                context=context,
                            )
                        else:
                            if len(colors) != 1:
                                raise LegacyOracleError(
                                    f"{context}: contracted color case must have one "
                                    "component"
                                )
                            observed_row = [
                                _canonical_decimal(_probe_value_decimal(probe))
                            ]
                        observed_decimals = tuple(
                            Decimal(value) for value in observed_row
                        )
                        if bool(helicity["structural_zero"]) and any(
                            value != 0 for value in observed_decimals
                        ):
                            raise LegacyOracleError(
                                f"{context}: Fortran did not preserve structural zero "
                                f"({observed_row})"
                            )
                        if len(observed_decimals) != len(expected_row) or any(
                            not _decimal_close(observed, expected)
                            for observed, expected in zip(
                                observed_decimals,
                                expected_row,
                                strict=True,
                            )
                        ):
                            raise LegacyOracleError(
                                f"{context}: Fortran resolved values {observed_row} "
                                f"do not match fixture {list(expected_row)}"
                            )
                        resolved_sum = sum(observed_decimals, Decimal(0))
                        if not _decimal_close(aggregate, resolved_sum):
                            raise LegacyOracleError(
                                f"{context}: resolved Fortran values do not sum to "
                                f"aggregate {aggregate}"
                            )
                        resolved_values.append(observed_row)
                        helicity_results.append(probe)

                    if not _decimal_close(helicity_sum, observed_total):
                        raise LegacyOracleError(
                            f"{case_id}/{point_id}: fixed-helicity Fortran sum "
                            f"{helicity_sum} does not match summed probe "
                            f"{observed_total}"
                        )

                    evidence_record: dict[str, object] = {
                            "id": expected_evidence_id,
                            "case_id": case_id,
                            "point_id": point_id,
                            "independent_of_pyamplicol": True,
                            "arithmetic_precision_bits": 53,
                            "round_trip_decimal_digits": 17,
                            "certified_decimal_digits": (
                                FORTRAN_CERTIFIED_DECIMAL_DIGITS
                            ),
                            "arithmetic": "binary64",
                            "coverage": coverage,
                            "helicity_ids": [
                                str(helicity["id"]) for helicity in helicities
                            ],
                            "color_ids": [str(color["id"]) for color in colors],
                            "observed_total": _canonical_decimal(
                                _probe_value_decimal(summed)
                            ),
                            "observed_helicity_totals": (
                                helicity_totals if multi_flow_lc else None
                            ),
                            "observed_values": (
                                None if multi_flow_lc else resolved_values
                            ),
                            "process_identity": {
                                "expression": expression,
                                "ordered_external_pdgs": list(entry.process_pdgs),
                                "ordered_external_leg_ids": list(row_leg_ids),
                                "source_to_row_permutation": list(permutation),
                                "row_id": (
                                    f"group:{entry.group}:integral:{entry.integral}"
                                ),
                                "color_order_count": summed.color_orders,
                                "ordered_color_legs": list(ordered_color_legs),
                            },
                            "input_sha256": reference._parse_point(
                                point,
                                0,
                            ).input_sha256(),
                            "physics_case_sha256": str(case["physics_case_sha256"]),
                            "oracle_output_sha256": _canonical_physics_output_sha256(
                                summed,
                                helicity_results,
                            ),
                            "command": _stable_command_description(
                                process=expression,
                                case_id=case_id,
                                point_id=point_id,
                                color_accuracy=color_accuracy,
                                entry=entry,
                                pdg_match_count=len(matching_entries),
                                compiler=compiler,
                                binary64_input_sha256=binary64_input_sha256,
                            ),
                            "tolerances": {
                                "relative": "0.0000000001",
                                "absolute": "0.000000000001",
                            },
                        }
                    evidence_record["evidence_record_sha256"] = (
                        reference.evidence_record_sha256(evidence_record)
                    )
                    records.append(evidence_record)

    evidence = {
        "evidence_schema_version": 2,
        "kind": "pyamplicol-reference-oracle-evidence",
        "evidence_set_id": evidence_set_id,
        "captured_at": str(fixture["provenance"]["captured_at"]),
        "oracle": {
            "id": "oracle:legacy-fortran-amplicol",
            "name": "Pinned legacy Fortran AmpliCol",
            "implementation": "independent amplicol_color_probe executable",
            "revision": expected_revision(),
            "content_sha256": _oracle_content_sha256(),
            "independence_statement": (
                "The pinned Fortran implementation is built and executed outside "
                "the pyAmpliCol generation and Rusticol runtime code paths."
            ),
            "validation_profile": "binary64",
            "tolerance_ceiling": {
                "relative": "0.0000000001",
                "absolute": "0.000000000001",
            },
        },
        "dependency_ids": list(_fortran_dependency_ids(fixture)),
        "records": records,
    }

    reference._validate_wire(
        evidence,
        "reference-oracle-evidence-v2.schema.json",
        "generated Fortran evidence",
    )
    return evidence


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

    fixture = _load_fixture(fixture_path)
    version = fixture.get("fixture_schema_version")
    if version == 1:
        return _verify_fixture_v1(repository, fixture_path, fixture)
    if version == 2:
        return _verify_fixture_v2(repository, fixture_path, fixture)
    raise LegacyOracleError(f"unsupported reference fixture schema version {version!r}")


_COMPILER_VARIANT_COMMAND_KEYS = _evidence._COMPILER_VARIANT_COMMAND_KEYS
_REQUIRED_PROVENANCE_COMMAND_KEYS = _evidence._REQUIRED_PROVENANCE_COMMAND_KEYS
_semantic_replay_command = _evidence._semantic_replay_command


def _assert_v2_reports_semantically_equal(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
) -> None:
    _evidence._assert_v2_reports_semantically_equal(
        expected,
        actual,
        reference_loader=_reference_fixture_v2_module,
        semantic_replay=_semantic_replay_command,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=DEFAULT_REPOSITORY)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--jobs", type=int, default=5)
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument(
        "--prepare-checkout",
        action="store_true",
        help="clone and patch the pinned public oracle checkout when absent",
    )
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--output", type=Path)
    output.add_argument(
        "--check-output",
        type=Path,
        help="fail unless the generated report exactly matches this tracked file",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.prepare_checkout:
        prepare_checkout(args.repository)
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
    if args.check_output is not None:
        expected = args.check_output.read_text(encoding="utf-8")
        fixture_version = json.loads(args.fixture.read_text(encoding="utf-8")).get(
            "fixture_schema_version"
        )
        if fixture_version == 2:
            try:
                expected_payload = json.loads(expected)
            except json.JSONDecodeError as error:
                raise LegacyOracleError(
                    f"tracked Fortran evidence is not valid JSON: {error}"
                ) from error
            _assert_v2_reports_semantically_equal(expected_payload, report)
        elif rendered != expected:
            difference = "".join(
                difflib.unified_diff(
                    expected.splitlines(keepends=True),
                    rendered.splitlines(keepends=True),
                    fromfile=str(args.check_output),
                    tofile="fresh legacy AmpliCol report",
                )
            )
            raise LegacyOracleError(
                "tracked legacy AmpliCol report is stale\n" + difference
            )
    print(rendered, end="")
    return 0


def _entrypoint(argv: Sequence[str] | None = None) -> int:
    try:
        return main(argv)
    except (LegacyOracleError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(_entrypoint())
