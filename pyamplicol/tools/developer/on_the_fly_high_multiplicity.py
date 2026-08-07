#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Guarded, process-parametric high-multiplicity LC study for on-the-fly mode.

Run ``selected`` first: it creates the multiplicity's sole fresh OTF artifact.
Run ``all-flow`` separately with ``--candidate-artifact`` pointing at that
artifact.  Selected n=5/6/7 cells use exactly one independent authority:
authenticated original AmpliCol where supported, otherwise recurrence for the
four-open-quark-line family.  Selected n=8/9 and every all-flow invocation are
intentionally OTF-only feasibility probes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pyamplicol.generation.service as generation_service  # noqa: E402
from pyamplicol import (  # noqa: E402
    BenchmarkRunner,
    Generator,
    ModelSource,
    ProcessRequest,
    Runtime,
)
from pyamplicol.api.errors import ArtifactError  # noqa: E402
from pyamplicol.artifacts import load_manifest  # noqa: E402
from pyamplicol.config import (  # noqa: E402
    ColorConfig,
    EvaluatorConfig,
    EvaluatorOptimizationConfig,
    GenerationConfig,
    GenerationRelationDiscoveryConfig,
    GenerationValidationConfig,
    JITConfig,
    RunConfig,
)
from pyamplicol.models.builtin.validation import generic_validation_point  # noqa: E402
from tools.ci.memory_watchdog import GIB, run_guarded  # noqa: E402
from tools.developer import legacy_amplicol  # noqa: E402
from tools.developer import on_the_fly_lc_gate as n4  # noqa: E402
from tools.developer.legacy_oracle.model import (  # noqa: E402
    PINNED_REFERENCE_REVISION,
)
from tools.performance_report.artifacts import (  # noqa: E402
    ArtifactStore,
    ArtifactStoreError,
)
from tools.performance_report.cache import validate_measurement  # noqa: E402
from tools.performance_report.catalog import REPORT_CATALOG  # noqa: E402
from tools.performance_report.legacy import (  # noqa: E402
    LegacyAdapterError,
    _canonical_mapped_color_word,
    _initial_state_count,
)
from tools.performance_report.models import Accuracy, Workload  # noqa: E402
from tools.performance_report.runner import (  # noqa: E402
    RunnerError,
    SelectorContract,
    derive_selector_contract,
    point_digest,
    validate_selector_contract,
)
from tools.performance_report.selector_policy import (  # noqa: E402
    SelectorPolicyError,
    canonical_lc_flow_word,
    fixed_selector_helicity,
    selector_color_flow_id,
    selector_helicity_id,
)
from tools.performance_report.service import (  # noqa: E402
    ReportPaths,
    validate_profile_name,
)
from tools.performance_report.source_identity import (  # noqa: E402
    ReportSourceIdentity,
    ReportSourceIdentityError,
    require_eligible_report_source,
)

KIND = "pyamplicol-on-the-fly-high-multiplicity-study"
SCHEMA_VERSION = 4
SUPPORTED_PROCESS_IDS = (7, 8, 11, 13, 14, 15)
SUPPORTED_MULTIPLICITIES = (5, 6, 7, 8, 9)
SUPPORTED_PROCESS_MULTIPLICITIES = {
    7: (5, 6, 7, 8, 9),
    8: (5, 6, 7, 8, 9),
    11: (5, 6, 7),
    13: (5, 6, 7),
    14: (6, 7),
    15: (5, 6, 7),
}
AMPLI_COL_AUTHORITY_IDS = frozenset({7, 8, 11, 13, 15})
AUTHORITY_AMPLICOL = "amplicol"
AUTHORITY_RECURRENCE = "recurrence"
AUTHORITY_OTF_ONLY = "otf-only"
SEEDS = n4.SEEDS
BATCH_SIZE = 128
WARMUP_RUNS = 2
MINIMUM_SAMPLES = 5
WATCHDOG_BYTES = 30 * GIB
STATE_KIND = "rusticol-on-the-fly-runtime-state-census-v1"
STATE_METHOD = "_on_the_fly_runtime_state_census_json"
STATE_COUNTS = (
    "process_preparation_count",
    "retained_family_count",
    "pending_family_count",
    "retained_selection_count",
    "retained_request_count",
    "retained_amplitude_destination_count",
    "retained_executor_handle_count",
    "retained_query_local_trace_count",
    "retained_embedded_lookup_key_count",
    "semantic_executor_binding_count",
)
COMPACT_ZERO_COUNTS = (
    "pending_family_count",
    "retained_query_local_trace_count",
    "retained_embedded_lookup_key_count",
)
ACTIVE_COUNT_FIELDS = (
    "query_count",
    "union_unique_current_count",
    "union_unique_current_component_count",
    "union_source_rows",
    "union_contribution_rows",
    "union_finalization_rows",
    "union_closure_rows",
    "union_amplitude_destination_count",
    "union_source_executor_call_groups",
    "union_contribution_executor_call_groups",
    "union_finalization_executor_call_groups",
    "union_closure_executor_call_groups",
)
OPERATION_ROLES = ("source", "contribution", "finalization", "closure")

Point = tuple[tuple[float, ...], ...]
Points = tuple[Point, ...]


class StudyError(RuntimeError):
    """The bounded study contract was not satisfied."""


@dataclass(frozen=True, slots=True)
class Case:
    process_table_id: int
    process_key: str
    multiplicity: int
    process: str
    process_id: str
    pdgs: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Selector:
    flow_word: tuple[int, ...]
    flow_id: str
    helicities: tuple[int, ...]
    helicity_id: str

    def mapping(self, points: Points) -> dict[str, object]:
        return {
            "selected_color_flow_ids": [self.flow_id],
            "selected_color_words": [list(self.flow_word)],
            "all_flow_helicity_ids": [self.helicity_id],
            "all_flow_source_helicities": {
                str(index): value
                for index, value in enumerate(self.helicities, start=1)
            },
            "point_digest": point_digest(points),
        }


@dataclass(frozen=True, slots=True)
class Evaluation:
    total: tuple[object, ...]
    resolved: object | None
    total_vs_resolved: dict[str, object] | None = None


@dataclass(frozen=True, slots=True)
class AmplicolReference:
    selector: Selector
    contract: SelectorContract
    selector_contract: dict[str, object]
    matrix_element: float
    timing: dict[str, object]
    lineage: dict[str, object]
    library_path: Path | None = None
    process_row: tuple[int, int] | None = None


def _positive_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--process-id", required=True, type=int, choices=SUPPORTED_PROCESS_IDS
    )
    parser.add_argument(
        "--multiplicity", required=True, type=int, choices=SUPPORTED_MULTIPLICITIES
    )
    parser.add_argument(
        "--workload", choices=("selected", "all-flow"), default="selected"
    )
    parser.add_argument("--prepared-model", type=Path)
    parser.add_argument(
        "--reference-repo-root",
        type=Path,
        help="repository root owning the authenticated AmpliCol profile; "
        "required only for AmpliCol-authority selected cells",
    )
    parser.add_argument(
        "--reference-profile",
        help="ReportPaths profile containing authenticated AmpliCol currents; "
        "required only for AmpliCol-authority selected cells",
    )
    parser.add_argument(
        "--candidate-artifact",
        type=Path,
        help="sole OTF artifact created by the selected invocation",
    )
    parser.add_argument(
        "--selected-report",
        type=Path,
        help="passed selected-workload report that created the OTF candidate",
    )
    parser.add_argument("--target-runtime", type=_positive_float, default=5.0)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def _case(process_table_id: int, multiplicity: int) -> Case:
    if process_table_id not in SUPPORTED_PROCESS_IDS:
        raise StudyError(f"unsupported process-table ID {process_table_id}")
    if multiplicity not in SUPPORTED_MULTIPLICITIES:
        raise StudyError(f"unsupported multiplicity n={multiplicity}")
    if multiplicity not in SUPPORTED_PROCESS_MULTIPLICITIES[process_table_id]:
        raise StudyError(
            f"process-table ID {process_table_id} does not define a planned "
            f"high-multiplicity cell at n={multiplicity}"
        )
    family = next(
        (
            candidate
            for candidate in REPORT_CATALOG.process_families
            if candidate.identifier == process_table_id
        ),
        None,
    )
    expected_keys = {
        7: "dd_tt_jets",
        8: "gg_gluons",
        11: "dd_ttzh_jets",
        13: "dd_3q_lines",
        14: "dd_4q_lines",
        15: "dd_3q_identical_lines",
    }
    if family is None or family.key != expected_keys[process_table_id]:
        raise StudyError("process-table catalog identity changed")
    process = family.process(multiplicity)
    if process is None:
        raise StudyError("process-table family does not define this multiplicity")
    bases = {
        7: (1, -1, 6, -6),
        8: (21, 21, 21, 21),
        11: (1, -1, 6, -6, 23, 25),
        13: (1, -1, 2, -2, 3, -3),
        14: (1, -1, 2, -2, 3, -3, 4, -4),
        15: (1, -1, 2, -2, 2, -2),
    }
    base_final_count = len(bases[process_table_id]) - 2
    pdgs = (
        *bases[process_table_id],
        *(21 for _ in range(multiplicity - base_final_count)),
    )
    return Case(
        process_table_id,
        family.key,
        multiplicity,
        process,
        f"otf_p{process_table_id}_{family.key}_n{multiplicity}",
        pdgs,
    )


def _authority_kind(case: Case, workload: str) -> str:
    if workload == "all-flow":
        return AUTHORITY_OTF_ONLY
    if workload != "selected":
        raise StudyError(f"unsupported workload {workload!r}")
    if case.process_table_id == 14:
        return AUTHORITY_RECURRENCE
    if case.multiplicity >= 8:
        return AUTHORITY_OTF_ONLY
    if case.process_table_id in AMPLI_COL_AUTHORITY_IDS:
        return AUTHORITY_AMPLICOL
    raise StudyError("selected cell has no declared authority policy")


def _validate_arguments(args: argparse.Namespace) -> None:
    case = _case(args.process_id, args.multiplicity)
    authority = _authority_kind(case, args.workload)
    has_reference_root = args.reference_repo_root is not None
    has_reference_profile = args.reference_profile is not None
    if has_reference_root != has_reference_profile:
        raise StudyError(
            "--reference-repo-root and --reference-profile must be supplied together"
        )
    if authority == AUTHORITY_AMPLICOL:
        if not has_reference_root:
            raise StudyError("AmpliCol-authority selected cells require reference args")
        if (
            not isinstance(args.reference_profile, str)
            or not args.reference_profile.strip()
        ):
            raise StudyError("--reference-profile must be non-empty")
    elif has_reference_root:
        raise StudyError("reference args are only valid for AmpliCol-authority cells")
    if args.workload == "selected":
        if args.prepared_model is None:
            raise StudyError("selected requires --prepared-model")
        if args.candidate_artifact is not None:
            raise StudyError("selected creates the candidate artifact")
        if args.selected_report is not None:
            raise StudyError("selected creates, rather than consumes, its report")
    elif args.candidate_artifact is None:
        raise StudyError("all-flow requires --candidate-artifact")
    elif args.selected_report is None:
        raise StudyError("all-flow requires --selected-report")
    elif args.prepared_model is not None:
        raise StudyError(
            "all-flow reuses the selected artifact without --prepared-model"
        )


def _existing(path: Path, *, directory: bool, label: str) -> Path:
    result = Path(os.path.abspath(path.expanduser()))
    if not (result.is_dir() if directory else result.is_file()):
        raise StudyError(f"{label} does not exist: {result}")
    return result


def _points(case: Case) -> Points:
    return tuple(
        tuple(tuple(map(float, particle.momentum)) for particle in point)
        for point in (
            generic_validation_point(case.process, sqrt_s=1000.0, seed=seed)
            for seed in SEEDS
        )
    )


def _timing_points(points: Points) -> Points:
    if not points:
        raise StudyError("benchmark requires the seed-101 correctness point")
    return (points[0],) * BATCH_SIZE


def _colored_source_labels(case: Case) -> tuple[int, ...]:
    """Return the exact public source labels carried by an LC flow word."""

    labels = tuple(
        label
        for label, pdg in enumerate(case.pdgs, start=1)
        if abs(pdg) in {1, 2, 3, 4, 5, 6, 21}
    )
    if not labels:
        raise StudyError("LC selector process has no colored external sources")
    return labels


def _selector_from_contract(case: Case, contract: SelectorContract) -> Selector:
    expected_helicities = fixed_selector_helicity(case.pdgs)
    expected_helicity_id = selector_helicity_id(expected_helicities)
    expected_sources = tuple(enumerate(expected_helicities, start=1))
    if (
        len(contract.selected_color_words) != 1
        or len(contract.selected_color_flow_ids) != 1
        or contract.runtime_all_flow_helicity_ids != (expected_helicity_id,)
        or contract.all_flow_source_helicities != expected_sources
    ):
        raise StudyError("AmpliCol selector contract has the wrong process axis")
    word = contract.selected_color_words[0]
    flow_id = contract.selected_color_flow_ids[0]
    if (
        len(word) != len(_colored_source_labels(case))
        or set(word) != set(_colored_source_labels(case))
        or selector_color_flow_id(word) != flow_id
    ):
        raise StudyError("AmpliCol selector flow does not round-trip")
    return Selector(word, flow_id, expected_helicities, expected_helicity_id)


def _compact_selector_context(
    runtime: object,
    case: Case,
    requested: tuple[str, ...],
) -> dict[str, object]:
    backend = getattr(runtime, "_backend", None)
    operation = getattr(backend, "_on_the_fly_benchmark_context", None)
    if not callable(operation):
        raise StudyError("candidate has no compact on-the-fly selector context")
    raw = operation(requested)
    if not isinstance(raw, Mapping):
        raise StudyError("candidate compact selector context is unavailable")
    selected = raw.get("selected_color_ids")
    if (
        raw.get("process_id") != case.process_id
        or raw.get("process_expression") != case.process
        or raw.get("color_accuracy") != "lc"
        or isinstance(raw.get("helicity_count"), bool)
        or not isinstance(raw.get("helicity_count"), int)
        or int(raw["helicity_count"]) < 1
        or isinstance(raw.get("color_count"), bool)
        or not isinstance(raw.get("color_count"), int)
        or int(raw["color_count"]) < 1
        or not isinstance(selected, Sequence)
        or isinstance(selected, (str, bytes))
        or len(selected) != len(requested)
        or any(not isinstance(value, str) or not value for value in selected)
    ):
        raise StudyError("candidate compact selector context has the wrong identity")
    return {
        "process_id": case.process_id,
        "process_expression": case.process,
        "color_accuracy": "lc",
        "helicity_count": raw["helicity_count"],
        "color_count": raw["color_count"],
        "requested_color_ids": list(requested),
        "selected_color_ids": list(selected),
    }


def _flow_word(flow_id: str) -> tuple[int, ...]:
    if not flow_id.startswith("flow:"):
        raise StudyError("compact selector did not return a semantic flow ID")
    try:
        word = canonical_lc_flow_word(
            tuple(int(token) for token in flow_id.removeprefix("flow:").split(","))
        )
    except (SelectorPolicyError, ValueError) as error:
        raise StudyError(
            "compact selector returned an invalid semantic flow ID"
        ) from error
    if selector_color_flow_id(word) != flow_id:
        raise StudyError("compact selector flow ID does not round-trip")
    return word


def _selector_from_compact_ordinal_one(
    runtime: object,
    case: Case,
    points: Points,
) -> tuple[Selector, SelectorContract, dict[str, object]]:
    """Derive the first labelled flow through the compact seed axis only."""

    context = _compact_selector_context(runtime, case, ("1",))
    selected = context["selected_color_ids"]
    if not isinstance(selected, list) or len(selected) != 1:
        raise StudyError("compact ordinal 1 did not resolve exactly one flow")
    flow_id = selected[0]
    if not isinstance(flow_id, str):
        raise StudyError("compact ordinal 1 did not resolve a semantic flow ID")
    flow_word = _flow_word(flow_id)
    if len(flow_word) != len(_colored_source_labels(case)) or set(flow_word) != set(
        _colored_source_labels(case)
    ):
        raise StudyError("compact ordinal-1 flow has the wrong external axis")
    helicities = fixed_selector_helicity(case.pdgs)
    selector = Selector(
        flow_word=flow_word,
        flow_id=flow_id,
        helicities=helicities,
        helicity_id=selector_helicity_id(helicities),
    )
    contract = SelectorContract(
        selected_color_flow_ids=(selector.flow_id,),
        selected_color_words=(selector.flow_word,),
        all_flow_helicity_ids=(selector.helicity_id,),
        all_flow_source_helicities=tuple(enumerate(selector.helicities, start=1)),
        point_digest=point_digest(points),
    )
    context["selector_source"] = "compact-seed-one-based-color-ordinal-1"
    context["resolved_semantic_flow_id"] = flow_id
    return selector, contract, context


def _reference_cell(case: Case, workload: str) -> object:
    expected_workload = (
        Workload.SELECTED_FLOW if workload == "selected" else Workload.ALL_FLOW
    )
    matches = tuple(
        cell
        for cell in REPORT_CATALOG.reference_cells()
        if cell.process_key == case.process_key
        and cell.n_final == case.multiplicity
        and cell.measurement.accuracy is Accuracy.LC
        and cell.workload is expected_workload
    )
    if len(matches) != 1:
        raise StudyError("AmpliCol reference catalog cell is not unique")
    cell = matches[0]
    if cell.process != case.process:
        raise StudyError("AmpliCol reference process differs from the process table")
    return cell


def _required_reference_number(
    measurement: Mapping[str, object], field: str, *, allow_zero: bool = False
) -> float:
    value = measurement.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StudyError(f"AmpliCol reference omitted {field}")
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or (result == 0.0 and not allow_zero):
        raise StudyError(f"AmpliCol reference has invalid {field}")
    return result


def _required_reference_sample_count(measurement: Mapping[str, object]) -> int:
    value = measurement.get("sample_count")
    if isinstance(value, bool) or not isinstance(value, int) or value < MINIMUM_SAMPLES:
        raise StudyError("AmpliCol reference has insufficient actual samples")
    return value


def _active_report_source() -> ReportSourceIdentity:
    try:
        return require_eligible_report_source(ROOT)
    except ReportSourceIdentityError as error:
        raise StudyError(f"active report source is ineligible: {error}") from error


def _authenticate_reference_source(
    provenance: Mapping[str, object], active_source: ReportSourceIdentity
) -> None:
    """Keep the current exact source-lineage policy isolated for later audit."""

    source_fields = {
        "revision": PINNED_REFERENCE_REVISION,
        "report_source_revision": active_source.revision,
        "report_measured_source_revision": active_source.revision,
        "report_source_tree": active_source.tree,
        "report_measured_source_tree": active_source.tree,
        "report_source_clean": True,
    }
    if any(
        provenance.get(field) != expected for field, expected in source_fields.items()
    ):
        raise StudyError("AmpliCol reference source differs from the active source")


def _reference_timing_contract(
    provenance: Mapping[str, object],
    case: Case,
    workload: str,
    target_runtime: float,
) -> dict[str, object]:
    manual = provenance.get("manual_campaign")
    if not isinstance(manual, Mapping):
        raise StudyError("AmpliCol reference omitted manual-campaign timing evidence")
    expected_workload = (
        Workload.SELECTED_FLOW if workload == "selected" else Workload.ALL_FLOW
    )
    expected_cell = _reference_cell(case, workload)
    expected_identity = {
        "accuracy": "lc",
        "backend": "fortran",
        "cell_id": expected_cell.cell_id,
        "dataset_id": "reference_amplicol_lc",
        "execution_mode": "amplicol",
        "model": None,
        "n_final": case.multiplicity,
        "process": case.process,
        "process_key": case.process_key,
        "variant": None,
        "workload": expected_workload.value,
    }
    identity = manual.get("cell_identity")
    batch_size = manual.get("batch_size")
    warmup_runs = manual.get("warmup_runs")
    minimum_samples = manual.get("minimum_samples")
    recorded_target = manual.get("target_runtime_seconds")
    if not isinstance(identity, Mapping) or dict(identity) != expected_identity:
        raise StudyError("AmpliCol manual-campaign cell identity is mismatched")
    if batch_size != BATCH_SIZE:
        raise StudyError("AmpliCol manual-campaign batch size is mismatched")
    if warmup_runs != WARMUP_RUNS:
        raise StudyError("AmpliCol manual-campaign warm-up count is mismatched")
    if (
        isinstance(minimum_samples, bool)
        or not isinstance(minimum_samples, int)
        or minimum_samples < MINIMUM_SAMPLES
    ):
        raise StudyError("AmpliCol manual-campaign sample minimum is insufficient")
    if (
        isinstance(recorded_target, bool)
        or not isinstance(recorded_target, (int, float))
        or float(recorded_target) != target_runtime
    ):
        raise StudyError("AmpliCol manual-campaign target runtime is mismatched")
    if provenance.get("generation_timing_is_workload_specific") is not True:
        raise StudyError("AmpliCol generation timing is not workload-specific")
    return {
        "batch_size": BATCH_SIZE,
        "warmup_runs": WARMUP_RUNS,
        "minimum_samples": int(minimum_samples),
        "target_runtime_seconds": target_runtime,
        "cell_identity": expected_identity,
        "generation_timing_is_workload_specific": True,
    }


def _rebind_reference_selector(
    raw_contract: Mapping[str, object], case: Case, points: Points
) -> tuple[SelectorContract, SelectorContract, Selector]:
    try:
        stored_contract = SelectorContract.from_mapping(raw_contract)
    except (TypeError, ValueError) as error:
        raise StudyError("AmpliCol reference selector contract is invalid") from error
    singleton_digest = point_digest((points[0],))
    if stored_contract.point_digest != singleton_digest:
        raise StudyError("AmpliCol selector digest is not the seed-101 singleton")
    rebound_contract = replace(stored_contract, point_digest=point_digest(points))
    if {
        field: value
        for field, value in rebound_contract.as_dict().items()
        if field != "point_digest"
    } != {
        field: value
        for field, value in stored_contract.as_dict().items()
        if field != "point_digest"
    }:
        raise StudyError("AmpliCol selector rebinding changed a physical axis")
    return (
        stored_contract,
        rebound_contract,
        _selector_from_contract(case, rebound_contract),
    )


def _load_amplicol_reference(
    repo_root: Path,
    profile: str,
    case: Case,
    workload: str,
    points: Points,
    target_runtime: float,
    active_source: ReportSourceIdentity,
) -> AmplicolReference:
    root = _existing(repo_root, directory=True, label="reference repository root")
    try:
        validated_profile = validate_profile_name(profile)
        docs = root / "docs" / "performance_reports" / validated_profile
        artifacts = docs / "campaign_artifacts"
        paths = ReportPaths.from_repo(
            root,
            docs_dir=docs,
            artifact_root=artifacts,
            coordination_root=artifacts / "coordination",
        )
    except ValueError as error:
        raise StudyError(f"invalid AmpliCol reference profile: {profile!r}") from error
    for path, label in (
        (paths.docs_dir, "reference report profile"),
        (paths.artifact_root, "reference artifact root"),
        (paths.coordination_root, "reference coordination root"),
    ):
        _existing(path, directory=True, label=label)
    cell = _reference_cell(case, workload)
    store = ArtifactStore(
        artifact_root=paths.artifact_root,
        lock_root=paths.coordination_root,
        current_publication_paths=paths,
    )
    try:
        current = store.load_current(cell.cell_id)
        assert current is not None
        validate_measurement(current.result, expected_cell=cell)
    except (ArtifactStoreError, FileNotFoundError, ValueError) as error:
        raise StudyError(
            f"AmpliCol reference current is not authenticated: {cell.cell_id}"
        ) from error
    if current.cell_id != cell.cell_id or current.result.get("status") != "ok":
        raise StudyError("AmpliCol reference current has the wrong workload identity")

    raw_contract = current.result.get("selector_contract")
    if not isinstance(raw_contract, Mapping):
        raise StudyError("AmpliCol reference omitted its selector contract")
    stored_contract, rebound_contract, selector = _rebind_reference_selector(
        raw_contract, case, points
    )

    measurement = current.result
    timing = {
        "execution_mode": "amplicol",
        "cell_id": cell.cell_id,
        "generation_seconds": _required_reference_number(
            measurement, "generation_seconds"
        ),
        "wall_seconds_per_point": _required_reference_number(
            measurement, "wall_seconds_per_point"
        ),
        "evaluator_seconds_per_point": _required_reference_number(
            measurement, "execution_seconds_per_point"
        ),
        "relative_standard_error": _required_reference_number(
            measurement, "relative_standard_error", allow_zero=True
        ),
        "sample_count": _required_reference_sample_count(measurement),
    }
    matrix_element = _required_reference_number(
        measurement, "matrix_element", allow_zero=True
    )
    raw_artifact = measurement.get("artifact")
    artifact_path = (
        raw_artifact.get("path") if isinstance(raw_artifact, Mapping) else None
    )
    process_row = (
        raw_artifact.get("process_row") if isinstance(raw_artifact, Mapping) else None
    )
    row_match = (
        re.fullmatch(r"group:([1-9][0-9]*):integral:([1-9][0-9]*)", process_row)
        if isinstance(process_row, str)
        else None
    )
    if not isinstance(artifact_path, str) or row_match is None:
        raise StudyError("AmpliCol reference omitted its generated-library row")
    recorded_artifact_path = _existing(
        Path(artifact_path),
        directory=True,
        label="AmpliCol current-owned artifact",
    )
    expected_artifact_path = _existing(
        current.result_path.parent / "artifact",
        directory=True,
        label="AmpliCol authenticated attempt artifact",
    )
    if recorded_artifact_path != expected_artifact_path:
        raise StudyError("AmpliCol reference artifact is outside its current attempt")
    library_path = _existing(
        recorded_artifact_path / "selected-flow-generated-library",
        directory=True,
        label="AmpliCol selected-flow generated-library snapshot",
    )
    executable = _existing(
        library_path / "amplicol_library_benchmark",
        directory=False,
        label="AmpliCol selected-flow replay executable",
    )
    if not os.access(executable, os.X_OK):
        raise StudyError("AmpliCol selected-flow replay executable is not executable")
    _existing(
        library_path / "processes.txt",
        directory=False,
        label="AmpliCol selected-flow process table",
    )
    _existing(
        library_path / "Library",
        directory=True,
        label="AmpliCol selected-flow serialized library",
    )
    if not tuple(library_path.glob("libamp*.so")):
        raise StudyError("AmpliCol selected-flow snapshot has no shared libraries")
    parsed_process_row = (int(row_match.group(1)), int(row_match.group(2)))
    provenance = measurement.get("provenance")
    if not isinstance(provenance, Mapping):
        raise StudyError("AmpliCol reference omitted source provenance")
    _authenticate_reference_source(provenance, active_source)
    timing_contract = _reference_timing_contract(
        provenance, case, workload, target_runtime
    )
    return AmplicolReference(
        selector=selector,
        contract=rebound_contract,
        selector_contract=rebound_contract.as_dict(),
        matrix_element=matrix_element,
        timing=timing,
        lineage={
            "profile": profile,
            "repo_root": str(paths.repo_root),
            "artifact_root": str(paths.artifact_root),
            "cell_id": cell.cell_id,
            "attempt_id": current.attempt_id,
            "manifest_sha256": current.manifest_sha256,
            "result_path": str(current.result_path),
            "process_table_id": case.process_table_id,
            "process_key": case.process_key,
            "process": case.process,
            "multiplicity": case.multiplicity,
            "workload": workload,
            "source_revision": active_source.revision,
            "source_tree": active_source.tree,
            "original_amplicol_revision": PINNED_REFERENCE_REVISION,
            "timing_contract": timing_contract,
            "stored_selector_point_digest": stored_contract.point_digest,
            "correctness_selector_point_digest": rebound_contract.point_digest,
            "selector_rebinding": "point-digest-only",
            "stored_matrix_element": matrix_element,
            "selected_flow_generated_library": str(library_path),
            "process_row": {
                "group": parsed_process_row[0],
                "integral": parsed_process_row[1],
            },
        },
        library_path=library_path,
        process_row=parsed_process_row,
    )


def _lower_hex(value: object, length: int, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise StudyError(f"{label} is not a lowercase hexadecimal identity")
    return value


def _artifact_identity(
    path: Path,
    runtime: object,
    case: Case,
    expected_mode: str,
) -> dict[str, object]:
    try:
        manifest = load_manifest(path, verify_payloads=False)
    except (ArtifactError, OSError) as error:
        raise StudyError(f"cannot load artifact manifest: {path}") from error
    records = tuple(manifest.processes)
    if len(records) != 1:
        raise StudyError("study artifacts must contain exactly one process")
    process = records[0]
    if (
        process.get("id") != case.process_id
        or process.get("expression") != case.process
        or tuple(process.get("external_pdgs", ())) != case.pdgs
        or process.get("color_accuracy") != "lc"
    ):
        raise StudyError("artifact manifest has the wrong process identity")

    artifact_id = _lower_hex(manifest.artifact_id, 64, "manifest artifact ID")
    if (
        getattr(runtime, "artifact_id", None) != artifact_id
        or getattr(runtime, "execution_mode", None) != expected_mode
        or getattr(runtime, "representative_process_key", None) != case.process_id
    ):
        raise StudyError("runtime identity differs from its artifact manifest")
    backend = getattr(runtime, "_backend", None)
    native = getattr(backend, "_runtime", None)
    native_artifact_id = getattr(native, "artifact_id", None)
    metadata = getattr(backend, "_native_metadata", None)
    if native_artifact_id != artifact_id or not isinstance(metadata, Mapping):
        raise StudyError("native runtime omitted its authenticated artifact identity")
    if (
        metadata.get("execution_mode") != expected_mode
        or metadata.get("process") != case.process
        or metadata.get("process_key") != case.process_id
        or metadata.get("representative_process") != case.process
        or metadata.get("representative_process_key") != case.process_id
        or metadata.get("color_accuracy") != "lc"
        or tuple(metadata.get("external_pdg_order", ())) != case.pdgs
        or metadata.get("external_count") != len(case.pdgs)
    ):
        raise StudyError("native runtime metadata has the wrong process identity")

    producer = manifest.producer
    source_revision = _lower_hex(
        producer.get("git_revision"), 40, "producer source revision"
    )
    native_digest = _lower_hex(
        producer.get("native_build_inputs_sha256"),
        64,
        "producer native-input digest",
    )
    model_identity = dict(manifest.model)
    return {
        "path": str(manifest.root),
        "artifact_id": artifact_id,
        "native_authenticated_artifact_id": native_artifact_id,
        "execution_mode": expected_mode,
        "process_id": case.process_id,
        "process_expression": case.process,
        "external_pdgs": list(case.pdgs),
        "producer_identity": {
            "source_revision": source_revision,
            "native_build_inputs_sha256": native_digest,
        },
        "model_identity": model_identity,
    }


def _common_producer_identity(
    artifacts: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    identities = tuple(record.get("producer_identity") for record in artifacts.values())
    if not identities or any(not isinstance(value, Mapping) for value in identities):
        raise StudyError("artifact producer identity evidence is incomplete")
    first = dict(identities[0])  # type: ignore[arg-type]
    if any(dict(value) != first for value in identities[1:]):  # type: ignore[arg-type]
        raise StudyError("study artifacts were produced by different native sources")
    return first


def _common_model_identity(
    artifacts: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    identities = tuple(record.get("model_identity") for record in artifacts.values())
    if not identities or any(not isinstance(value, Mapping) for value in identities):
        raise StudyError("artifact model identity evidence is incomplete")
    first = dict(identities[0])  # type: ignore[arg-type]
    if any(dict(value) != first for value in identities[1:]):  # type: ignore[arg-type]
        raise StudyError("study artifacts were generated from different models")
    return first


def _candidate_source_binding(
    active: ReportSourceIdentity, producer: Mapping[str, object]
) -> dict[str, object]:
    producer_revision = _lower_hex(
        producer.get("source_revision"), 40, "candidate producer source revision"
    )
    if active.revision != producer_revision:
        raise StudyError("generated candidate differs from the active source")
    return {
        "status": "passed",
        "policy": "exact-source-revision",
        "source_revision": producer_revision,
        "source_tree": active.tree,
    }


def _config(mode: str, layout: str) -> RunConfig:
    if mode not in {"on-the-fly", "recurrence"}:
        raise StudyError(f"high-multiplicity harness forbids execution mode {mode!r}")
    return RunConfig(
        action="generate",
        color=ColorConfig(accuracy="lc", lc_flow_layout=layout),
        generation=GenerationConfig(
            workers=1,
            emit_api_bundle=False,
            relation_discovery=GenerationRelationDiscoveryConfig(mode="off"),
            validation=GenerationValidationConfig(
                enabled=False, post_build_validation=False
            ),
        ),
        evaluator=EvaluatorConfig(
            backend="jit",
            execution_mode=mode,
            optimization=EvaluatorOptimizationConfig(cores=1),
            jit=JITConfig(optimization_level=2),
        ),
    )


def _generate_otf(case: Case, path: Path, model: object) -> dict[str, object]:
    original = generation_service._invoke_rust_on_the_fly_seed_batch_builder_v1
    seed_calls = seed_count = materialized_calls = 0

    def capture(*args: object, **kwargs: object) -> object:
        nonlocal seed_calls, seed_count
        seed_calls += 1
        projections = n4._on_the_fly_source_projections(args, kwargs)
        result = original(*args, **kwargs)
        if not isinstance(result, tuple) or len(result) != len(projections):
            raise StudyError("OTF seed binding changed the process batch")
        seed_count += len(result)
        return result

    def reject(name: str) -> Callable[..., object]:
        def operation(*_args: object, **_kwargs: object) -> object:
            nonlocal materialized_calls
            materialized_calls += 1
            raise StudyError(f"OTF generation entered materialized lane {name}")

        return operation

    started = time.perf_counter()
    with ExitStack() as patches:
        patches.enter_context(
            mock.patch.object(
                generation_service,
                "_invoke_rust_on_the_fly_seed_batch_builder_v1",
                capture,
            )
        )
        for name, owner, attribute in n4._materialized_process_lane_patch_targets():
            patches.enter_context(mock.patch.object(owner, attribute, reject(name)))
        Generator(_config("on-the-fly", "topology-replay")).generate(
            ProcessRequest.parse(case.process, name=case.process_id), path, model=model
        )
    if (seed_calls, seed_count, materialized_calls) != (1, 1, 0):
        raise StudyError("OTF generation did not create exactly one compact seed")
    return {
        "seconds": time.perf_counter() - started,
        "seed_binding_calls": seed_calls,
        "seed_count": seed_count,
        "materialized_lane_calls": materialized_calls,
    }


def _generate_recurrence(case: Case, path: Path, model: object) -> dict[str, object]:
    started = time.perf_counter()
    Generator(_config("recurrence", "topology-replay")).generate(
        ProcessRequest.parse(case.process, name=case.process_id), path, model=model
    )
    return {
        "execution_mode": "recurrence",
        "layout": "topology-replay",
        "seconds": time.perf_counter() - started,
    }


def _census(runtime: object, process_id: str) -> dict[str, Any]:
    native = getattr(getattr(runtime, "_backend", None), "_runtime", None)
    operation = getattr(native, STATE_METHOD, None)
    if not callable(operation):
        raise StudyError(f"OTF runtime does not expose {STATE_METHOD}")
    raw = operation()
    try:
        value = json.loads(raw) if isinstance(raw, str) else None
    except json.JSONDecodeError as error:
        raise StudyError("OTF runtime census is invalid JSON") from error
    if not isinstance(value, dict):
        raise StudyError("OTF runtime census is unavailable")
    if value.get("kind") != STATE_KIND or value.get("process_id") != process_id:
        raise StudyError("OTF runtime census has the wrong identity")
    for field in STATE_COUNTS:
        count = value.get(field)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise StudyError(f"OTF runtime census has invalid {field}")
    active = value.get("active_family_union_census")
    if active is not None and not isinstance(active, Mapping):
        raise StudyError("OTF active-family census is invalid")
    return value


def _assert_cold(value: Mapping[str, object], label: str) -> None:
    if (
        any(value[field] != 0 for field in STATE_COUNTS)
        or value.get("active_family_union_census") is not None
    ):
        raise StudyError(f"{label} retained mutable OTF state")


def _active_family_census(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise StudyError(f"{label} has no active family-union census")
    if (
        value.get("basis") != "shared-query-family-union-v1"
        or value.get("scope") != "active-family-union"
    ):
        raise StudyError(f"{label} has the wrong active family-union identity")
    result = dict(value)
    for field in ACTIVE_COUNT_FIELDS:
        count = result.get(field)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise StudyError(f"{label} has invalid active-family {field}")
    if (
        result["query_count"] < 1
        or result["union_unique_current_count"]
        > result["union_unique_current_component_count"]
        or result["union_amplitude_destination_count"] < 1
        or result["union_amplitude_destination_count"] > result["query_count"]
    ):
        raise StudyError(f"{label} has inconsistent active-family dimensions")
    for role in OPERATION_ROLES:
        if result[f"union_{role}_executor_call_groups"] > result[f"union_{role}_rows"]:
            raise StudyError(f"{label} has more {role} call groups than rows")
    return result


def _assert_family_state(
    value: Mapping[str, object],
    label: str,
    *,
    families: int,
    selections: int,
    handles: int,
    minimum_bindings: int,
) -> dict[str, object]:
    if (
        value["process_preparation_count"] != 1
        or value["retained_family_count"] != families
        or value["retained_selection_count"] != selections
        or value["retained_executor_handle_count"] != handles
        or value["semantic_executor_binding_count"] < minimum_bindings
        or any(value[field] != 0 for field in COMPACT_ZERO_COUNTS)
        or value["retained_request_count"] < 1
        or value["retained_amplitude_destination_count"] < 1
        or value["retained_amplitude_destination_count"]
        > value["retained_request_count"]
    ):
        raise StudyError(f"{label} has an invalid compact retained-state census")
    active = _active_family_census(value.get("active_family_union_census"), label)
    if (
        active["query_count"] > value["retained_request_count"]
        or active["union_amplitude_destination_count"]
        > value["retained_amplitude_destination_count"]
    ):
        raise StudyError(f"{label} active family exceeds retained requests")
    return active


def _same_counts(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    return all(left[field] == right[field] for field in STATE_COUNTS)


def _selector_kwargs(workload: str, selector: Selector) -> dict[str, tuple[str, ...]]:
    if workload == "selected":
        return {"color_flows": (selector.flow_id,)}
    if workload == "all-flow":
        return {"helicities": (selector.helicity_id,)}
    if workload == "exact":
        return {
            "color_flows": (selector.flow_id,),
            "helicities": (selector.helicity_id,),
        }
    raise StudyError(f"unknown study workload {workload!r}")


def _evaluate(
    runtime: object,
    points: Points,
    workload: str,
    selector: Selector,
    *,
    resolved: bool,
    _precomputed_total: tuple[object, ...] | None = None,
) -> Evaluation:
    kwargs = _selector_kwargs(workload, selector)
    total = (
        tuple(runtime.evaluate(points, **kwargs))
        if _precomputed_total is None
        else _precomputed_total
    )
    detail = runtime.evaluate_resolved(points, **kwargs) if resolved else None
    total_vs_resolved = None
    if len(total) != len(points):
        raise StudyError("runtime returned the wrong point count")
    if detail is not None:
        axes = (
            (("color_ids", (selector.flow_id,)),)
            if workload == "selected"
            else (("helicity_ids", (selector.helicity_id,)),)
            if workload == "all-flow"
            else (
                ("color_ids", (selector.flow_id,)),
                ("helicity_ids", (selector.helicity_id,)),
            )
        )
        if any(tuple(getattr(detail, axis, ())) != expected for axis, expected in axes):
            raise StudyError("resolved output changed the public selector ID")
        resolved_total = getattr(detail, "total", None)
        if not callable(resolved_total):
            raise StudyError("resolved output omitted its public total")
        try:
            total_vs_resolved = n4._series(
                total,
                tuple(resolved_total()),
                f"{workload} public total/resolved total",
            )
        except n4.GateError as error:
            raise StudyError(str(error)) from error
    return Evaluation(total, detail, total_vs_resolved)


def _first_selected_evaluation(
    runtime: object,
    points: Points,
    selector: Selector,
    requested_workload: str,
    *,
    resolved: bool,
) -> Evaluation:
    """Build selected family A for lifecycle checks without timing it."""

    if not points:
        raise StudyError("selected lifecycle evaluation requires at least one point")
    kwargs = _selector_kwargs("selected", selector)
    total = tuple(runtime.evaluate(points, **kwargs))
    return _evaluate(
        runtime,
        points,
        "selected",
        selector,
        resolved=resolved and requested_workload == "selected",
        _precomputed_total=total,
    )


def _requested_workload_cold_warmup(
    runtime: object,
    case: Case,
    selector: Selector,
    points: Points,
    workload: str,
    *,
    ratio_eligible: bool,
) -> dict[str, object]:
    """Time one requested-workload call on a cold 128xseed-101 batch."""

    cold = _census(runtime, case.process_id)
    _assert_cold(cold, "requested-workload warm-up cold runtime")
    batch = _timing_points(points)
    kwargs = _selector_kwargs(workload, selector)
    started_ns = time.perf_counter_ns()
    total = tuple(runtime.evaluate(batch, **kwargs))
    elapsed_ns = time.perf_counter_ns() - started_ns
    if elapsed_ns < 0 or len(total) != BATCH_SIZE:
        raise StudyError("requested-workload cold warm-up returned invalid evidence")
    warmed = _census(runtime, case.process_id)
    active = _assert_family_state(
        warmed,
        f"cold {workload} warm-up",
        families=1,
        selections=1,
        handles=1,
        minimum_bindings=1,
    )
    runtime.clear()
    cleared = _census(runtime, case.process_id)
    _assert_cold(cleared, "requested-workload warm-up clear")
    seconds = elapsed_ns / 1_000_000_000.0
    return {
        "kind": "on-the-fly-requested-workload-cold-warmup-v1",
        "timer": "time.perf_counter_ns",
        "runtime_state": "census-proven-cold",
        "requested_workload": workload,
        "batch_size": BATCH_SIZE,
        "point_count": len(batch),
        "seed": SEEDS[0],
        "singleton_seed_point_digest": point_digest((points[0],)),
        "batch_point_policy": "seed-101 correctness point 0 repeated 128 times",
        "elapsed_nanoseconds": elapsed_ns,
        "seconds": seconds,
        "seconds_per_point": seconds / BATCH_SIZE,
        "output_count": len(total),
        "before": cold,
        "after_first_evaluation": warmed,
        "active_family_union_census": active,
        "after_clear": cleared,
        "clear_restored_cold_state": True,
        "excluded_from_elapsed": [
            "Runtime.load",
            "artifact generation",
            "resolved-output follow-up",
            "correctness lifecycle",
            "warmed BenchmarkRunner",
        ],
        "ratio_eligible": ratio_eligible,
        "generation_pair_eligible": True,
        "acceptance_eligible": False,
    }


def _compare(left: Evaluation, right: Evaluation, label: str) -> dict[str, object]:
    try:
        return {
            "total": n4._series(left.total, right.total, f"{label} total"),
            "resolved": (
                None
                if left.resolved is None or right.resolved is None
                else n4._resolved_component_checks(
                    left.resolved, right.resolved, f"{label} resolved"
                )
            ),
        }
    except n4.GateError as error:
        raise StudyError(str(error)) from error


def _amplicol_singleton_check(
    evaluation: Evaluation,
    reference: AmplicolReference,
    points: Points,
) -> dict[str, object]:
    if not evaluation.total or not points:
        raise StudyError("AmpliCol singleton comparison has no seed-101 result")
    try:
        comparison = n4._series(
            (evaluation.total[0],),
            (reference.matrix_element,),
            "OTF seed-101/AmpliCol stored matrix_element",
            n4.AMPICOL_REL_TOL,
        )
    except n4.GateError as error:
        raise StudyError(str(error)) from error
    return {
        "status": "passed",
        "seed": SEEDS[0],
        "point_count": 1,
        "point_digest": point_digest((points[0],)),
        "stored_amplicol_matrix_element": reference.matrix_element,
        "otf_matrix_element": float(complex(evaluation.total[0]).real),
        "comparison": comparison,
    }


def _amplicol_series(
    candidate: Sequence[object],
    reference: Sequence[object],
    label: str,
) -> dict[str, object]:
    try:
        return n4._series(candidate, reference, label, n4.AMPICOL_REL_TOL)
    except n4.GateError as error:
        raise StudyError(str(error)) from error


def _replay_amplicol(
    reference: AmplicolReference,
    case: Case,
    points: Points,
) -> tuple[tuple[float, ...], dict[str, object]]:
    """Replay the immutable selected-flow library at every correctness point."""

    library = reference.library_path
    process_row = reference.process_row
    if library is None or process_row is None:
        raise StudyError("AmpliCol reference omitted replayable library evidence")
    try:
        entries = legacy_amplicol.parse_process_file(library / "processes.txt")
        matches = tuple(
            entry
            for entry in entries
            if (int(entry.group), int(entry.integral)) == process_row
        )
        if len(matches) != 1:
            raise StudyError("AmpliCol generated-library process row is not unique")
        entry = matches[0]
        mapped = legacy_amplicol.source_mapped_color_order(
            entry,
            source_pdgs=case.pdgs,
        )
        replay_word = _canonical_mapped_color_word(
            case.pdgs,
            mapped,
            initial_state_count=_initial_state_count(case.process),
        )
        if replay_word != reference.selector.flow_word:
            raise StudyError("AmpliCol replay row differs from its selector contract")
        root = str(library.resolve(strict=True))
        environment: dict[str, str] = {}
        for name in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"):
            existing = os.environ.get(name)
            environment[name] = (
                root if not existing else f"{root}{os.pathsep}{existing}"
            )
        values: list[float] = []
        with mock.patch.dict(os.environ, environment):
            for point in points:
                probe = legacy_amplicol.run_selected_flow_library_probe(
                    library,
                    entry=entry,
                    source_pdgs=case.pdgs,
                    momenta=point,
                    helicities=None,
                    points=1,
                )
                emitted_entry = replace(
                    entry,
                    process_pdgs=tuple(probe.process_pdgs),
                    color_order=tuple(probe.color_order),
                )
                emitted_mapped = legacy_amplicol.source_mapped_color_order(
                    emitted_entry,
                    source_pdgs=case.pdgs,
                )
                emitted_word = _canonical_mapped_color_word(
                    case.pdgs,
                    emitted_mapped,
                    initial_state_count=_initial_state_count(case.process),
                )
                if emitted_word != reference.selector.flow_word:
                    raise StudyError(
                        "AmpliCol replay emitted a different semantic selector flow"
                    )
                values.append(float(probe.value))
    except (LegacyAdapterError, legacy_amplicol.LegacyOracleError, OSError) as error:
        raise StudyError(
            f"AmpliCol generated-library replay failed: {error}"
        ) from error
    replayed = tuple(values)
    if len(replayed) != len(points):
        raise StudyError("AmpliCol generated-library replay has the wrong point axis")
    stored_check = _amplicol_series(
        replayed[:1],
        (reference.matrix_element,),
        "AmpliCol replay seed-101/stored matrix_element",
    )
    return replayed, {
        "status": "passed",
        "source": "authenticated-selected-flow-generated-library-snapshot",
        "library_path": str(library),
        "process_row": {"group": process_row[0], "integral": process_row[1]},
        "selector_flow_id": reference.selector.flow_id,
        "point_count": len(points),
        "point_digest": point_digest(points),
        "seeds": list(SEEDS),
        "helicity_policy": "sum-all-helicities",
        "subprocess_launches": len(points),
        "seed101_replay_vs_stored_matrix_element": stored_check,
    }


def _lifecycle(
    runtime: object,
    case: Case,
    selector: Selector,
    points: Points,
    workload: str,
    resolved: bool,
) -> tuple[Evaluation, Evaluation, dict[str, object]]:
    cold = _census(runtime, case.process_id)
    _assert_cold(cold, "cold runtime")
    selected_a = _first_selected_evaluation(
        runtime,
        points,
        selector,
        workload,
        resolved=resolved,
    )
    census_a = _census(runtime, case.process_id)
    active_a = _assert_family_state(
        census_a,
        "selected A",
        families=1,
        selections=1,
        handles=1,
        minimum_bindings=1,
    )
    if workload == "selected":
        exact_c = _evaluate(runtime, points, "exact", selector, resolved=False)
        census_c = _census(runtime, case.process_id)
        _assert_family_state(
            census_c,
            "exact-selector C",
            families=2,
            selections=2,
            handles=2,
            minimum_bindings=int(census_a["semantic_executor_binding_count"]),
        )
        repeated = _evaluate(runtime, points, "exact", selector, resolved=False)
        repeated_census = _census(runtime, case.process_id)
        if repeated_census != census_c:
            raise StudyError("repeated exact-selector C did not plateau")
        _compare(repeated, exact_c, "exact-selector C repeat")
        selected_revisit = _evaluate(
            runtime, points, "selected", selector, resolved=False
        )
        revisit = _census(runtime, case.process_id)
        _assert_family_state(
            revisit,
            "selected A revisit",
            families=2,
            selections=2,
            handles=2,
            minimum_bindings=int(census_c["semantic_executor_binding_count"]),
        )
        if (
            not _same_counts(revisit, census_c)
            or revisit["active_family_union_census"] != active_a
        ):
            raise StudyError("A -> C -> C -> A did not retain both families")
        _compare(selected_revisit, selected_a, "selected A revisit")

        runtime.clear()
        cleared = _census(runtime, case.process_id)
        _assert_cold(cleared, "Runtime.clear()")
        rebuilt = _evaluate(runtime, points, "selected", selector, resolved=resolved)
        rebuilt_a_census = _census(runtime, case.process_id)
        if rebuilt_a_census != census_a:
            raise StudyError("post-clear selected A did not rebuild exactly")
        _evaluate(runtime, points, "exact", selector, resolved=False)
        rebuilt_c_census = _census(runtime, case.process_id)
        if rebuilt_c_census != census_c:
            raise StudyError("post-clear exact-selector C did not rebuild exactly")
        rebuilt_revisit = _evaluate(
            runtime, points, "selected", selector, resolved=False
        )
        rebuilt_census = _census(runtime, case.process_id)
        if rebuilt_census != revisit:
            raise StudyError("post-clear A/C/A retention did not rebuild exactly")
        _compare(rebuilt_revisit, rebuilt, "post-clear selected A revisit")
        _compare(rebuilt, selected_a, "post-clear selected A headline")
        return (
            selected_a,
            rebuilt,
            {
                "cold": cold,
                "selected_a": census_a,
                "requested_family": census_c,
                "requested_repeat": repeated_census,
                "selected_a_revisit": revisit,
                "after_clear": cleared,
                "after_rebuild_selected_a": rebuilt_a_census,
                "after_rebuild_exact_c": rebuilt_c_census,
                "after_rebuild": rebuilt_census,
                "sequence": "A,C,C,A; clear; A,C,A",
            },
        )

    first = _evaluate(runtime, points, "all-flow", selector, resolved=resolved)
    target_census = _census(runtime, case.process_id)
    _assert_family_state(
        target_census,
        "all-flow B",
        families=2,
        selections=2,
        handles=2,
        minimum_bindings=int(census_a["semantic_executor_binding_count"]),
    )
    repeated = _evaluate(runtime, points, "all-flow", selector, resolved=resolved)
    repeated_census = _census(runtime, case.process_id)
    if repeated_census != target_census:
        raise StudyError("repeated all-flow B did not plateau")
    _compare(repeated, first, "all-flow B repeat")
    _evaluate(runtime, points, "selected", selector, resolved=False)
    revisit = _census(runtime, case.process_id)
    _assert_family_state(
        revisit,
        "selected A revisit",
        families=2,
        selections=2,
        handles=2,
        minimum_bindings=int(target_census["semantic_executor_binding_count"]),
    )
    if (
        not _same_counts(revisit, target_census)
        or revisit["active_family_union_census"] != active_a
    ):
        raise StudyError("A -> B -> B -> A did not retain both families")

    runtime.clear()
    cleared = _census(runtime, case.process_id)
    _assert_cold(cleared, "Runtime.clear()")
    _evaluate(runtime, points, "selected", selector, resolved=False)
    rebuilt = _evaluate(runtime, points, "all-flow", selector, resolved=resolved)
    rebuilt_census = _census(runtime, case.process_id)
    if rebuilt_census != target_census:
        raise StudyError("post-clear all-flow B did not rebuild exactly")
    _compare(rebuilt, first, "post-clear all-flow B")
    return (
        first,
        rebuilt,
        {
            "cold": cold,
            "selected_a": census_a,
            "requested_family": target_census,
            "requested_repeat": repeated_census,
            "selected_a_revisit": revisit,
            "after_clear": cleared,
            "after_rebuild": rebuilt_census,
            "sequence": "A,B,B,A; clear; A,B",
        },
    )


def _recurrence_authority_contract(
    case: Case,
    recurrence: object,
    points: Points,
) -> tuple[Selector, SelectorContract, dict[str, object]]:
    if getattr(recurrence, "execution_mode", None) != "recurrence":
        raise StudyError("recurrence authority has the wrong execution mode")
    try:
        contract = derive_selector_contract(recurrence, points)
        validate_selector_contract(recurrence, contract, points)
    except RunnerError as error:
        raise StudyError(str(error)) from error
    return _selector_from_contract(case, contract), contract, contract.as_dict()


def _cross_check_compact_selector(
    runtime: object,
    case: Case,
    selector: Selector,
    *,
    require_ordinal_one: bool,
) -> dict[str, object]:
    requested = (selector.flow_id, "1") if require_ordinal_one else (selector.flow_id,)
    context = _compact_selector_context(runtime, case, requested)
    selected = context["selected_color_ids"]
    if (
        selected != [selector.flow_id] * len(requested)
        or _flow_word(selector.flow_id) != selector.flow_word
    ):
        source = " and compact ordinal 1" if require_ordinal_one else ""
        raise StudyError(f"candidate compact selector differs from authority{source}")
    context["semantic_authority_flow_id"] = selector.flow_id
    context["ordinal_one_required"] = require_ordinal_one
    context["ordinal_one_matches_authority"] = True if require_ordinal_one else None
    return context


def _benchmark(
    runtime: object,
    mode: str,
    points: Points,
    workload: str,
    selector: Selector,
    target: float,
) -> dict[str, object]:
    kwargs = _selector_kwargs(workload, selector)
    config = n4._benchmark_config(
        target,
        BATCH_SIZE,
        helicities=kwargs.get("helicities", ()),
        flows=kwargs.get("color_flows", ()),
    )
    result = BenchmarkRunner(config).run(runtime, points=_timing_points(points))
    effective = result.effective_config
    if result.interrupted:
        raise StudyError("benchmark was interrupted")
    if (
        effective.target_runtime != target
        or effective.precision != 16
        or effective.batch_size != BATCH_SIZE
        or effective.warmup_runs != WARMUP_RUNS
        or effective.minimum_samples < MINIMUM_SAMPLES
        or result.sample_count < MINIMUM_SAMPLES
        or tuple(effective.helicity_ids) != tuple(config.helicity_ids)
        or tuple(effective.color_flow_ids) != tuple(config.color_flow_ids)
    ):
        raise StudyError("benchmark did not satisfy the effective timing contract")
    try:
        timing = n4._public_timing(result, mode)
    except n4.GateError as error:
        raise StudyError(str(error)) from error
    timing["simd"] = {
        "status": "not-proven",
        "reason": "native executed-block telemetry is absent",
    }
    return timing


def _selected_report_lineage(
    path: Path,
    case: Case,
    candidate: Mapping[str, object],
    active_source: ReportSourceIdentity,
) -> dict[str, object]:
    outer = _read(path)
    selected = outer.get("study")
    if (
        outer.get("kind") != f"{KIND}-run"
        or outer.get("schema_version") != SCHEMA_VERSION
        or outer.get("status") != "passed"
        or not isinstance(selected, Mapping)
        or selected.get("kind") != KIND
        or selected.get("schema_version") != SCHEMA_VERSION
        or selected.get("status") != "passed"
        or selected.get("workload") != "selected"
        or selected.get("process") != dataclass_payload(case)
    ):
        raise StudyError("selected report has the wrong study identity")
    expected_authority = _authority_kind(case, "selected")
    authority = selected.get("authority")
    correctness = selected.get("correctness")
    compact = selected.get("compact_selector_context")
    candidate_source_binding = selected.get("candidate_source_binding")
    expected_producer = candidate.get("producer_identity")
    expected_model = candidate.get("model_identity")
    if (
        not isinstance(authority, Mapping)
        or authority.get("kind") != expected_authority
        or not isinstance(correctness, Mapping)
        or not isinstance(compact, Mapping)
        or not isinstance(candidate_source_binding, Mapping)
        or not isinstance(expected_producer, Mapping)
        or not isinstance(expected_model, Mapping)
        or selected.get("producer_identity") != dict(expected_producer)
        or selected.get("model_identity") != dict(expected_model)
        or selected.get("active_source") != active_source.provenance()
        or candidate_source_binding
        != _candidate_source_binding(active_source, expected_producer)
    ):
        raise StudyError("selected report omitted its authority/source lineage")
    if expected_authority == AUTHORITY_OTF_ONLY:
        self_consistency = correctness.get("pre_post_clear_self_consistency")
        correctness_valid = (
            correctness.get("status") == "not-claimed"
            and isinstance(self_consistency, Mapping)
            and isinstance(self_consistency.get("total"), Mapping)
            and self_consistency["total"].get("checks") == len(SEEDS)
        )
        compact_valid = compact.get(
            "selector_source"
        ) == "compact-seed-one-based-color-ordinal-1" and compact.get(
            "requested_color_ids"
        ) == ["1"]
    else:
        before_key, after_key = (
            ("otf_before_vs_amplicol_replay", "otf_after_vs_amplicol_replay")
            if expected_authority == AUTHORITY_AMPLICOL
            else ("otf_before_vs_recurrence", "otf_after_vs_recurrence")
        )
        before_comparison = correctness.get(before_key)
        after_comparison = correctness.get(after_key)
        if expected_authority == AUTHORITY_AMPLICOL:
            authority_checks_valid = (
                isinstance(before_comparison, Mapping)
                and before_comparison.get("checks") == len(SEEDS)
                and isinstance(after_comparison, Mapping)
                and after_comparison.get("checks") == len(SEEDS)
            )
        else:
            authority_checks_valid = (
                isinstance(before_comparison, Mapping)
                and isinstance(before_comparison.get("total"), Mapping)
                and before_comparison["total"].get("checks") == len(SEEDS)
                and isinstance(after_comparison, Mapping)
                and isinstance(after_comparison.get("total"), Mapping)
                and after_comparison["total"].get("checks") == len(SEEDS)
            )
        correctness_valid = (
            correctness.get("status") == "passed" and authority_checks_valid
        )
        compact_valid = (
            compact.get("ordinal_one_required") is False
            and compact.get("ordinal_one_matches_authority") is None
        )
    if not correctness_valid or not compact_valid:
        raise StudyError("selected report omitted its correctness/selector proof")
    watchdog = outer.get("watchdog")
    if (
        not isinstance(watchdog, Mapping)
        or watchdog.get("passes") is not True
        or watchdog.get("limit_bytes") != WATCHDOG_BYTES
    ):
        raise StudyError("selected report omitted its 30 GiB watchdog proof")
    generation = selected.get("generation")
    on_the_fly = (
        generation.get("on_the_fly") if isinstance(generation, Mapping) else None
    )
    if (
        not isinstance(on_the_fly, Mapping)
        or on_the_fly.get("seed_binding_calls") != 1
        or on_the_fly.get("seed_count") != 1
        or on_the_fly.get("materialized_lane_calls") != 0
    ):
        raise StudyError("selected report does not prove one compact OTF seed")
    raw_contract = selected.get("selector_contract")
    if not isinstance(raw_contract, Mapping):
        raise StudyError("selected report omitted its selector contract")
    try:
        selected_contract = SelectorContract.from_mapping(raw_contract).as_dict()
    except (TypeError, ValueError) as error:
        raise StudyError("selected report selector contract is invalid") from error
    selected_flow_ids = selected_contract["selected_color_flow_ids"]
    if not isinstance(selected_flow_ids, list) or len(selected_flow_ids) != 1:
        raise StudyError("selected report selector contract has the wrong flow axis")
    selected_flow_id = selected_flow_ids[0]
    if expected_authority == AUTHORITY_OTF_ONLY:
        compact_contract_valid = (
            compact.get("requested_color_ids") == ["1"]
            and compact.get("selected_color_ids") == [selected_flow_id]
            and compact.get("resolved_semantic_flow_id") == selected_flow_id
        )
    else:
        compact_contract_valid = (
            compact.get("requested_color_ids") == [selected_flow_id]
            and compact.get("selected_color_ids") == [selected_flow_id]
            and compact.get("semantic_authority_flow_id") == selected_flow_id
        )
    if not compact_contract_valid:
        raise StudyError("selected report selector contract and compact proof disagree")
    artifacts = selected.get("artifacts")
    recorded = artifacts.get("candidate") if isinstance(artifacts, Mapping) else None
    recorded_path = recorded.get("path") if isinstance(recorded, Mapping) else None
    current_path = candidate.get("path")
    if not isinstance(recorded_path, str) or not isinstance(current_path, str):
        raise StudyError("selected report omitted the candidate artifact path")
    try:
        canonical = str(Path(recorded_path).expanduser().resolve(strict=True))
    except OSError as error:
        raise StudyError("selected report candidate artifact is unavailable") from error
    if (
        recorded_path != canonical
        or current_path != canonical
        or recorded.get("artifact_id") != candidate.get("artifact_id")
    ):
        raise StudyError("selected report and --candidate-artifact disagree")
    generation_seconds = on_the_fly.get("seconds")
    if (
        isinstance(generation_seconds, bool)
        or not isinstance(generation_seconds, (int, float))
        or not math.isfinite(float(generation_seconds))
        or float(generation_seconds) <= 0.0
    ):
        raise StudyError("selected report omitted OTF generation timing")
    return {
        "path": str(path.expanduser().resolve(strict=True)),
        "kind": outer["kind"],
        "status": "passed",
        "candidate_path": canonical,
        "candidate_artifact_id": candidate["artifact_id"],
        "selected_authority_kind": expected_authority,
        "producer_identity": dict(expected_producer),
        "model_identity": dict(expected_model),
        "active_source": active_source.provenance(),
        "correctness_status": correctness["status"],
        "compact_selector_proof": dict(compact),
        "watchdog": dict(watchdog),
        "seed_binding_calls": 1,
        "seed_count": 1,
        "on_the_fly_generation_seconds": float(generation_seconds),
        "selector_contract": selected_contract,
    }


def _generation_reporting(
    workload: str,
    *,
    generation_seconds: float,
    warmup_seconds: float,
    amplicol_generation_seconds: float | None,
    source: str,
) -> tuple[dict[str, object], dict[str, float]]:
    if workload not in {"selected", "all-flow"}:
        raise StudyError(f"unknown generation-report workload {workload!r}")
    if (
        not math.isfinite(generation_seconds)
        or generation_seconds <= 0.0
        or not math.isfinite(warmup_seconds)
        or warmup_seconds < 0.0
    ):
        raise StudyError("OTF generation/warm-up comparison timing is invalid")
    has_amplicol = amplicol_generation_seconds is not None
    comparison: dict[str, object] = {
        "notation": "([xG] x(G+W))" if has_amplicol else "[G] G+W",
        "reporting": (
            "ratios-to-amplicol" if has_amplicol else "absolute-on-the-fly-seconds"
        ),
        "source": source,
        "warmup_source": f"this-{workload}-cold-batch128-run",
        "on_the_fly_generation_seconds": generation_seconds,
        "on_the_fly_warmup_seconds": warmup_seconds,
        "on_the_fly_generation_plus_warmup_seconds": (
            generation_seconds + warmup_seconds
        ),
    }
    if not has_amplicol:
        return comparison, {}
    assert amplicol_generation_seconds is not None
    if (
        not math.isfinite(amplicol_generation_seconds)
        or amplicol_generation_seconds <= 0.0
    ):
        raise StudyError("AmpliCol generation comparison timing is invalid")
    generation_ratios = {
        "generation_only": generation_seconds / amplicol_generation_seconds,
        "generation_plus_warmup": (generation_seconds + warmup_seconds)
        / amplicol_generation_seconds,
    }
    comparison.update(
        {
            "amplicol_generation_seconds": amplicol_generation_seconds,
            "generation_only_over_amplicol": generation_ratios["generation_only"],
            "generation_plus_warmup_over_amplicol": generation_ratios[
                "generation_plus_warmup"
            ],
        }
    )
    return comparison, generation_ratios


def _run_worker(args: argparse.Namespace) -> dict[str, object]:
    _validate_arguments(args)
    case = _case(args.process_id, args.multiplicity)
    authority_kind = _authority_kind(case, args.workload)
    points = _points(case)
    active_source = _active_report_source()
    reference: AmplicolReference | None = None
    if authority_kind == AUTHORITY_AMPLICOL:
        assert args.reference_repo_root is not None
        assert isinstance(args.reference_profile, str)
        reference = _load_amplicol_reference(
            args.reference_repo_root,
            args.reference_profile,
            case,
            "selected",
            points,
            args.target_runtime,
            active_source,
        )
    output = Path(os.path.abspath(args.output.expanduser()))
    generation: dict[str, object] = {}
    model = None
    if args.workload == "selected":
        prepared = _existing(
            args.prepared_model, directory=False, label="prepared model"
        )
        model = ModelSource.from_path(prepared).compile()
        candidate_path = output / "candidate-artifact"
        generation["on_the_fly"] = _generate_otf(case, candidate_path, model)
    else:
        candidate_path = _existing(
            args.candidate_artifact, directory=True, label="candidate artifact"
        )
    candidate = Runtime.load(candidate_path, process=case.process_id)
    artifacts: dict[str, dict[str, object]] = {
        "candidate": _artifact_identity(candidate_path, candidate, case, "on-the-fly")
    }
    candidate_producer_identity = artifacts["candidate"].get("producer_identity")
    if not isinstance(candidate_producer_identity, Mapping):
        raise StudyError("candidate artifact producer identity evidence is incomplete")
    candidate_source_binding = _candidate_source_binding(
        active_source, candidate_producer_identity
    )
    selected_report_lineage = None
    selector: Selector
    selector_contract: SelectorContract
    contract: dict[str, object]
    compact_selector_context: dict[str, object]
    recurrence = None
    if args.workload == "all-flow":
        selected_report_path = _existing(
            args.selected_report, directory=False, label="selected report"
        )
        selected_report_lineage = _selected_report_lineage(
            selected_report_path, case, artifacts["candidate"], active_source
        )
        raw_contract = selected_report_lineage.get("selector_contract")
        if not isinstance(raw_contract, Mapping):
            raise StudyError("selected-report lineage omitted its selector contract")
        try:
            selector_contract = SelectorContract.from_mapping(raw_contract)
        except (TypeError, ValueError) as error:
            raise StudyError("selected-report selector contract is invalid") from error
        if selector_contract.point_digest != point_digest(points):
            raise StudyError("selected-report selector points differ from all-flow")
        selector = _selector_from_contract(case, selector_contract)
        contract = selector_contract.as_dict()
        compact_selector_context = _cross_check_compact_selector(
            candidate, case, selector, require_ordinal_one=False
        )
    elif authority_kind == AUTHORITY_AMPLICOL:
        assert reference is not None
        selector = reference.selector
        selector_contract = reference.contract
        contract = reference.selector_contract
        compact_selector_context = _cross_check_compact_selector(
            candidate, case, selector, require_ordinal_one=False
        )
    elif authority_kind == AUTHORITY_RECURRENCE:
        assert model is not None
        recurrence_path = output / "recurrence-authority"
        generation["recurrence"] = _generate_recurrence(case, recurrence_path, model)
        recurrence = Runtime.load(recurrence_path, process=case.process_id)
        artifacts["recurrence_authority"] = _artifact_identity(
            recurrence_path, recurrence, case, "recurrence"
        )
        selector, selector_contract, contract = _recurrence_authority_contract(
            case, recurrence, points
        )
        compact_selector_context = _cross_check_compact_selector(
            candidate, case, selector, require_ordinal_one=False
        )
    else:
        selector, selector_contract, compact_selector_context = (
            _selector_from_compact_ordinal_one(candidate, case, points)
        )
        contract = selector_contract.as_dict()

    producer_identity = _common_producer_identity(artifacts)
    model_identity = _common_model_identity(artifacts)

    requested_cold_warmup = _requested_workload_cold_warmup(
        candidate,
        case,
        selector,
        points,
        args.workload,
        ratio_eligible=reference is not None or recurrence is not None,
    )
    before, after, lifecycle = _lifecycle(
        candidate,
        case,
        selector,
        points,
        args.workload,
        authority_kind in {AUTHORITY_AMPLICOL, AUTHORITY_RECURRENCE},
    )
    otf_self_consistency = _compare(before, after, "OTF pre/post clear")
    if reference is not None:
        amplicol_values, amplicol_replay = _replay_amplicol(reference, case, points)
        correctness = {
            "status": "passed",
            "authority": "authenticated-amplicol-generated-library-replay",
            "claim_scope": {
                "external_amplicol_point_count": len(points),
                "external_amplicol_seeds": list(SEEDS),
                "otf_lifecycle_point_count": len(points),
                "otf_lifecycle_seeds": list(SEEDS),
            },
            "amplicol_replay": amplicol_replay,
            "otf_seed101_vs_amplicol_matrix_element": _amplicol_singleton_check(
                before, reference, points
            ),
            "otf_before_vs_amplicol_replay": _amplicol_series(
                before.total,
                amplicol_values,
                "OTF before/AmpliCol generated-library replay",
            ),
            "otf_after_vs_amplicol_replay": _amplicol_series(
                after.total,
                amplicol_values,
                "OTF after/AmpliCol generated-library replay",
            ),
            "pre_post_clear_self_consistency": otf_self_consistency,
            "within_runtime_total_vs_resolved": {
                "on_the_fly_before": before.total_vs_resolved,
                "on_the_fly_after": after.total_vs_resolved,
            },
        }
    elif recurrence is not None:
        rec = _evaluate(recurrence, points, args.workload, selector, resolved=True)
        correctness = {
            "status": "passed",
            "authority": "fresh-recurrence",
            "pre_post_clear_self_consistency": otf_self_consistency,
            "within_runtime_total_vs_resolved": {
                "on_the_fly_before": before.total_vs_resolved,
                "on_the_fly_after": after.total_vs_resolved,
                "recurrence": rec.total_vs_resolved,
            },
            "otf_before_vs_recurrence": _compare(before, rec, "OTF before/recurrence"),
            "otf_after_vs_recurrence": _compare(after, rec, "OTF after/recurrence"),
        }
    else:
        correctness = {
            "status": "not-claimed",
            "reason": (
                "all-flow is an OTF-only feasibility workload"
                if args.workload == "all-flow"
                else "n=8/9 is OTF-only; numerical parity is not claimed"
            ),
            "pre_post_clear_self_consistency": otf_self_consistency,
        }

    timings = {
        "on_the_fly": _benchmark(
            candidate,
            "on-the-fly",
            points,
            args.workload,
            selector,
            args.target_runtime,
        ),
    }
    if reference is not None:
        timings["amplicol"] = reference.timing
    if _census(candidate, case.process_id) != lifecycle["after_rebuild"]:
        raise StudyError("timed requested family did not plateau")
    if recurrence is not None:
        timings["recurrence"] = _benchmark(
            recurrence,
            "recurrence",
            points,
            args.workload,
            selector,
            args.target_runtime,
        )
    candidate_timing = timings["on_the_fly"]
    candidate_wall = float(candidate_timing["wall_seconds_per_point"])
    candidate_evaluator = float(candidate_timing["evaluator_seconds_per_point"])

    generation_warmup_seconds = float(requested_cold_warmup.get("seconds", -1.0))
    if args.workload == "selected":
        on_the_fly_generation = generation.get("on_the_fly")
        if not isinstance(on_the_fly_generation, Mapping):
            raise StudyError("selected study omitted OTF generation timing")
        generation_seconds = float(on_the_fly_generation.get("seconds", -1.0))
        generation_source = "this-selected-run"
    else:
        if not isinstance(selected_report_lineage, Mapping):
            raise StudyError("all-flow study omitted selected-report lineage")
        generation_seconds = float(
            selected_report_lineage.get("on_the_fly_generation_seconds", -1.0)
        )
        generation_source = "reused-selected-artifact-generation"
    generation_comparison, generation_ratios = _generation_reporting(
        args.workload,
        generation_seconds=generation_seconds,
        warmup_seconds=generation_warmup_seconds,
        amplicol_generation_seconds=(
            None if reference is None else float(reference.timing["generation_seconds"])
        ),
        source=generation_source,
    )
    ratios: dict[str, object] = {}
    if reference is not None:
        runtime_ratio: dict[str, object] = {
            "wall_seconds_per_point": candidate_wall
            / float(reference.timing["wall_seconds_per_point"]),
            "evaluator_seconds_per_point": candidate_evaluator
            / float(reference.timing["evaluator_seconds_per_point"]),
        }
        runtime_ratio.update(generation_ratios)
        ratios["on_the_fly_over_amplicol"] = runtime_ratio
    if recurrence is not None:
        recurrence_timing = timings["recurrence"]
        ratios["on_the_fly_over_recurrence"] = {
            "wall_seconds_per_point": candidate_wall
            / float(recurrence_timing["wall_seconds_per_point"]),
            "evaluator_seconds_per_point": candidate_evaluator
            / float(recurrence_timing["evaluator_seconds_per_point"]),
        }
        recurrence_generation = generation.get("recurrence")
        if not isinstance(recurrence_generation, Mapping):
            raise StudyError("recurrence authority omitted generation timing")
        recurrence_generation_seconds = float(recurrence_generation["seconds"])
        if (
            not math.isfinite(recurrence_generation_seconds)
            or recurrence_generation_seconds <= 0.0
        ):
            raise StudyError("recurrence authority generation timing is invalid")
        recurrence_generation_ratios = {
            "generation_only": generation_seconds / recurrence_generation_seconds,
            "generation_plus_warmup": (generation_seconds + generation_warmup_seconds)
            / recurrence_generation_seconds,
        }
        generation_comparison.update(
            {
                "recurrence_notation": "([xG] x(G+W))",
                "recurrence_generation_seconds": recurrence_generation_seconds,
                "generation_only_over_recurrence": recurrence_generation_ratios[
                    "generation_only"
                ],
                "generation_plus_warmup_over_recurrence": (
                    recurrence_generation_ratios["generation_plus_warmup"]
                ),
            }
        )
        recurrence_ratio = ratios["on_the_fly_over_recurrence"]
        assert isinstance(recurrence_ratio, dict)
        recurrence_ratio.update(recurrence_generation_ratios)
    return {
        "kind": KIND,
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "scope": (
            "otf-only-feasibility"
            if authority_kind == AUTHORITY_OTF_ONLY
            else f"{authority_kind}-authority"
        ),
        "authority": {
            "kind": authority_kind,
            "workload_policy": (
                "all-flow-reuses-selected-artifact-without-comparator"
                if args.workload == "all-flow"
                else "selected-cell-authority"
            ),
        },
        "process": dataclass_payload(case),
        "workload": args.workload,
        "artifacts": artifacts,
        "producer_identity": producer_identity,
        "model_identity": model_identity,
        "amplicol_reference": None if reference is None else reference.lineage,
        "active_source": active_source.provenance(),
        "candidate_source_binding": candidate_source_binding,
        "selected_report_lineage": selected_report_lineage,
        "candidate_reuse": (
            {
                "status": "contract",
                "selected_created_candidate": True,
                "all_flow_must_reuse_this_artifact": True,
            }
            if args.workload == "selected"
            else {
                "status": "actual-reuse",
                "selected_created_candidate": False,
                "all_flow_reused_candidate": True,
            }
        ),
        "selector_contract": contract,
        "compact_selector_context": compact_selector_context,
        "points": {
            "seeds": list(SEEDS),
            "sqrt_s": 1000.0,
            "count": len(points),
            "digest": point_digest(points),
        },
        "generation": generation,
        "correctness": correctness,
        "requested_workload_cold_warmup": requested_cold_warmup,
        "cache_lifecycle": lifecycle,
        "timing_contract": {
            "batch_size": BATCH_SIZE,
            "batch_point_source": "seed-101 correctness point 0 repeated 128 times",
            "correctness_point_count": len(points),
            "correctness_seeds": list(SEEDS),
            "warmup_runs": WARMUP_RUNS,
            "minimum_samples": MINIMUM_SAMPLES,
            "target_runtime_seconds": args.target_runtime,
            "runtime_notation": "([evaluator] wall)",
            "performance_is_acceptance_gate": False,
        },
        "timings": timings,
        "generation_comparison": generation_comparison,
        "descriptive_ratios": ratios,
        "forbidden_paths": {
            "compiled_generation": "not-called",
            "recurrence_generation": (
                "authority" if recurrence is not None else "not-called"
            ),
            "amplicol_reference": (
                "authority" if reference is not None else "not-loaded"
            ),
            "external_comparator": (
                authority_kind
                if authority_kind != AUTHORITY_OTF_ONLY
                else "not-created"
            ),
            "physics_enumeration": (
                "recurrence-authority-only" if recurrence is not None else "not-called"
            ),
            "resolved_output": (
                "authority-cells-only"
                if authority_kind != AUTHORITY_OTF_ONLY
                else "not-called"
            ),
            "prepared_model_on_all_flow": (
                "forbidden" if args.workload == "all-flow" else "not-applicable"
            ),
            "materialized_process_lanes": "OTF-generation-poisoned",
        },
    }


def dataclass_payload(case: Case) -> dict[str, object]:
    return {
        "process_table_id": case.process_table_id,
        "process_key": case.process_key,
        "multiplicity": case.multiplicity,
        "expression": case.process,
        "process_id": case.process_id,
        "external_pdgs": list(case.pdgs),
    }


def _write(path: Path, value: object) -> None:
    if path.exists() or path.is_symlink():
        raise StudyError(f"refusing to replace evidence: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except (OSError, TypeError, ValueError) as error:
        temporary.unlink(missing_ok=True)
        raise StudyError(f"cannot write evidence: {path}") from error


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StudyError(f"cannot read evidence: {path}") from error
    if not isinstance(value, dict):
        raise StudyError(f"evidence is not an object: {path}")
    return value


def _worker_command(args: argparse.Namespace, output: Path) -> list[str]:
    _validate_arguments(args)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--output",
        str(output),
        "--process-id",
        str(args.process_id),
        "--multiplicity",
        str(args.multiplicity),
        "--workload",
        args.workload,
        "--target-runtime",
        str(args.target_runtime),
    ]
    if args.reference_repo_root is not None:
        command.extend(
            (
                "--reference-repo-root",
                str(
                    _existing(
                        args.reference_repo_root,
                        directory=True,
                        label="reference repository root",
                    )
                ),
                "--reference-profile",
                args.reference_profile,
            )
        )
    for option, value, directory, label in (
        ("--prepared-model", args.prepared_model, False, "prepared model"),
        ("--candidate-artifact", args.candidate_artifact, True, "candidate artifact"),
        ("--selected-report", args.selected_report, False, "selected report"),
    ):
        if value is not None:
            command.extend(
                (option, str(_existing(value, directory=directory, label=label)))
            )
    return command


def _worker_main(args: argparse.Namespace) -> int:
    try:
        result = _run_worker(args)
    except Exception as error:
        _write(
            args.output / "worker.json",
            {
                "kind": KIND,
                "schema_version": SCHEMA_VERSION,
                "status": "failed",
                "error": str(error),
            },
        )
        return 1
    _write(args.output / "worker.json", result)
    return 0


def _driver_main(args: argparse.Namespace) -> int:
    _validate_arguments(args)
    output = Path(os.path.abspath(args.output.expanduser()))
    if output.exists() or output.is_symlink():
        raise StudyError(f"output already exists: {output}")
    output.mkdir(parents=True)
    watchdog_path = output / "watchdog.json"
    exit_code = run_guarded(
        _worker_command(args, output),
        limit_bytes=WATCHDOG_BYTES,
        report_path=watchdog_path,
        physical_footprint_probe=n4._physical_footprint_probe(),
    )
    try:
        watchdog = n4._watchdog_summary(_read(watchdog_path))
    except n4.GateError as error:
        raise StudyError(str(error)) from error
    worker_path = output / "worker.json"
    worker = _read(worker_path) if worker_path.is_file() else None
    if (
        exit_code
        or watchdog["passes"] is not True
        or not worker
        or worker.get("status") != "passed"
    ):
        detail = None if worker is None else worker.get("error")
        raise StudyError(f"guarded worker failed: {watchdog['outcome']!r}, {detail!r}")
    report = {
        "kind": f"{KIND}-run",
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "study": worker,
        "watchdog": watchdog,
    }
    _write(output / "report.json", report)
    print(f"On-the-fly high-multiplicity study passed: {output / 'report.json'}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return _worker_main(args) if args.worker else _driver_main(args)
    except StudyError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
