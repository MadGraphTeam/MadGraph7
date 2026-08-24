# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import hashlib
from copy import deepcopy
from decimal import Decimal
from pathlib import Path

from pyamplicol.artifacts import ArtifactBuilder, load_manifest
from pyamplicol.generation.structural_source_proof import (
    ROLE as STRUCTURAL_SOURCE_PROOF_ROLE,
)
from pyamplicol.generation.structural_source_proof import (
    SEMANTIC_MAP_DOMAINS,
    build_generation_structural_proof,
)
from tools.developer.catalog_restart_parity_gate import (
    SCHEMA,
    validate_manifest,
)
from tools.developer.final_source_numerical_truth import (
    REFERENCE_SCHEMA,
    _agreement,
    _value_record,
    canonical_sha256,
    comparison_sha256,
)
from tools.developer.final_source_numerical_truth import (
    SCHEMA as TRUTH_SCHEMA,
)
from tools.performance_report.catalog import REPORT_CATALOG
from tools.performance_report.models import ExecutionMode

REVISION = "1" * 40
DIGEST = "2" * 64


def _counts(value: int = 10) -> dict[str, int]:
    return {
        "source_current_count": 2,
        "produced_current_count": value - 2,
        "kernel_evaluation_count": value,
        "attachment_count": value,
        "amplitude_destination_count": 2,
    }


def _truth(
    cell: object,
    *,
    candidate_id: str,
    structural_path: str,
    structural_sha256: str,
) -> dict[str, object]:
    reference_id = hashlib.sha256(f"reference:{cell.cell_id}".encode()).hexdigest()
    contract: dict[str, object] = {
        "schema": REFERENCE_SCHEMA,
        "status": "materialized",
        "semantics": "pre-optimization-reference",
        "source_revision": REVISION,
        "artifact_id": reference_id,
        "process_id": "process",
        "contract_sha256": hashlib.sha256(
            f"contract:{cell.cell_id}".encode()
        ).hexdigest(),
        "semantic_payloads": [
            {
                "path": "reference.json",
                "sha256": hashlib.sha256(
                    f"reference-payload:{cell.cell_id}".encode()
                ).hexdigest(),
            }
        ],
    }
    contract["semantic_identity_sha256"] = canonical_sha256(
        "independent-reference-semantics-v1",
        contract,
    )
    momenta: dict[str, object] = {
        "candidate_payload_sha256": hashlib.sha256(b"candidate-points").hexdigest(),
        "reference_payload_sha256": hashlib.sha256(b"reference-points").hexdigest(),
        "points": [
            [
                {"pdg": 1, "momentum": ["5", "0", "0", "5"]},
                {"pdg": -1, "momentum": ["5", "0", "0", "-5"]},
            ]
        ],
    }
    momenta["validation_momenta_sha256"] = canonical_sha256(
        "final-source-validation-momenta-v1",
        momenta,
    )
    selected = cell.workload.value == "selected-flow"
    selectors: dict[str, object] = {
        "workload": cell.workload.value,
        "helicity_ids": ["h:0"] if selected else [],
        "color_ids": ["c:0"] if selected else [],
    }
    selectors["selector_sha256"] = canonical_sha256(
        "final-source-selectors-v1",
        selectors,
    )
    axes: dict[str, object] = {
        "color_accuracy": cell.measurement.accuracy.value,
        "helicity_ids": ["h:0"],
        "color_ids": ["c:0"],
    }
    axes["resolved_axes_sha256"] = canonical_sha256(
        "final-source-resolved-axes-v1",
        axes,
    )
    exact_values = [[["1"]]]
    native_values = [[[["1", "0"]]]]
    agreement = _agreement(
        exact_values,
        native_values,
        exact_values,
        precision=80,
        exact_agreement_digits=50,
        native_relative_tolerance=Decimal("1e-10"),
        native_absolute_tolerance=Decimal("1e-14"),
    )
    truth: dict[str, object] = {
        "schema": TRUTH_SCHEMA,
        "status": "ok",
        "cell_id": cell.cell_id,
        "source_revision": REVISION,
        "mode": cell.measurement.execution_mode.value,
        "accuracy": cell.measurement.accuracy.value,
        "workload": cell.workload.value,
        "precision_decimal_digits": 80,
        "validation_momenta": momenta,
        "selectors": selectors,
        "resolved_axes": axes,
        "candidate_artifact": {
            "artifact_id": candidate_id,
            "source_revision": REVISION,
            "structural_proof_path": structural_path,
            "structural_proof_sha256": structural_sha256,
        },
        "reference_artifact": {
            "artifact_id": reference_id,
            "source_revision": REVISION,
            "semantics_contract": contract,
        },
        "candidate_native": _value_record(
            precision=16,
            executor="native",
            values=native_values,
            domain="final-source-candidate-native-values-v1",
        ),
        "candidate_exact": _value_record(
            precision=80,
            executor="exact",
            values=exact_values,
            domain="final-source-candidate-exact-values-v1",
        ),
        "reference_exact": _value_record(
            precision=80,
            executor="reference",
            values=exact_values,
            domain="final-source-reference-exact-values-v1",
        ),
        "agreement": agreement,
    }
    truth["comparison_sha256"] = comparison_sha256(truth)
    return truth


def _candidate(
    cell: object,
    evidence_root: Path,
    *,
    artifact_id: str,
    structural_path: str,
    structural_sha256: str,
    generation_proof: dict[str, object],
) -> dict[str, object]:
    maps = generation_proof["semantic_maps"]
    witnesses = {name: maps[name]["rows"] for name in SEMANTIC_MAP_DOMAINS}
    return {
        "source_revision": REVISION,
        "evidence_root": str(evidence_root),
        "artifact_identity": {
            "artifact_id": artifact_id,
            "process_id": "process",
            "structural_proof_path": structural_path,
            "structural_proof_sha256": structural_sha256,
        },
        "active": _counts(),
        "final_materialized": _counts(),
        "peak_materialized": _counts(),
        "semantic_proof": {
            "status": "proven",
            "strength": "exact-symbolic",
            "source_revision": REVISION,
            "current_member_map_sha256": maps["current_member_map"]["sha256"],
            "interaction_row_map_sha256": maps["interaction_row_map"]["sha256"],
            "closure_map_sha256": maps["closure_map"]["sha256"],
            "source_contract_sha256": maps["source_contract"]["sha256"],
            "witnesses": witnesses,
        },
        "numerical_validation": _truth(
            cell,
            candidate_id=artifact_id,
            structural_path=structural_path,
            structural_sha256=structural_sha256,
        ),
        "generation_structural_proof": generation_proof,
        "persisted_lane_inventory": generation_proof["physical_lane_inventory"],
    }


def _legacy(cell: object) -> dict[str, object]:
    if not REPORT_CATALOG.legacy_reference_available(cell):
        return {
            "scope": "unavailable",
            "reason": "original-amplicol-open-quark-line-limit",
        }
    return {
        "scope": "available",
        "active": _counts(),
        "static": _counts(),
        "object_mapping": {
            "status": "exact",
            "current_object_map_sha256": DIGEST,
            "kernel_term_map_sha256": DIGEST,
            "combine_route_map_sha256": DIGEST,
        },
        "row_multiplicity": {
            "status": "exact",
            "histogram_sha256": DIGEST,
            "call_count": 1,
        },
    }


def _artifact_fixture(
    tmp_path: Path,
) -> tuple[Path, str, str, str, dict[str, object]]:
    artifact = tmp_path / "artifact"
    execution_path = "processes/process/execution.json"
    structural_path = "processes/process/structural-source-proof.json"
    execution: dict[str, object] = {
        "schema_version": 3,
        "kind": "pyamplicol-runtime-recurrence-execution",
        "key": "process",
        "process": "d d~ > z",
        "color_accuracy": "lc",
        "external_pdg_order": [1, -1, 23],
        "source_count": 2,
        "current_count": 10,
        "interaction_count": 10,
        "amplitude_root_count": 2,
    }
    producer = {
        "distribution": "pyamplicol",
        "version": "0.1.0",
        "versions": {
            "python_api": 1,
            "toml": 1,
            "compiled_model": 1,
            "process_artifact": 3,
            "runtime_physics": 1,
            "symbolica_serialization": "test",
            "c_abi": 1,
        },
        "target": {"triple": "test", "cpu_features": []},
        "git_revision": REVISION,
        "native_build_inputs_sha256": DIGEST,
    }
    with ArtifactBuilder(artifact) as builder:
        execution_record = builder.add_json(
            execution_path,
            execution,
            role="evaluator-manifest",
            process_id="process",
            compact=True,
        )
        generation_proof = build_generation_structural_proof(
            artifact_root=builder.root,
            process_id="process",
            source_revision=REVISION,
            native_build_inputs_sha256=DIGEST,
            execution_path=execution_path,
            execution_sha256=execution_record.sha256,
            execution=execution,
            evaluator_container_path=None,
            evaluator_container_index_sha256=None,
        )
        structural_record = builder.add_json(
            structural_path,
            generation_proof,
            role=STRUCTURAL_SOURCE_PROOF_ROLE,
            process_id="process",
            compact=True,
        )
        builder.finalize(
            kind="pyamplicol-process",
            producer=producer,
            model={
                "name": "built-in-sm",
                "source_kind": "built-in-sm",
                "content_sha256": DIGEST,
                "compiled_schema_version": 1,
            },
            configuration={
                "toml_schema_version": 1,
                "requested_path": "processes/process/execution.json",
                "effective_path": "processes/process/execution.json",
                "adjustments": [],
            },
            processes=[
                {
                    "id": "process",
                    "expression": "d d~ > z",
                    "color_accuracy": "lc",
                    "external_pdgs": [1, -1, 23],
                    "physics_path": execution_path,
                    "required_runtime_capabilities": [
                        "rusticol.recurrence-direct-arena.complex-f64.v1"
                    ],
                    "aliases": [],
                }
            ],
            default_process_id="process",
            runtime={
                "engine": "rusticol",
                "engine_version": "0.1.0",
                "evaluator_manifest_path": execution_path,
                "api_bundle_path": None,
                "required_runtime_capabilities": [
                    "rusticol.recurrence-direct-arena.complex-f64.v1"
                ],
            },
        )
    manifest = load_manifest(artifact)
    return (
        artifact,
        manifest.artifact_id,
        structural_path,
        structural_record.sha256,
        generation_proof,
    )


def _manifest(tmp_path: Path) -> dict[str, object]:
    (
        artifact,
        artifact_id,
        structural_path,
        structural_sha,
        generation_proof,
    ) = _artifact_fixture(tmp_path)
    cells = []
    for cell in REPORT_CATALOG.matrix_cells():
        if cell.measurement.execution_mode not in {
            ExecutionMode.RECURRENCE,
            ExecutionMode.COMPILED,
            ExecutionMode.EAGER,
        }:
            continue
        comparable = REPORT_CATALOG.legacy_reference_available(cell)
        cells.append(
            {
                "cell_id": cell.cell_id,
                "process_key": cell.process_key,
                "n_final": cell.n_final,
                "mode": cell.measurement.execution_mode.value,
                "model": cell.measurement.model.value,
                "accuracy": cell.measurement.accuracy.value,
                "workload": cell.workload.value,
                "classification": (
                    "certified-parity" if comparable else "legacy-scope-unavailable"
                ),
                "candidate": _candidate(
                    cell,
                    artifact,
                    artifact_id=artifact_id,
                    structural_path=structural_path,
                    structural_sha256=structural_sha,
                    generation_proof=generation_proof,
                ),
                "legacy": _legacy(cell),
            }
        )
    return {"schema": SCHEMA, "source_revision": REVISION, "cells": cells}


def test_complete_exact_manifest_passes_every_catalog_row(tmp_path: Path) -> None:
    result = validate_manifest(_manifest(tmp_path), expected_revision=REVISION)
    assert result["summary"]["expected_cell_count"] == 1356
    assert result["summary"]["failed_row_count"] == 0
    assert result["summary"]["classification_counts"] == {
        "certified-parity": 1314,
        "legacy-scope-unavailable": 42,
    }
    assert result["summary"]["restart_ready"] is True
    assert len(result["rows"]) == 1356


def test_summary_boolean_cannot_replace_per_row_evidence() -> None:
    result = validate_manifest(
        {
            "schema": SCHEMA,
            "source_revision": REVISION,
            "fully_certified_catalog_parity": True,
            "cells": [],
        },
        expected_revision=REVISION,
    )
    assert result["summary"]["restart_ready"] is False
    assert result["summary"]["failure_reason_counts"] == {"missing-record": 1356}


def test_missing_and_unexpected_rows_are_listed_with_reasons(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    cells = manifest["cells"]
    assert isinstance(cells, list)
    cells.pop()
    cells.append(
        {
            "cell_id": "not-in-the-report-catalog",
            "classification": "certified-parity",
        }
    )
    result = validate_manifest(manifest, expected_revision=REVISION)
    assert result["summary"]["failure_reason_counts"] == {
        "missing-record": 1,
        "unexpected-record": 1,
    }


def test_numerical_only_generation_relation_fails_closed(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    cells = manifest["cells"]
    assert isinstance(cells, list)
    candidate = cells[0]["candidate"]
    candidate["numerical_relation_discovery"] = {
        "used_for_generation": True,
        "reconstructed_exact": False,
    }
    result = validate_manifest(manifest, expected_revision=REVISION)
    assert result["summary"]["failure_reason_counts"] == {
        "numerical-only-relation-used-for-generation": 1
    }


def test_counts_only_certificate_and_stale_source_are_rejected(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    cells = manifest["cells"]
    assert isinstance(cells, list)
    candidate = cells[0]["candidate"]
    candidate["source_revision"] = "3" * 40
    candidate["semantic_proof"] = {"status": "proven"}
    result = validate_manifest(manifest, expected_revision=REVISION)
    assert result["summary"]["failure_reason_counts"] == {
        "candidate-source-revision-mismatch": 1,
        "semantic-proof-incomplete": 1,
        "semantic-proof-not-exact": 1,
        "semantic-proof-source-revision-mismatch": 1,
    }


def test_static_inventory_and_ratio_are_recomputed_not_trusted(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    cells = manifest["cells"]
    assert isinstance(cells, list)
    record = cells[0]
    record["producer_ratio"] = 0.01
    candidate = record["candidate"]
    candidate["final_materialized"] = _counts(20)
    result = validate_manifest(manifest, expected_revision=REVISION)
    reasons = result["rows"][0]["reason_codes"]
    assert "final_materialized-logical_current-exceeds-1.05" in reasons
    assert "final_materialized-kernel_evaluation-exceeds-1.05" in reasons
    assert "final_materialized-attachment-exceeds-1.05" in reasons


def test_legacy_scope_unavailable_still_requires_candidate_self_proof(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    cells = manifest["cells"]
    assert isinstance(cells, list)
    target = next(cell for cell in cells if cell["process_key"] == "dd_4q_lines")
    target["candidate"] = deepcopy(target["candidate"])
    del target["candidate"]["numerical_validation"]
    result = validate_manifest(manifest, expected_revision=REVISION)
    row = next(row for row in result["rows"] if row["cell_id"] == target["cell_id"])
    assert row["reason_codes"] == ["numerical-validation-incomplete"]
