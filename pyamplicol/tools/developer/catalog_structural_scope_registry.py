#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Reviewed, fail-closed scope for matrix structural-parity coverage gaps.

The 1,152-row restart gate covers every candidate cell that the canonical
matrix catalog defines.  This registry makes the two intentionally absent
classes explicit so that they cannot be mistaken for silently unproved rows.
Any catalog change in these classes requires a review and registry update.
"""

from __future__ import annotations

from typing import Any

from tools.performance_report.catalog import PROCESS_FAMILIES, REPORT_CATALOG
from tools.performance_report.models import (
    Accuracy,
    ExecutionMode,
    ModelKey,
    Workload,
)

SCHEMA = "pyamplicol-reviewed-matrix-structural-scope-v1"
REVIEW_ID = "matrix-structural-scope-2026-07-29"

_PROCESS_KEYS = (
    "dd_z_jets",
    "ud_w_jets",
    "dd_epem_jets",
    "ud_epve_jets",
    "dd_zz_jets",
    "gg_tt_jets",
    "dd_tt_jets",
    "gg_gluons",
    "dd_zzz_jets",
    "dd_epemzh_jets",
    "dd_ttzh_jets",
    "dd_4l_jets",
    "dd_3q_lines",
    "dd_4q_lines",
)
_CANDIDATE_MODES = {
    ExecutionMode.RECURRENCE,
    ExecutionMode.COMPILED,
    ExecutionMode.EAGER,
}
_FOUR_QUARK_LC_PLANES = {
    (ExecutionMode.RECURRENCE, ModelKey.BUILTIN_SM, Workload.SELECTED_FLOW),
    (ExecutionMode.RECURRENCE, ModelKey.BUILTIN_SM, Workload.ALL_FLOW),
    (ExecutionMode.RECURRENCE, ModelKey.UFO_SM, Workload.SELECTED_FLOW),
    (ExecutionMode.RECURRENCE, ModelKey.UFO_SM, Workload.ALL_FLOW),
    (ExecutionMode.COMPILED, ModelKey.BUILTIN_SM, Workload.SELECTED_FLOW),
    (ExecutionMode.COMPILED, ModelKey.BUILTIN_SM, Workload.ALL_FLOW),
    (ExecutionMode.EAGER, ModelKey.BUILTIN_SM, Workload.SELECTED_FLOW),
    (ExecutionMode.EAGER, ModelKey.BUILTIN_SM, Workload.ALL_FLOW),
}
_FOUR_QUARK_CONTRACTED_PLANES = {
    (ExecutionMode.RECURRENCE, ModelKey.BUILTIN_SM, Workload.CONTRACTED),
    (ExecutionMode.RECURRENCE, ModelKey.UFO_SM, Workload.CONTRACTED),
    (ExecutionMode.COMPILED, ModelKey.BUILTIN_SM, Workload.CONTRACTED),
    (ExecutionMode.EAGER, ModelKey.BUILTIN_SM, Workload.CONTRACTED),
}

REVIEWED_MATRIX_SCOPE: dict[str, Any] = {
    "schema": SCHEMA,
    "review": {
        "status": "reviewed",
        "review_id": REVIEW_ID,
    },
    "intentionally_out_of_catalog": [
        {
            "scope_id": "ufo-compiled-and-eager",
            "status": "reviewed-unavailable",
            "process_keys": list(_PROCESS_KEYS),
            "mode_model_pairs": [
                {"mode": "compiled", "model": "ufo_sm"},
                {"mode": "eager", "model": "ufo_sm"},
            ],
            "accuracies": ["lc", "nlc", "full"],
            "workload_by_accuracy": {
                "lc": ["selected-flow", "all-flow"],
                "nlc": ["contracted"],
                "full": ["contracted"],
            },
            "reason": (
                "canonical-matrix-defines-compiled-and-eager-for-built-in-sm-only"
            ),
            "catalog_cell_count": 0,
        },
    ],
    "candidate_only_requirements": [
        {
            "scope_id": "four-open-quark-lines-lc-candidate-proof",
            "process_key": "dd_4q_lines",
            "accuracy": "lc",
            "n_final": [6, 7, 8],
            "mode_model_workload_plane_count": 8,
            "catalog_cell_count": 24,
            "legacy_comparison": {
                "status": "unavailable",
                "reason": "original-amplicol-open-quark-line-limit",
            },
            "proof_requirement": (
                "candidate exact structural and precision-50 self-proof remains "
                "mandatory in catalog_restart_parity_gate"
            ),
        },
        {
            "scope_id": "four-open-quark-lines-contracted-candidate-proof",
            "process_key": "dd_4q_lines",
            "accuracies": ["nlc", "full"],
            "n_final": [6],
            "mode_model_workload_plane_count": 4,
            "catalog_cell_count": 8,
            "legacy_comparison": {
                "status": "unavailable",
                "reason": "original-amplicol-open-quark-line-limit",
            },
            "proof_requirement": (
                "candidate exact structural and precision-50 self-proof remains "
                "mandatory in catalog_restart_parity_gate"
            ),
        },
    ],
}


class CatalogStructuralScopeError(RuntimeError):
    """The canonical matrix drifted from its reviewed coverage scope."""


def validate_reviewed_matrix_scope() -> dict[str, Any]:
    """Validate and return the reviewed out-of-catalog scope declaration."""

    process_keys = tuple(family.key for family in PROCESS_FAMILIES)
    if process_keys != _PROCESS_KEYS:
        raise CatalogStructuralScopeError(
            "process catalog changed; reviewed matrix structural scope is stale"
        )
    candidates = tuple(
        cell
        for cell in REPORT_CATALOG.matrix_cells()
        if cell.measurement.execution_mode in _CANDIDATE_MODES
    )

    ufo_compiled_or_eager = [
        cell
        for cell in candidates
        if cell.measurement.model is ModelKey.UFO_SM
        and cell.measurement.execution_mode
        in {ExecutionMode.COMPILED, ExecutionMode.EAGER}
    ]
    if ufo_compiled_or_eager:
        raise CatalogStructuralScopeError(
            "UFO compiled/eager cells entered the matrix; "
            "reviewed scope must be updated"
        )

    four_quark = [
        cell for cell in candidates if cell.process_key == "dd_4q_lines"
    ]
    non_lc = [
        cell
        for cell in four_quark
        if cell.measurement.accuracy in {Accuracy.NLC, Accuracy.FULL}
    ]
    actual_non_lc = {
        (
            cell.measurement.accuracy,
            cell.n_final,
            cell.measurement.execution_mode,
            cell.measurement.model,
            cell.workload,
        )
        for cell in non_lc
    }
    expected_non_lc = {
        (accuracy, 6, mode, model, workload)
        for accuracy in {Accuracy.NLC, Accuracy.FULL}
        for mode, model, workload in _FOUR_QUARK_CONTRACTED_PLANES
    }
    if actual_non_lc != expected_non_lc or len(non_lc) != 8:
        raise CatalogStructuralScopeError(
            "four-quark contracted candidate-only structural proof coverage "
            "is incomplete"
        )
    lc = [
        cell for cell in four_quark if cell.measurement.accuracy is Accuracy.LC
    ]
    actual_lc = {
        (
            cell.n_final,
            cell.measurement.execution_mode,
            cell.measurement.model,
            cell.workload,
        )
        for cell in lc
    }
    expected_lc = {
        (n_final, mode, model, workload)
        for n_final in (6, 7, 8)
        for mode, model, workload in _FOUR_QUARK_LC_PLANES
    }
    if actual_lc != expected_lc or len(lc) != 24:
        raise CatalogStructuralScopeError(
            "four-quark LC candidate-only structural proof coverage is incomplete"
        )
    if any(REPORT_CATALOG.legacy_reference_available(cell) for cell in four_quark):
        raise CatalogStructuralScopeError(
            "four-quark candidate-only cells unexpectedly have a legacy reference"
        )
    unavailable = [
        cell
        for cell in candidates
        if not REPORT_CATALOG.legacy_reference_available(cell)
    ]
    if {cell.cell_id for cell in unavailable} != {
        cell.cell_id for cell in four_quark
    }:
        raise CatalogStructuralScopeError(
            "an unreviewed legacy-unavailable matrix class is present"
        )
    return REVIEWED_MATRIX_SCOPE


__all__ = [
    "REVIEWED_MATRIX_SCOPE",
    "REVIEW_ID",
    "SCHEMA",
    "CatalogStructuralScopeError",
    "validate_reviewed_matrix_scope",
]
