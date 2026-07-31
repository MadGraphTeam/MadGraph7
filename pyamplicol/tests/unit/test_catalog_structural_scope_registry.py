# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json

from tools.developer.catalog_structural_scope_registry import (
    SCHEMA,
    validate_reviewed_matrix_scope,
)


def test_reviewed_out_of_catalog_scope_is_exact_and_machine_readable() -> None:
    scope = validate_reviewed_matrix_scope()
    assert scope["schema"] == SCHEMA
    assert scope["review"]["status"] == "reviewed"
    assert {
        item["scope_id"] for item in scope["intentionally_out_of_catalog"]
    } == {"ufo-compiled-and-eager"}
    requirements = {
        item["scope_id"]: item for item in scope["candidate_only_requirements"]
    }
    assert requirements[
        "four-open-quark-lines-lc-candidate-proof"
    ]["catalog_cell_count"] == 32
    contracted = requirements[
        "four-open-quark-lines-contracted-candidate-proof"
    ]
    assert contracted["catalog_cell_count"] == 8
    assert contracted["mode_model_workload_plane_count"] == 4
    assert contracted["legacy_comparison"] == {
        "status": "unavailable",
        "reason": "original-amplicol-open-quark-line-limit",
    }
    json.dumps(scope, sort_keys=True)
