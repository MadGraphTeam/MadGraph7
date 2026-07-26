# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import subprocess
from copy import deepcopy
from pathlib import Path

import pytest

from tools.performance_report.catalog import REPORT_CATALOG
from tools.performance_report.measurement import (
    _baseline_matrix_element,
    _baseline_selector_contract,
    _stable_runtime_identity,
    _validate_runtime_identity_postflight,
    failure_measurement,
    source_revision,
)
from tools.performance_report.models import ResultStatus
from tools.performance_report.runner import RunnerError, SelectorContract


def _contract() -> SelectorContract:
    return SelectorContract(
        selected_color_flow_ids=("flow:1,2,3",),
        selected_color_words=((1, 2, 3),),
        all_flow_helicity_ids=("h:-1,+1,-1",),
        all_flow_source_helicities=((1, -1), (2, 1), (3, -1)),
        point_digest="a" * 64,
    )


def test_baseline_contract_and_matrix_element_are_strict() -> None:
    baseline = {
        "status": "ok",
        "matrix_element": 2.0,
        "selector_contract": _contract().as_dict(),
    }
    assert _baseline_selector_contract(baseline) == _contract()
    assert _baseline_matrix_element(baseline) == 2.0

    with pytest.raises(RunnerError, match="not a valid completed"):
        _baseline_matrix_element({"status": "error"})
    with pytest.raises(RunnerError, match="no matrix element"):
        _baseline_matrix_element({"status": "ok", "matrix_element": None})


def test_failure_measurement_preserves_compact_cache_shape() -> None:
    measurement = failure_measurement(
        ResultStatus.MEMORY_LIMIT,
        RuntimeError("over limit"),
        resources={"peak_rss_bytes": 42},
    )

    assert measurement["status"] == "memory_limit"
    assert measurement["generation_seconds"] is None
    assert measurement["resources"] == {"peak_rss_bytes": 42}
    assert measurement["failure"] == {
        "kind": "RuntimeError",
        "message": "over limit",
    }


def test_catalog_contains_no_amplicol_candidate_matrix_cell() -> None:
    assert all(
        cell.measurement.execution_mode.value != "amplicol"
        for cell in REPORT_CATALOG.matrix_cells()
    )


def test_source_revision_rejects_source_dirt_but_allows_report_outputs(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(("git", "init", "-q"), cwd=repo, check=True)
    subprocess.run(
        ("git", "config", "user.email", "report-test@example.invalid"),
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ("git", "config", "user.name", "Report Test"),
        cwd=repo,
        check=True,
    )
    source = repo / "source.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(("git", "add", "source.py"), cwd=repo, check=True)
    subprocess.run(("git", "commit", "-q", "-m", "initial"), cwd=repo, check=True)

    revision = source_revision(repo, require_clean=True)
    assert len(revision) == 40

    cache = repo / "docs/arxiv/results/z_builtin_sm.json"
    cache.parent.mkdir(parents=True)
    cache.write_text("{}\n", encoding="ascii")
    table = repo / "docs/arxiv/result_z_builtin_sm_table.tex"
    table.write_text("% generated\n", encoding="ascii")
    assert source_revision(repo, require_clean=True) == revision

    source.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(RunnerError, match="outside generated report outputs"):
        source_revision(repo, require_clean=True)


def _runtime_identity() -> dict[str, object]:
    observations = [
        {
            "module": "pyamplicol",
            "kind": "package-member",
            "root_index": 0,
            "path": "__init__.py",
            "size": 10,
            "sha256": "1" * 64,
        }
    ]
    return {
        "kind": "pyamplicol-report-runtime-identity-v1",
        "loaded_module_origin_policy": {
            "kind": "pyamplicol-loaded-module-origin-policy-v1",
            "all_loaded_origins_authenticated": True,
            "native_image_origin_bound": True,
            "loaded_bytecode_eligible": False,
            "observed_module_count": len(observations),
            "observations": observations,
            "observations_sha256": "a" * 64,
        },
    }


def test_runtime_identity_postflight_is_stable_and_monotonic() -> None:
    initial = _runtime_identity()
    postflight = deepcopy(initial)
    policy = postflight["loaded_module_origin_policy"]
    assert isinstance(policy, dict)
    observations = policy["observations"]
    assert isinstance(observations, list)
    observations.append(
        {
            "module": "pyamplicol.runtime",
            "kind": "package-member",
            "root_index": 0,
            "path": "runtime/__init__.py",
            "size": 20,
            "sha256": "2" * 64,
        }
    )
    policy["observed_module_count"] = len(observations)
    _validate_runtime_identity_postflight(initial, postflight)
    assert _stable_runtime_identity(initial) == _stable_runtime_identity(postflight)

    changed = deepcopy(postflight)
    changed["native_build_inputs_sha256"] = "3" * 64
    with pytest.raises(RunnerError, match="changed during report measurement"):
        _validate_runtime_identity_postflight(initial, changed)

    lost = deepcopy(postflight)
    lost_policy = lost["loaded_module_origin_policy"]
    assert isinstance(lost_policy, dict)
    lost_observations = lost_policy["observations"]
    assert isinstance(lost_observations, list)
    lost_observations.pop(0)
    lost_policy["observed_module_count"] = len(lost_observations)
    with pytest.raises(RunnerError, match="lost an authenticated"):
        _validate_runtime_identity_postflight(initial, lost)
