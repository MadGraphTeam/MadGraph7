# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from tools.performance_report.campaign_policy import (
    MACBOOK_M3_POLICY,
    MACBOOK_M3_Z_TABLE_F_POLICY,
    PolicyCensorKind,
    policy_censor_measurement,
)
from tools.performance_report.models import ArtifactPolicy
from tools.performance_report.service import ReportPaths, ReportService
from tools.performance_report.source_identity import ReportSourceIdentity
from tools.performance_report.study_contract import (
    StudyContractError,
    audit_z_table_f_policy_projection,
    bind_z_table_f_attempt,
    load_z_table_f_study_contract,
    require_z_table_f_explicit_cell,
    write_z_table_f_study_contract,
    z_table_f_cell_ids,
)


def _git_repo(path: Path, marker: str) -> Path:
    path.mkdir()
    subprocess.run(("git", "init", "-q"), cwd=path, check=True)
    subprocess.run(
        ("git", "config", "user.email", "test@example.invalid"),
        cwd=path,
        check=True,
    )
    subprocess.run(
        ("git", "config", "user.name", "Test"),
        cwd=path,
        check=True,
    )
    (path / "README.md").write_text(marker + "\n", encoding="ascii")
    subprocess.run(("git", "add", "README.md"), cwd=path, check=True)
    subprocess.run(
        ("git", "commit", "-qm", "fixture"),
        cwd=path,
        check=True,
    )
    return path


def test_study_contract_binds_source_wrapper_policy_and_exact_limits(
    tmp_path: Path,
) -> None:
    source = _git_repo(tmp_path / "source", "measured")
    wrapper = _git_repo(tmp_path / "wrapper", "policy")
    contract_path = tmp_path / "contract.json"

    created = write_z_table_f_study_contract(
        contract_path,
        source,
        wrapper,
    )
    loaded = load_z_table_f_study_contract(
        contract_path,
        source,
        wrapper,
    )

    assert loaded == created
    assert loaded["campaign_policy"] == (
        MACBOOK_M3_Z_TABLE_F_POLICY.as_manifest()
    )
    assert loaded["campaign_policy"] != MACBOOK_M3_POLICY.as_manifest()
    assert loaded["policy_profile"] == "macbook_M3"
    assert set(loaded["policy_wrapper"]) == {"revision", "tree"}
    assert loaded["memory_guard"] == {
        "metric_abi": "pyamplicol-process-tree-memory-metric-v1",
        "metric": (
            "max(aggregate-process-tree-rss,"
            "darwin-physical-footprint)"
        ),
        "limit_bytes": 30_000_000_000,
    }
    assert loaded["generation_guard"] == {
        "evidence_abi": (
            "pyamplicol-report-generation-phase-evidence-v1"
        ),
        "limit_seconds": 3600.0,
        "scope": (
            "every legacy, selected-flow, and all-flow generation"
        ),
    }
    assert loaded["allowed_cell_ids"] == list(z_table_f_cell_ids())
    assert len(loaded["allowed_cell_ids"]) == 28
    assert loaded["retained_prior_evidence"] == {
        "n_final": list(range(1, 8)),
        "treatment": (
            "outside this contract; retain each original attempt, "
            "provenance record, and resource ABI unchanged"
        ),
        "memory_guard_interpretation": (
            "legacy RSS-only evidence remains legacy and is not "
            "relabeled as Darwin physical-footprint or exact-decimal-"
            "30GB evidence"
        ),
    }


def test_study_contract_replays_after_wrapper_path_relocation(
    tmp_path: Path,
) -> None:
    source = _git_repo(tmp_path / "source", "measured")
    wrapper = _git_repo(tmp_path / "wrapper", "policy")
    contract_path = tmp_path / "contract.json"
    created = write_z_table_f_study_contract(
        contract_path,
        source,
        wrapper,
    )
    relocated = tmp_path / "relocated-wrapper"
    shutil.move(wrapper, relocated)

    loaded = load_z_table_f_study_contract(
        contract_path,
        source,
        relocated,
    )

    assert loaded == created


def test_study_contract_rejects_tamper_wrapper_drift_and_broad_cell(
    tmp_path: Path,
) -> None:
    source = _git_repo(tmp_path / "source", "measured")
    wrapper = _git_repo(tmp_path / "wrapper", "policy")
    contract_path = tmp_path / "contract.json"
    contract = write_z_table_f_study_contract(
        contract_path,
        source,
        wrapper,
    )
    allowed = contract["allowed_cell_ids"]
    assert isinstance(allowed, list)
    require_z_table_f_explicit_cell(contract, str(allowed[0]))
    with pytest.raises(StudyContractError, match="outside"):
        require_z_table_f_explicit_cell(
            contract,
            "matrix-compiled-builtin-sm-lc-n1-dd-z-jets-contracted",
        )

    raw = json.loads(contract_path.read_text(encoding="ascii"))
    raw["memory_guard"]["limit_bytes"] = 30 * 1024**3
    contract_path.write_text(
        json.dumps(raw, sort_keys=True),
        encoding="ascii",
    )
    with pytest.raises(StudyContractError, match="digest"):
        load_z_table_f_study_contract(contract_path, source, wrapper)

    contract_path.unlink()
    write_z_table_f_study_contract(contract_path, source, wrapper)
    (wrapper / "README.md").write_text("changed\n", encoding="ascii")
    with pytest.raises(StudyContractError, match="clean"):
        load_z_table_f_study_contract(contract_path, source, wrapper)


def test_study_audit_replays_f_policy_without_ordinary_mac_policy(
    tmp_path: Path,
) -> None:
    source = _git_repo(tmp_path / "source", "measured")
    wrapper = _git_repo(tmp_path / "wrapper", "policy")
    contract_path = tmp_path / "contract.json"
    contract = write_z_table_f_study_contract(
        contract_path,
        source,
        wrapper,
    )
    source_record = contract["measured_source"]
    assert isinstance(source_record, dict)
    identity = ReportSourceIdentity(
        str(source_record["revision"]),
        str(source_record["tree"]),
        (),
    )
    docs = tmp_path / "docs"
    docs.mkdir()
    service = ReportService(
        ReportPaths.from_repo(
            source,
            docs_dir=docs,
            artifact_root=tmp_path / "artifacts",
            coordination_root=tmp_path / "coordination",
        )
    )
    service.publish(reset=True, merge_artifacts=False)
    allowed = frozenset(str(value) for value in contract["allowed_cell_ids"])
    cells = tuple(
        cell
        for cell in service.catalog.measurement_cells()
        if cell.cell_id in allowed and cell.n_final == 8
    )
    assert len(cells) == 14
    measured_cells = tuple(
        cell
        for cell in cells
        if service.catalog.static_na_reason(cell) is None
    )
    assert len(measured_cells) == 10
    phase = {
        "abi": "pyamplicol-report-generation-phase-evidence-v1",
        "phase_state_abi": "pyamplicol-report-worker-phase-state-v1",
        "configured_timeout_seconds": 3600.0,
        "supervisor_reason": "generation_timeout",
        "authenticated": True,
        "run_id": "study-run",
        "worker_pid": 123,
        "final_sequence": 1,
        "final_phase": "generation",
        "generation_started_monotonic_ns": 1,
        "generation_finished_monotonic_ns": None,
        "generation_elapsed_seconds": 3600.0,
        "final_state_sha256": "4" * 64,
        "error": None,
    }
    resources = {
        "available": True,
        "current_rss_bytes": 100,
        "peak_rss_bytes": 100,
        "child_count": 0,
        "cpu_seconds": 1.0,
        "wall_seconds": 3600.0,
        "probe_error": None,
        "memory_metric_abi": (
            "pyamplicol-process-tree-memory-metric-v1"
        ),
        "current_physical_footprint_bytes": 120,
        "peak_physical_footprint_bytes": 120,
        "current_guard_bytes": 120,
        "peak_guard_bytes": 120,
        "memory_limit_bytes": 30_000_000_000,
        "memory_limit_reason": None,
        "memory_probe_reason": None,
        "generation_phase": phase,
    }
    for cell in measured_cells:
        result = policy_censor_measurement(
            MACBOOK_M3_Z_TABLE_F_POLICY,
            "macbook_M3",
            cell,
            kind=PolicyCensorKind.GENERATION_LIMIT,
            source_identity=identity,
            resources=resources,
            observed_generation_seconds=3600.0,
            phase_evidence=phase,
        )
        bind_z_table_f_attempt(result, str(contract["sha256"]))
        service.store.new_attempt(
            cell.cell_id,
            ArtifactPolicy.REGENERATE,
        ).publish(result)
    service.publish()

    audited = audit_z_table_f_policy_projection(
        contract,
        service,
        maximum_n=8,
    )

    assert audited["status"] == "ok"
    assert audited["declared_cell_count"] == 14
    assert audited["static_na_cell_count"] == 4
    assert audited["policy_state_counts"] == {"generation_limit": 10}
    assert audited["campaign_policy"] == (
        MACBOOK_M3_Z_TABLE_F_POLICY.as_manifest()
    )
    assert audited["campaign_policy"] != MACBOOK_M3_POLICY.as_manifest()

    current = service.store.load_current(measured_cells[0].cell_id)
    assert current is not None
    transplanted = json.loads(
        json.dumps(current.result, allow_nan=False, sort_keys=True)
    )
    transplanted["provenance"]["study_contract"][
        "study_contract_sha256"
    ] = "f" * 64
    with service.store.new_attempt(
        measured_cells[0].cell_id,
        ArtifactPolicy.REGENERATE,
        based_on=current,
    ) as attempt:
        attempt.publish(transplanted)
    service.publish()
    with pytest.raises(StudyContractError, match="not bound"):
        audit_z_table_f_policy_projection(
            contract,
            service,
            maximum_n=8,
        )
