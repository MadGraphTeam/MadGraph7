# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import hashlib
import json
import os
import struct
import uuid
import zlib
from pathlib import Path

import pytest

from tools.performance_report.artifacts import (
    ATTEMPT_SCHEMA,
    CURRENT_SCHEMA,
    ArtifactStore,
)
from tools.performance_report.catalog import REPORT_CATALOG
from tools.performance_report.manual_campaign import (
    Palette,
    _index_metadata_dirty_paths,
    _run_campaign,
    build_parser,
    lightweight_current,
)
from tools.performance_report.measurement import failure_measurement
from tools.performance_report.models import (
    ArtifactPolicy,
    CellSpec,
    ExecutionMode,
    ResultStatus,
    Workload,
)
from tools.performance_report.scheduler import CampaignSettings, plan_campaign
from tools.performance_report.service import ReportPaths, ReportService
from tools.performance_report.source_identity import ReportSourceIdentity

ROOT = Path(__file__).resolve().parents[2]


def _cell_identity(cell: CellSpec) -> dict[str, object]:
    return {
        "cell_id": cell.cell_id,
        "dataset_id": cell.dataset_id,
        "process_key": cell.process_key,
        "process": cell.process,
        "n_final": cell.n_final,
        "workload": cell.workload.value,
        "execution_mode": cell.measurement.execution_mode.value,
        "model": (
            None if cell.measurement.model is None else cell.measurement.model.value
        ),
        "accuracy": cell.measurement.accuracy.value,
        "backend": cell.measurement.backend,
        "variant": cell.variant,
    }


def _amplicol_leaf_cell() -> CellSpec:
    return next(
        cell
        for cell in REPORT_CATALOG.measurement_cells()
        if cell.measurement.execution_mode is ExecutionMode.AMPLICOL
        and cell.workload is Workload.SELECTED_FLOW
        and cell.n_final == 1
    )


def _write_lightweight_terminal_current(
    service: ReportService,
    cell: CellSpec,
    *,
    revision: str,
) -> tuple[Path, Path]:
    attempt_id = str(uuid.uuid4())
    attempt_root = service.store._cell_root(cell.cell_id) / "attempts" / attempt_id
    attempt_root.mkdir(parents=True)
    result = failure_measurement(ResultStatus.TIMEOUT, "generation cap")
    result["provenance"] = {
        "report_source_revision": revision,
        "manual_campaign": {"cell_identity": _cell_identity(cell)},
    }
    result_path = attempt_root / "result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    manifest = {
        "schema": ATTEMPT_SCHEMA,
        "cell_id": cell.cell_id,
        "attempt_id": attempt_id,
        "status": "ok",
        "artifact_policy": ArtifactPolicy.REUSE.value,
        "based_on": None,
        "result_path": "result.json",
        "artifacts": [
            {
                "path": "result.json",
                "size": result_path.stat().st_size,
                "sha256": "0" * 64,
            }
        ],
        "error": None,
    }
    manifest_path = attempt_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    pointer = {
        "schema": CURRENT_SCHEMA,
        "cell_id": cell.cell_id,
        "attempt_id": attempt_id,
        "manifest_path": f"attempts/{attempt_id}/manifest.json",
        "manifest_sha256": "1" * 64,
    }
    pointer_path = service.store._cell_root(cell.cell_id) / "current.json"
    pointer_path.write_text(json.dumps(pointer), encoding="utf-8")
    return manifest_path, result_path


def _git_object(git_dir: Path, kind: bytes, body: bytes) -> str:
    payload = kind + b" " + str(len(body)).encode("ascii") + b"\0" + body
    digest = hashlib.sha1(payload, usedforsecurity=False).hexdigest()
    path = git_dir / "objects" / digest[:2] / digest[2:]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(zlib.compress(payload))
    return digest


def _write_index(repo: Path, relative: str, object_id: str) -> None:
    path = repo / relative
    observed = path.stat()
    encoded = os.fsencode(relative)
    fields = struct.pack(
        "!10I20sH",
        int(observed.st_ctime_ns // 1_000_000_000),
        int(observed.st_ctime_ns % 1_000_000_000),
        int(observed.st_mtime_ns // 1_000_000_000),
        int(observed.st_mtime_ns % 1_000_000_000),
        int(observed.st_dev),
        int(observed.st_ino),
        int(observed.st_mode),
        int(observed.st_uid),
        int(observed.st_gid),
        int(observed.st_size),
        bytes.fromhex(object_id),
        len(encoded),
    )
    entry = fields + encoded + b"\0"
    entry += b"\0" * ((8 - len(entry) % 8) % 8)
    body = struct.pack("!4sII", b"DIRC", 2, 1) + entry
    checksum = hashlib.sha1(body, usedforsecurity=False).digest()
    (repo / ".git/index").write_bytes(body + checksum)


def test_index_vs_head_detects_staged_content_with_matching_stat_metadata(
    tmp_path: Path,
) -> None:
    git_dir = tmp_path / ".git"
    (git_dir / "refs/heads").mkdir(parents=True)
    tracked = tmp_path / "tracked.py"
    tracked.write_text("value = 1\n", encoding="utf-8")

    original_blob = _git_object(git_dir, b"blob", tracked.read_bytes())
    tree_body = b"100644 tracked.py\0" + bytes.fromhex(original_blob)
    tree = _git_object(git_dir, b"tree", tree_body)
    commit = _git_object(
        git_dir,
        b"commit",
        f"tree {tree}\n\nfixture\n".encode("ascii"),
    )
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="ascii")
    (git_dir / "refs/heads/main").write_text(f"{commit}\n", encoding="ascii")

    _write_index(tmp_path, "tracked.py", original_blob)
    assert _index_metadata_dirty_paths(tmp_path) == ()

    tracked.write_text("value = 2\n", encoding="utf-8")
    staged_blob = _git_object(git_dir, b"blob", tracked.read_bytes())
    _write_index(tmp_path, "tracked.py", staged_blob)

    assert _index_metadata_dirty_paths(tmp_path) == (
        "<staged index differs from HEAD>",
    )


def test_lightweight_current_rejects_attempt_and_measurement_schema_errors(
    tmp_path: Path,
) -> None:
    revision = "a" * 40
    service = ReportService(
        ReportPaths.from_repo(
            ROOT,
            profile="macbook_M3_manual",
            artifact_root=tmp_path / "artifacts",
            coordination_root=tmp_path / "coordination",
        )
    )
    cell = _amplicol_leaf_cell()
    manifest_path, result_path = _write_lightweight_terminal_current(
        service,
        cell,
        revision=revision,
    )

    current = lightweight_current(
        service.store,
        cell,
        source_revision=revision,
    )
    assert current is not None
    assert current.complete
    assert current.reusable
    assert current.record is not None

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema"] = "unsupported-attempt-schema"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    wrong_attempt_schema = lightweight_current(
        service.store,
        cell,
        source_revision=revision,
    )
    assert wrong_attempt_schema is not None
    assert not wrong_attempt_schema.complete
    assert not wrong_attempt_schema.reusable
    assert wrong_attempt_schema.reason == "incomplete metadata"

    manifest["schema"] = ATTEMPT_SCHEMA
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result.pop("failure")
    result_path.write_text(json.dumps(result), encoding="utf-8")
    wrong_measurement_schema = lightweight_current(
        service.store,
        cell,
        source_revision=revision,
    )
    assert wrong_measurement_schema is not None
    assert not wrong_measurement_schema.complete
    assert not wrong_measurement_schema.reusable
    assert wrong_measurement_schema.reason == "invalid measurement schema"


def test_manual_dry_run_plans_only_from_lightweight_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    revision = "b" * 40
    service = ReportService(
        ReportPaths.from_repo(
            ROOT,
            profile="macbook_M3_manual",
            artifact_root=tmp_path / "artifacts",
            coordination_root=tmp_path / "coordination",
        )
    )
    cell = _amplicol_leaf_cell()
    _write_lightweight_terminal_current(service, cell, revision=revision)

    def forbidden_load_current(
        _store: ArtifactStore,
        _cell_id: str,
        *,
        missing_ok: bool = False,
    ) -> None:
        del missing_ok
        raise AssertionError("manual planning called authenticated artifact loading")

    monkeypatch.setattr(ArtifactStore, "load_current", forbidden_load_current)
    arguments = build_parser().parse_args(
        ("run", "--dry-run", "--no-color", "--cell-id", cell.cell_id)
    )
    result = _run_campaign(
        arguments,
        repo_root=ROOT,
        service=service,
        source=ReportSourceIdentity(revision, "c" * 40, ()),
        cells=(cell,),
        palette=Palette(False),
    )

    assert result == 0
    output = capsys.readouterr().out
    assert "recycled" in output
    assert "1" in output
    assert "planned with dependencies" in output


def test_non_manual_planning_keeps_authenticated_store_lookup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def observed_load_current(
        _store: ArtifactStore,
        cell_id: str,
        *,
        missing_ok: bool = False,
    ) -> None:
        assert missing_ok
        calls.append(cell_id)
        return None

    monkeypatch.setattr(ArtifactStore, "load_current", observed_load_current)
    store = ArtifactStore(
        artifact_root=tmp_path / "artifacts",
        lock_root=tmp_path / "locks",
    )
    cell = _amplicol_leaf_cell()

    planned = plan_campaign((cell,), store=store, settings=CampaignSettings())

    assert planned
    assert cell.cell_id in calls


@pytest.mark.parametrize(
    "setting",
    (
        "target_runtime_seconds",
        "resource_sample_interval_seconds",
        "termination_grace_seconds",
        "timeout_seconds",
    ),
)
@pytest.mark.parametrize("value", (float("nan"), float("inf")))
def test_campaign_settings_reject_non_finite_float_controls(
    setting: str,
    value: float,
) -> None:
    with pytest.raises(ValueError):
        CampaignSettings(**{setting: value})
