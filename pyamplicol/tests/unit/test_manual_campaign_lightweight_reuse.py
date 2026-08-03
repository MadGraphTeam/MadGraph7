# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import builtins
import hashlib
import json
import os
import struct
import subprocess
import uuid
import zlib
from collections.abc import Mapping
from pathlib import Path

import pytest

from tools.performance_report import manual_campaign
from tools.performance_report.agreements import DIRECT_AGREEMENT_FIELD
from tools.performance_report.artifacts import (
    ATTEMPT_SCHEMA,
    CURRENT_SCHEMA,
    ArtifactStore,
    ManifestValidationError,
)
from tools.performance_report.cache import empty_measurement
from tools.performance_report.catalog import REPORT_CATALOG
from tools.performance_report.manual_campaign import (
    Palette,
    _index_metadata_dirty_paths,
    _run_campaign,
    build_parser,
    lightweight_current,
)
from tools.performance_report.measurement import failure_measurement, load_measurement
from tools.performance_report.models import (
    Accuracy,
    ArtifactPolicy,
    CellSpec,
    ExecutionMode,
    ResultStatus,
    Workload,
)
from tools.performance_report.scheduler import CampaignSettings, plan_campaign
from tools.performance_report.service import ReportPaths, ReportService
from tools.performance_report.source_identity import ReportSourceIdentity
from tools.performance_report.worker import _portable_current_paths

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


def _amplicol_contracted_cell() -> CellSpec:
    return next(
        cell
        for cell in REPORT_CATALOG.measurement_cells()
        if cell.measurement.execution_mode is ExecutionMode.AMPLICOL
        and cell.measurement.accuracy is Accuracy.NLC
        and cell.workload is Workload.CONTRACTED
        and cell.n_final == 1
    )


def _pyamplicol_leaf_cell() -> CellSpec:
    return next(
        cell
        for cell in REPORT_CATALOG.measurement_cells()
        if cell.measurement.execution_mode is ExecutionMode.RECURRENCE
        and cell.n_final == 1
    )


def _write_lightweight_terminal_current(
    service: ReportService,
    cell: CellSpec,
    *,
    revision: str,
    extra_provenance: Mapping[str, object] | None = None,
) -> tuple[Path, Path]:
    attempt_id = str(uuid.uuid4())
    attempt_root = service.store._cell_root(cell.cell_id) / "attempts" / attempt_id
    attempt_root.mkdir(parents=True)
    result = failure_measurement(ResultStatus.TIMEOUT, "generation cap")
    result["provenance"] = {
        "report_source_revision": revision,
        "manual_campaign": {"cell_identity": _cell_identity(cell)},
        **({} if extra_provenance is None else dict(extra_provenance)),
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


def _write_lightweight_success_current(
    service: ReportService,
    cell: CellSpec,
    *,
    revision: str,
    numerical_relation_correctness: Mapping[str, object],
) -> tuple[Path, Path]:
    attempt_id = str(uuid.uuid4())
    attempt_root = service.store._cell_root(cell.cell_id) / "attempts" / attempt_id
    artifact_root = attempt_root / "artifact"
    artifact_root.mkdir(parents=True)
    (artifact_root / "artifact.json").write_text("{}\n", encoding="ascii")
    result = empty_measurement()
    result.update(
        {
            "status": ResultStatus.OK.value,
            "generation_seconds": 1.0,
            "wall_seconds_per_point": 1.0e-6,
            "execution_seconds_per_point": 8.0e-7,
            "matrix_element": 1.0,
            "sample_count": 5,
            "standard_error_seconds_per_point": 0.0,
            "relative_standard_error": 0.0,
            "artifact": {
                "path": str(artifact_root),
                "process_id": cell.process,
            },
            "validation": {
                "status": ResultStatus.OK.value,
                DIRECT_AGREEMENT_FIELD: [],
            },
            "resources": {},
            "provenance": {
                "report_source_revision": revision,
                "manual_campaign": {"cell_identity": _cell_identity(cell)},
                "numerical_relation_correctness": dict(numerical_relation_correctness),
            },
        }
    )
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


def _portable_manual_service(campaign: Path) -> ReportService:
    campaign = campaign.expanduser().resolve(strict=False)
    state = campaign / "campaign_artifacts"
    paths = ReportPaths(
        repo_root=ROOT,
        docs_dir=campaign,
        results_dir=campaign / "results",
        artifact_root=state,
        coordination_root=state / "coordination",
    )
    return ReportService(paths, portable_current_results=True)


def _publish_success_current(
    service: ReportService,
    cell: CellSpec,
    *,
    revision: str,
) -> None:
    attempt = service.store.new_attempt(cell.cell_id, ArtifactPolicy.REGENERATE)
    artifact = attempt.path("artifact")
    artifact.mkdir()
    (artifact / "artifact.json").write_text("{}\n", encoding="ascii")
    result = empty_measurement()
    result.update(
        {
            "status": ResultStatus.OK.value,
            "generation_seconds": 1.0,
            "wall_seconds_per_point": 1.0e-6,
            "execution_seconds_per_point": 8.0e-7,
            "matrix_element": 1.0,
            "sample_count": 5,
            "standard_error_seconds_per_point": 0.0,
            "relative_standard_error": 0.0,
            "artifact": {"path": str(artifact), "process_id": cell.process},
            "validation": {
                "status": ResultStatus.OK.value,
                DIRECT_AGREEMENT_FIELD: [],
            },
            "resources": {},
            "provenance": {
                "report_source_revision": revision,
                "manual_campaign": {
                    "cell_identity": _cell_identity(cell),
                    "public_cli_reproduction": manual_campaign.reproduction_recipe(
                        cell,
                        repo_root=service.paths.repo_root,
                        artifact_root=service.paths.artifact_root,
                    ).as_dict(),
                },
                "numerical_relation_correctness": {},
            },
        }
    )
    attempt.publish(result, artifact_paths=("artifact/artifact.json",))


def test_manual_current_reuse_survives_whole_campaign_move(
    tmp_path: Path,
) -> None:
    revision = "b" * 40
    recurrence = next(
        cell
        for cell in REPORT_CATALOG.measurement_cells()
        if cell.cell_id
        == "matrix-recurrence-builtin-sm-full-n1-dd-z-jets-contracted"
    )
    compiled = next(
        cell
        for cell in REPORT_CATALOG.measurement_cells()
        if cell.cell_id
        == "matrix-compiled-builtin-sm-full-n1-dd-z-jets-contracted"
    )
    campaign_a = tmp_path / "parent-a" / "campaign-a"
    service_a = _portable_manual_service(campaign_a)
    _publish_success_current(service_a, recurrence, revision=revision)

    campaign_b = tmp_path / "other-parent" / "renamed-campaign"
    campaign_b.parent.mkdir()
    campaign_a.rename(campaign_b)
    service_b = _portable_manual_service(campaign_b)
    state_b = campaign_b / "campaign_artifacts"

    full = service_b.store.load_current(recurrence.cell_id)
    lightweight = lightweight_current(
        service_b.store,
        recurrence,
        source_revision=revision,
    )
    assert full is not None
    assert lightweight is not None and lightweight.reusable
    assert Path(str(full.result["artifact"]["path"])).is_relative_to(state_b)
    assert Path(str(lightweight.result["artifact"]["path"])).is_relative_to(state_b)
    portable_recipe = lightweight.result["provenance"]["manual_campaign"][
        "public_cli_reproduction"
    ]
    assert isinstance(portable_recipe["generate"], list)
    assert any(str(state_b) in argument for argument in portable_recipe["generate"])
    assert str(campaign_a) not in json.dumps(portable_recipe, sort_keys=True)

    resolver = manual_campaign._lightweight_current_resolver(
        service_b,
        source_revision=revision,
        initial={recurrence.cell_id: lightweight},
    )
    resolved = resolver(recurrence)
    assert resolved is not None
    assert Path(str(resolved[0].result["artifact"]["path"])).is_relative_to(state_b)
    planned = plan_campaign(
        (compiled,),
        store=service_b.store,
        settings=CampaignSettings(),
        expected_revision=revision,
        current_resolver=resolver,
    )
    assert [item.cell.cell_id for item in planned] == [compiled.cell_id]
    assert planned[0].baseline_cell_id == recurrence.cell_id

    worker_attempt = service_b.store.new_attempt(
        compiled.cell_id,
        ArtifactPolicy.REGENERATE,
    )
    worker_paths = _portable_current_paths(
        repo_root=ROOT,
        attempt_root=worker_attempt.root,
    )
    assert worker_paths is not None
    worker_loaded = load_measurement(
        full.result_path,
        publication_paths=worker_paths,
    )
    assert Path(str(worker_loaded["artifact"]["path"])).is_relative_to(state_b)
    worker_attempt.discard()

    inspect_payload = manual_campaign._inspect_payload(
        service_b,
        (recurrence,),
        source_revision=revision,
        renderer_revision=revision,
    )
    assert inspect_payload["reusable_count"] == 1
    assert inspect_payload["statuses"] == {ResultStatus.OK.value: 1}
    assert str(campaign_a) not in json.dumps(inspect_payload, sort_keys=True)

    merged_payloads, merged_count = manual_campaign._merge_lightweight_snapshot(
        service_b,
        {recurrence.cell_id: lightweight},
    )
    assert merged_count == 1
    merged_measurement = next(
        entry["measurement"]
        for payload in merged_payloads.values()
        for entry in payload["entries"]
        if entry["cell_id"] == recurrence.cell_id
    )
    assert Path(str(merged_measurement["artifact"]["path"])).is_relative_to(state_b)
    assert str(campaign_a) not in json.dumps(merged_measurement, sort_keys=True)

    recipe = manual_campaign.reproduction_recipe(
        recurrence,
        repo_root=ROOT,
        artifact_root=state_b,
        measurement=lightweight.result,
    )
    reproduction_root = state_b / "manual-reproductions" / recurrence.cell_id
    recipe_commands = tuple(
        command
        for command in (recipe.prepare, recipe.generate, recipe.profile)
        if command is not None
    )
    rendered_recipe = "\n".join(" ".join(command) for command in recipe_commands)
    assert str(reproduction_root) in rendered_recipe
    assert str(campaign_b / ".artifacts") not in rendered_recipe
    assert str(campaign_a) not in rendered_recipe

    recycled_workers = manual_campaign._recycled_attention_workers(
        (recurrence,),
        {recurrence.cell_id: lightweight},
        {recurrence.cell_id},
        repo_root=ROOT,
        artifact_root=state_b,
        settings=manual_campaign.ReproductionSettings(),
    )
    recycled = recycled_workers[recurrence.cell_id]
    assert recycled.status == "recycled"
    assert recycled.recycled
    rendered_worker_commands = "\n".join(
        command
        for command in (
            recycled.reproduce_prepare,
            recycled.reproduce_generate,
            recycled.reproduce_profile,
        )
        if command is not None
    )
    assert str(reproduction_root) in rendered_worker_commands
    assert str(campaign_b / ".artifacts") not in rendered_worker_commands
    assert str(campaign_a) not in rendered_worker_commands


def test_manual_portable_reuse_rejects_old_absolute_current(
    tmp_path: Path,
) -> None:
    revision = "b" * 40
    cell = _pyamplicol_leaf_cell()
    campaign = tmp_path / "manual"
    state = campaign / "campaign_artifacts"
    ordinary = ReportService(
        ReportPaths(
            repo_root=ROOT,
            docs_dir=campaign,
            results_dir=campaign / "results",
            artifact_root=state,
            coordination_root=state / "coordination",
        )
    )
    _publish_success_current(ordinary, cell, revision=revision)
    portable = _portable_manual_service(campaign)

    with pytest.raises(ManifestValidationError, match="current pointer schema"):
        portable.store.load_current(cell.cell_id)
    lightweight = lightweight_current(
        portable.store,
        cell,
        source_revision=revision,
    )
    assert lightweight is None


def _git_object(git_dir: Path, kind: bytes, body: bytes) -> str:
    payload = kind + b" " + str(len(body)).encode("ascii") + b"\0" + body
    digest = hashlib.sha1(payload, usedforsecurity=False).hexdigest()
    path = git_dir / "objects" / digest[:2] / digest[2:]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(zlib.compress(payload))
    return digest


def _git(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


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


def test_lightweight_source_identity_accepts_clean_packed_head(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "campaign@example.invalid")
    _git(repo, "config", "user.name", "Campaign Test")
    (repo / "tracked.py").write_text("value = 1\n", encoding="utf-8")
    _git(repo, "add", "tracked.py")
    _git(repo, "commit", "-q", "-m", "fixture")
    revision = _git(repo, "rev-parse", "HEAD")
    tree = _git(repo, "rev-parse", "HEAD^{tree}")

    _git(repo, "gc", "--prune=now")
    loose_commit = repo / ".git" / "objects" / revision[:2] / revision[2:]
    assert not loose_commit.exists()

    identity = manual_campaign.lightweight_source_identity(repo)

    assert identity.revision == revision
    assert identity.tree == tree
    assert identity.dirty_paths == ()

    (repo / "tracked.py").write_text("value = 2\n", encoding="utf-8")
    _git(repo, "add", "tracked.py")

    assert manual_campaign.lightweight_source_identity(repo).dirty_paths == (
        "<staged index differs from HEAD>",
    )


def test_measurement_readiness_distinguishes_absent_runtime_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = ReportSourceIdentity("a" * 40, "b" * 40, ())
    monkeypatch.setattr(
        manual_campaign,
        "_installed_source_revision",
        lambda: None,
    )

    with pytest.raises(
        manual_campaign.ManualCampaignError,
        match="runtime contains no clean source revision",
    ):
        manual_campaign.require_measurement_ready(source)


def test_measurement_readiness_reports_runtime_revision_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = ReportSourceIdentity("a" * 40, "b" * 40, ())
    monkeypatch.setattr(
        manual_campaign,
        "_installed_source_revision",
        lambda: "c" * 40,
    )

    with pytest.raises(
        manual_campaign.ManualCampaignError,
        match=r"checkout=a{40}, installed=c{40}",
    ):
        manual_campaign.require_measurement_ready(source)


def test_installed_source_revision_wraps_package_import_provenance_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def failing_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "pyamplicol._internal.versions":
            raise RuntimeError("stale native build inputs")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing_import)

    with pytest.raises(
        manual_campaign.ManualCampaignError,
        match="runtime provenance cannot be loaded: stale native build inputs",
    ):
        manual_campaign._installed_source_revision()


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


def test_historical_current_reuse_requires_explicit_continuation(
    tmp_path: Path,
) -> None:
    historical_revision = "a" * 40
    active_revision = "b" * 40
    service = ReportService(
        ReportPaths.from_repo(
            ROOT,
            profile="macbook_M3_manual",
            artifact_root=tmp_path / "artifacts",
            coordination_root=tmp_path / "coordination",
        )
    )
    cell = _amplicol_leaf_cell()
    _write_lightweight_terminal_current(
        service,
        cell,
        revision=historical_revision,
    )

    strict = lightweight_current(
        service.store,
        cell,
        source_revision=active_revision,
    )
    continued = lightweight_current(
        service.store,
        cell,
        source_revision=active_revision,
        accept_historical_source=True,
    )

    assert strict is not None and strict.complete and not strict.reusable
    assert strict.reason == "source mismatch"
    assert continued is not None and continued.reusable
    assert continued.reason == "historical resource-capped terminal"

    inspected = manual_campaign._inspect_payload(
        service,
        (cell,),
        source_revision=active_revision,
        renderer_revision=active_revision,
        accept_historical_source=True,
    )
    assert inspected["source_policy"] == "continue_across_revisions"
    assert inspected["source_cohorts"] == {historical_revision: 1}
    assert inspected["reusable_count"] == 1


def test_successful_legacy_current_requires_corrected_numerical_authority(
    tmp_path: Path,
) -> None:
    from tools.performance_report.legacy import (
        LEGACY_NUMERICAL_AUTHORITY_ABI,
        LEGACY_NUMERICAL_AUTHORITY_FIELD,
    )

    revision = "a" * 40
    service = ReportService(
        ReportPaths.from_repo(
            ROOT,
            profile="macbook_M3_manual",
            artifact_root=tmp_path / "artifacts",
            coordination_root=tmp_path / "coordination",
        )
    )
    cell = _amplicol_contracted_cell()
    _manifest_path, result_path = _write_lightweight_success_current(
        service,
        cell,
        revision=revision,
        numerical_relation_correctness={},
    )

    stale = lightweight_current(service.store, cell, source_revision=revision)
    assert stale is not None and stale.complete and not stale.reusable
    assert stale.reason == "legacy numerical authority requires refresh"

    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["validation"][LEGACY_NUMERICAL_AUTHORITY_FIELD] = {
        "abi": LEGACY_NUMERICAL_AUTHORITY_ABI,
        "source": "contracted-generated-library",
    }
    result_path.write_text(json.dumps(result), encoding="utf-8")

    corrected = lightweight_current(service.store, cell, source_revision=revision)
    assert corrected is not None and corrected.complete and corrected.reusable


@pytest.mark.parametrize(
    ("workload", "generation_source", "expected_compatible"),
    (
        (Workload.SELECTED_FLOW, "generated-library-create-mode-1", True),
        (Workload.ALL_FLOW, "direct-imode2-generation-setup", False),
        (Workload.CONTRACTED, "generated-library-create-raw", False),
        (
            Workload.CONTRACTED,
            "direct-imode2-three-quark-line-setup",
            True,
        ),
    ),
)
def test_pre_authority_legacy_compatibility_is_narrowly_path_based(
    workload: Workload,
    generation_source: str,
    expected_compatible: bool,
) -> None:
    cell = next(
        candidate
        for candidate in REPORT_CATALOG.measurement_cells()
        if candidate.measurement.execution_mode is ExecutionMode.AMPLICOL
        and candidate.workload is workload
    )
    result = {
        "status": ResultStatus.OK.value,
        "validation": {},
        "provenance": {"generation_source": generation_source},
    }

    assert (
        manual_campaign._legacy_numerical_authority_compatible(cell, result)
        is expected_compatible
    )


@pytest.mark.parametrize(
    ("extra_provenance", "expected_reusable"),
    (
        (
            {
                "effective_config": {
                    "generation": {"relation_discovery": {"mode": "off"}}
                }
            },
            True,
        ),
        (
            {
                "numerical_relation_correctness": {
                    "abi": "pyamplicol-numerical-current-relation-correctness-v1",
                    "state": "no-applied-relations",
                    "applied_relation_count": 0,
                }
            },
            True,
        ),
        (
            {
                "numerical_relation_correctness": {
                    "abi": "pyamplicol-numerical-current-relation-correctness-v1",
                    "state": "member-scoped-v1",
                    "applied_relation_count": 3,
                }
            },
            True,
        ),
        ({}, False),
        (
            {
                "numerical_relation_correctness": {
                    "abi": "pyamplicol-numerical-current-relation-correctness-v1",
                    "state": "no-applied-relations",
                    "applied_relation_count": 1,
                }
            },
            False,
        ),
        (
            {
                "numerical_relation_correctness": {
                    "abi": "obsolete",
                    "state": "member-scoped-v1",
                    "applied_relation_count": 1,
                }
            },
            False,
        ),
    ),
)
def test_historical_pyamplicol_reuse_requires_compact_relation_safety(
    tmp_path: Path,
    extra_provenance: Mapping[str, object],
    expected_reusable: bool,
) -> None:
    historical_revision = "a" * 40
    active_revision = "b" * 40
    service = ReportService(
        ReportPaths.from_repo(
            ROOT,
            profile="macbook_M3_manual",
            artifact_root=tmp_path / "artifacts",
            coordination_root=tmp_path / "coordination",
        )
    )
    cell = _pyamplicol_leaf_cell()
    _write_lightweight_terminal_current(
        service,
        cell,
        revision=historical_revision,
        extra_provenance=extra_provenance,
    )

    current = lightweight_current(
        service.store,
        cell,
        source_revision=active_revision,
        accept_historical_source=True,
    )

    assert current is not None
    assert current.complete
    assert current.reusable is expected_reusable
    if not expected_reusable:
        assert current.reason == "historical numerical-relation policy is not reusable"


def test_successful_historical_pyamplicol_reuse_is_metadata_only_and_planned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    historical_revision = "a" * 40
    active_revision = "b" * 40
    service = ReportService(
        ReportPaths.from_repo(
            ROOT,
            profile="macbook_M3_manual",
            artifact_root=tmp_path / "artifacts",
            coordination_root=tmp_path / "coordination",
        )
    )
    cell = next(
        candidate
        for candidate in REPORT_CATALOG.measurement_cells()
        if candidate.measurement.execution_mode is ExecutionMode.RECURRENCE
        and candidate.measurement.accuracy is not Accuracy.LC
        and candidate.n_final == 1
    )
    _manifest_path, result_path = _write_lightweight_success_current(
        service,
        cell,
        revision=historical_revision,
        numerical_relation_correctness={
            "abi": "pyamplicol-numerical-current-relation-correctness-v1",
            "state": "no-applied-relations",
            "applied_relation_count": 0,
        },
    )

    def forbidden_load_current(
        _store: ArtifactStore,
        _cell_id: str,
        *,
        missing_ok: bool = False,
    ) -> None:
        del missing_ok
        raise AssertionError("historical planning inspected authenticated history")

    monkeypatch.setattr(ArtifactStore, "load_current", forbidden_load_current)
    requested_batches: list[tuple[str, ...]] = []

    def observe_plan(
        requested: tuple[CellSpec, ...],
        **_kwargs: object,
    ) -> tuple[object, ...]:
        requested_batches.append(tuple(item.cell_id for item in requested))
        return ()

    monkeypatch.setattr(manual_campaign, "plan_campaign", observe_plan)
    arguments = build_parser().parse_args(
        (
            "run",
            "--dry-run",
            "--no-color",
            "--continue-across-revisions",
            "--cell-id",
            cell.cell_id,
        )
    )
    source = ReportSourceIdentity(active_revision, "c" * 40, ())

    assert (
        _run_campaign(
            arguments,
            repo_root=ROOT,
            service=service,
            source=source,
            cells=(cell,),
            palette=Palette(False),
        )
        == 0
    )
    assert requested_batches == [()]
    capsys.readouterr()

    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["provenance"]["numerical_relation_correctness"] = {
        "abi": "obsolete",
        "state": "member-scoped-v1",
        "applied_relation_count": 1,
    }
    result_path.write_text(json.dumps(result), encoding="utf-8")

    assert (
        _run_campaign(
            arguments,
            repo_root=ROOT,
            service=service,
            source=source,
            cells=(cell,),
            palette=Palette(False),
        )
        == 0
    )
    assert requested_batches == [(), (cell.cell_id,)]


def test_same_source_pyamplicol_reuse_does_not_require_historical_abi(
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
    cell = _pyamplicol_leaf_cell()
    _write_lightweight_terminal_current(service, cell, revision=revision)

    current = lightweight_current(
        service.store,
        cell,
        source_revision=revision,
    )

    assert current is not None and current.reusable


def test_continuation_keeps_direct_history_but_not_dependency_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    historical_revision = "a" * 40
    active_revision = "b" * 40
    service = ReportService(
        ReportPaths.from_repo(
            ROOT,
            profile="macbook_M3_manual",
            artifact_root=tmp_path / "artifacts",
            coordination_root=tmp_path / "coordination",
        )
    )
    historical = _amplicol_leaf_cell()
    missing = next(
        cell
        for cell in REPORT_CATALOG.measurement_cells()
        if cell.measurement.execution_mode is ExecutionMode.AMPLICOL
        and cell.workload is Workload.SELECTED_FLOW
        and cell.cell_id != historical.cell_id
    )
    _write_lightweight_terminal_current(
        service,
        historical,
        revision=historical_revision,
    )
    observed_requested: list[str] = []

    def observe_plan(
        requested: tuple[CellSpec, ...],
        **kwargs: object,
    ) -> tuple[object, ...]:
        observed_requested.extend(cell.cell_id for cell in requested)
        resolver = kwargs["current_resolver"]
        assert callable(resolver)
        assert resolver(historical) is None
        return ()

    monkeypatch.setattr(manual_campaign, "plan_campaign", observe_plan)
    arguments = build_parser().parse_args(
        (
            "run",
            "--dry-run",
            "--no-color",
            "--continue-across-revisions",
        )
    )

    result = _run_campaign(
        arguments,
        repo_root=ROOT,
        service=service,
        source=ReportSourceIdentity(active_revision, "c" * 40, ()),
        cells=(historical, missing),
        palette=Palette(False),
    )

    assert result == 0
    assert observed_requested == [missing.cell_id]
    output = capsys.readouterr().out
    assert "Cross-revision continuation is enabled" in output
    assert historical_revision in output


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
