# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import errno
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from tools.performance_report.agreements import DIRECT_AGREEMENT_FIELD
from tools.performance_report.cache import empty_measurement, reset_entry
from tools.performance_report.catalog import REPORT_CATALOG
from tools.performance_report.measurement import failure_measurement
from tools.performance_report.models import ArtifactPolicy, ResultStatus
from tools.performance_report.publication import publication_absolute_paths
from tools.performance_report.render import render_all_tables
from tools.performance_report.service import (
    ReportPaths,
    ReportService,
    ReportServiceError,
)
from tools.performance_report.source_identity import require_eligible_report_source


def _service(tmp_path: Path) -> ReportService:
    repo = tmp_path / "repo"
    (repo / "docs/arxiv/results").mkdir(parents=True)
    subprocess.run(("git", "init", "-q"), cwd=repo, check=True)
    subprocess.run(
        ("git", "config", "user.email", "report-tests@example.invalid"),
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ("git", "config", "user.name", "Report Tests"),
        cwd=repo,
        check=True,
    )
    (repo / "README.md").write_text("# report fixture\n", encoding="ascii")
    subprocess.run(("git", "add", "README.md"), cwd=repo, check=True)
    subprocess.run(
        ("git", "commit", "-q", "-m", "Initialize fixture"),
        cwd=repo,
        check=True,
    )
    return ReportService(
        ReportPaths.from_repo(
            repo,
            artifact_root=tmp_path / "artifacts",
        )
    )


def test_reset_publishes_only_canonical_na_caches_and_seventeen_tables(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)

    paths = service.publish(reset=True, merge_artifacts=False)
    result = service.validate()

    assert result["table_count"] == 20
    assert result["statuses"] == {
        "not_available": len(REPORT_CATALOG.measurement_cells())
    }
    assert len([path for path in paths if path.suffix == ".tex"]) == 20
    assert (service.paths.results_dir / "report-cache.schema.json").is_file()


def test_static_na_cache_rejects_terminal_measurement(tmp_path: Path) -> None:
    service = _service(tmp_path)
    caches = service.reset_payloads()
    cell = service.catalog.cell(
        "reference-amplicol-full-n6-dd-4q-lines-contracted"
    )
    payload = caches[f"{cell.dataset_id}.json"]
    entry = next(
        item for item in payload["entries"] if item["cell_id"] == cell.cell_id
    )
    entry["measurement"] = failure_measurement(
        ResultStatus.UNSUPPORTED,
        "historical oracle scope",
    )

    with pytest.raises(
        ReportServiceError,
        match="catalog static N/A cache entry",
    ):
        service.validate_payloads(caches)


def test_static_na_current_never_merges_into_publication_cache(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    cell = service.catalog.cell(
        "reference-amplicol-full-n6-dd-4q-lines-contracted"
    )
    source = require_eligible_report_source(service.paths.repo_root)
    measurement = failure_measurement(
        ResultStatus.UNSUPPORTED,
        "historical oracle scope",
    )
    measurement["provenance"] = {
        "report_source_revision": source.revision,
        "report_source_tree": source.tree,
    }
    published = service.store.new_attempt(
        cell.cell_id,
        ArtifactPolicy.REGENERATE,
    ).publish(measurement)
    caches = service.reset_payloads()
    payload = caches[f"{cell.dataset_id}.json"]
    entry = next(
        item for item in payload["entries"] if item["cell_id"] == cell.cell_id
    )
    entry["measurement"] = failure_measurement(
        ResultStatus.UNSUPPORTED,
        "noncanonical cached terminal",
    )

    assert service.merge_current(caches) == 0
    assert entry["measurement"] == reset_entry(cell)["measurement"]
    assert service.store.load_current(cell.cell_id).attempt_id == published.attempt_id


def test_merge_joins_immutable_current_record_by_cell_id(tmp_path: Path) -> None:
    service = _service(tmp_path)
    service.publish(reset=True, merge_artifacts=False)
    source = require_eligible_report_source(service.paths.repo_root)
    cell = next(
        item
        for item in service.catalog.measurement_cells()
        if item.measurement.accuracy.value != "lc"
    )
    observations = [
        {
            "module": "pyamplicol",
            "kind": "package-member",
            "root_index": 0,
            "path": "__init__.py",
            "size": 1,
            "sha256": "1" * 64,
        }
    ]
    loaded_origin_policy = {
        "kind": "pyamplicol-loaded-module-origin-policy-v1",
        "all_loaded_origins_authenticated": True,
        "native_image_origin_bound": True,
        "loaded_bytecode_eligible": False,
        "observed_module_count": 1,
        "observations": observations,
        "observations_sha256": hashlib.sha256(
            json.dumps(
                observations,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
        ).hexdigest(),
    }
    runtime_identity = {
        "extension": {
            "path": str(service.paths.repo_root / "native/_rusticol.so"),
        },
        "loaded_module_origin_policy": loaded_origin_policy,
    }
    runtime_identity_sha256 = hashlib.sha256(
        json.dumps(
            runtime_identity,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()
    stable_runtime_identity = json.loads(json.dumps(runtime_identity))
    stable_policy = stable_runtime_identity["loaded_module_origin_policy"]
    for field in (
        "observed_module_count",
        "observations",
        "observations_sha256",
    ):
        stable_policy.pop(field)
    runtime_identity_stable_sha256 = hashlib.sha256(
        json.dumps(
            stable_runtime_identity,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()
    measurement = empty_measurement()
    measurement.update(
        {
            "status": "ok",
            "generation_seconds": 1.0,
            "wall_seconds_per_point": 2.0,
            "execution_seconds_per_point": None,
            "matrix_element": 3.0,
            "sample_count": 5,
            "standard_error_seconds_per_point": 0.0,
            "relative_standard_error": 0.0,
            "artifact": {
                "path": str(service.paths.artifact_root / "cells/example/artifact"),
                "log_path": str(
                    service.paths.artifact_root / "cells/example/worker.log"
                ),
            },
            "selector_contract": None,
            "validation": {"status": "ok", DIRECT_AGREEMENT_FIELD: []},
            "resources": {},
            "provenance": {
                "report_source_revision": source.revision,
                "report_source_tree": source.tree,
                "effective_config": {
                    "model": {
                        "cache_dir": str(service.paths.repo_root / ".cache/model"),
                    }
                },
                "runtime_identity": runtime_identity,
                "runtime_identity_sha256": runtime_identity_sha256,
                "runtime_identity_stable_sha256": (
                    runtime_identity_stable_sha256
                ),
                "runtime_identity_postflight_stable_sha256": (
                    runtime_identity_stable_sha256
                ),
                "runtime_identity_postflight_loaded_module_origin_policy": (
                    loaded_origin_policy
                ),
                "runtime_identity_postflight_match": True,
            },
        }
    )
    service.store.new_attempt(cell.cell_id, ArtifactPolicy.REGENERATE).publish(
        measurement
    )

    service.publish()
    cache_path = service.paths.results_dir / f"{cell.dataset_id}.json"
    payload = json.loads(cache_path.read_text(encoding="ascii"))
    entry = next(item for item in payload["entries"] if item["cell_id"] == cell.cell_id)
    assert entry["measurement"]["matrix_element"] == 3.0
    publication = entry["measurement"]
    assert publication["artifact"]["path"].startswith(
        "${PYAMPLICOL_REPORT_ARTIFACT_ROOT}/"
    )
    assert publication["artifact"]["log_path"].startswith(
        "${PYAMPLICOL_REPORT_ARTIFACT_ROOT}/"
    )
    assert publication["provenance"]["effective_config"]["model"][
        "cache_dir"
    ].startswith("${PYAMPLICOL_SOURCE_ROOT}/")
    assert publication["provenance"]["runtime_identity"] == runtime_identity
    retained_identity = publication["provenance"]["runtime_identity"]
    assert (
        hashlib.sha256(
            json.dumps(
                retained_identity,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
        ).hexdigest()
        == publication["provenance"]["runtime_identity_sha256"]
    )
    assert publication_absolute_paths(payload) == ()


def test_failed_snapshot_publication_restores_previous_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)
    service.publish(reset=True, merge_artifacts=False)
    table = service.paths.docs_dir / "result_scalar_contact_table.tex"
    cache = service.paths.results_dir / "scalar_contact.json"
    old_table = table.read_bytes()
    old_cache = cache.read_bytes()

    original_replace = Path.replace
    original_copy2 = shutil.copy2
    calls = 0
    staging_sources: list[Path] = []

    def observe_staging_copy(source: Path, destination: Path) -> Path:
        if ".report-snapshot-" in str(source):
            staging_sources.append(source)
        return original_copy2(source, destination)

    def fail_after_one_replace(source: Path, destination: Path) -> Path:
        nonlocal calls
        if ".report-install-" in source.name:
            calls += 1
            if calls == 3:
                raise OSError("injected publication interruption")
        return original_replace(source, destination)

    monkeypatch.setattr(shutil, "copy2", observe_staging_copy)
    monkeypatch.setattr(Path, "replace", fail_after_one_replace)
    with pytest.raises(OSError, match="injected"):
        service.publish(reset=True, merge_artifacts=False)

    assert table.read_bytes() == old_table
    assert cache.read_bytes() == old_cache
    assert staging_sources
    assert all(
        source.is_relative_to(service.paths.artifact_root)
        for source in staging_sources
    )
    assert not tuple(service.paths.docs_dir.glob(".report-snapshot-*"))
    snapshot_root = service.paths.artifact_root / "report-snapshot-builds"
    assert snapshot_root.is_dir()
    assert not tuple(snapshot_root.iterdir())
    assert not tuple(service.paths.docs_dir.glob(".*.report-*-*"))
    assert not tuple(service.paths.results_dir.glob(".*.report-*-*"))


def test_snapshot_install_remains_atomic_across_artifact_filesystems(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path)
    service.publish(reset=True, merge_artifacts=False)
    original_replace = Path.replace
    installs: list[tuple[Path, Path]] = []

    def reject_cross_directory_replace(source: Path, destination: Path) -> Path:
        if source.parent != destination.parent:
            raise OSError(errno.EXDEV, "injected cross-device replacement")
        installs.append((source, destination))
        return original_replace(source, destination)

    monkeypatch.setattr(Path, "replace", reject_cross_directory_replace)
    written = service.publish(reset=True, merge_artifacts=False)

    assert written
    assert installs
    assert all(source.parent == destination.parent for source, destination in installs)
    assert not tuple(service.paths.docs_dir.glob(".*.report-*-*"))
    assert not tuple(service.paths.results_dir.glob(".*.report-*-*"))


def test_audit_rejects_nonpublication_timing_and_source_evidence(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path)
    caches = service.reset_payloads()
    payload = caches["scalar_contact.json"]
    entry = payload["entries"][0]
    measurement = empty_measurement()
    measurement.update(
        {
            "status": "ok",
            "generation_seconds": 1.0,
            "wall_seconds_per_point": 2.0e-6,
            "execution_seconds_per_point": 1.0e-6,
            "matrix_element": 3.0,
            "sample_count": 5,
            "standard_error_seconds_per_point": 1.0e-9,
            "relative_standard_error": 1.0e-3,
            "artifact": {},
            "selector_contract": None,
            "validation": {"status": "ok", DIRECT_AGREEMENT_FIELD: []},
            "resources": {},
            "provenance": {},
            "failure": None,
        }
    )
    entry["measurement"] = measurement
    service._snapshot_files(
        caches,
        render_all_tables(caches, catalog=service.catalog),
    )

    with pytest.raises(ReportServiceError, match="publication policy"):
        service.audit()
