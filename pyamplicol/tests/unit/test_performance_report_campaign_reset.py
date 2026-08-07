# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.performance_report.artifacts import ArtifactStore
from tools.performance_report.campaign_policy import MACBOOK_M3_POLICY
from tools.performance_report.campaign_reset import (
    BASELINE_GATE_FILENAME,
    CAMPAIGN_MARKER_FILENAME,
    EXPECTED_AMPLICOL_CELL_COUNT,
    EXPECTED_CATALOG_CELL_COUNT,
    EXPECTED_NON_AMPLICOL_CELL_COUNT,
    CampaignResetError,
    OriginalAmplicolSeed,
    ResetTransactionPaths,
    _lc_seed_family_rejections,
    _legacy_contract,
    assert_campaign_marker,
    build_seed_manifest,
    campaign_marker,
    commit_or_recover_reset,
    mark_campaign_ready,
    sha256_payload,
    stage_seed_store,
    write_prepared_journal,
)
from tools.performance_report.catalog import REPORT_CATALOG
from tools.performance_report.models import (
    Accuracy,
    ArtifactPolicy,
    ExecutionMode,
    Workload,
)
from tools.performance_report.scheduler import CampaignSettings, plan_campaign
from tools.performance_report.service import ReportPaths, ReportService
from tools.performance_report.source_identity import ReportSourceIdentity

_REVISION = "1" * 40
_TREE = "2" * 40
_SHA = "3" * 64


def _store(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(
        artifact_root=tmp_path / "artifacts",
        lock_root=tmp_path / "locks",
    )


def _cell(mode: ExecutionMode):
    return next(
        cell
        for cell in REPORT_CATALOG.measurement_cells()
        if cell.measurement.execution_mode is mode
        and (
            mode is not ExecutionMode.AMPLICOL
            or cell.measurement.accuracy is not Accuracy.LC
        )
    )


def test_reset_cardinality_matches_the_current_catalog() -> None:
    cells = REPORT_CATALOG.measurement_cells()
    amplicol = sum(
        cell.measurement.execution_mode is ExecutionMode.AMPLICOL for cell in cells
    )

    assert EXPECTED_CATALOG_CELL_COUNT == len(cells) == 1962
    assert EXPECTED_AMPLICOL_CELL_COUNT == amplicol == 314
    assert EXPECTED_NON_AMPLICOL_CELL_COUNT == len(cells) - amplicol == 1648


def _contract() -> dict[str, object]:
    return {
        "legacy_revision": "4" * 40,
        "report_source_revision": _REVISION,
        "report_source_tree": _TREE,
        "compiler_sha256": "5" * 64,
        "selector_contract_sha256": "6" * 64,
        "validation_sha256": "7" * 64,
        "resource_evidence_sha256": "8" * 64,
        "target_runtime_seconds": 5.0,
        "workload_specific_generation": True,
        "row_selection_policy": "row-v1",
        "selector_color_word_policy": "selector-v1",
    }


def _publish(store: ArtifactStore, cell_id: str):
    return store.new_attempt(
        cell_id,
        ArtifactPolicy.REGENERATE,
    ).publish(
        {
            "status": "ok",
            "provenance": {
                "report_source_revision": _REVISION,
                "report_source_tree": _TREE,
                "report_measured_source_revision": _REVISION,
                "report_measured_source_tree": _TREE,
            },
        }
    )


def _activate_marker(
    root: Path,
    seed: dict[str, object],
    *,
    profile: str,
    revision: str,
    tree: str,
) -> None:
    marker = campaign_marker(
        campaign_id="campaign-1",
        profile=profile,
        source_revision=revision,
        source_tree=tree,
        policy_sha256=_SHA,
        seed_manifest_sha256=str(seed["seed_manifest_sha256"]),
        archive_manifest_sha256=_SHA,
    )
    (root / CAMPAIGN_MARKER_FILENAME).write_text(json.dumps(marker))
    gate: dict[str, object] = {
        "campaign_id": "campaign-1",
        "profile": profile,
        "source_revision": revision,
        "source_tree": tree,
        "prepared_marker_sha256": marker["marker_sha256"],
    }
    gate["baseline_gate_sha256"] = sha256_payload(gate)
    (root / BASELINE_GATE_FILENAME).write_text(json.dumps(gate))
    mark_campaign_ready(
        root,
        baseline_gate_sha256=str(gate["baseline_gate_sha256"]),
    )


def test_contracted_amplicol_contract_accepts_no_selector() -> None:
    record = SimpleNamespace(
        cell_id="reference-amplicol-full-example",
        result={
            "selector_contract": None,
            "resources": {"monitor": "test", "peak_rss_bytes": 1},
            "validation": {
                "status": "ok",
                "method": "independent-original-amplicol-oracle",
                "point_digest": _SHA,
            },
            "provenance": {
                "method": "original-amplicol-generated-library",
                "revision": _REVISION,
                "report_source_revision": _REVISION,
                "report_source_tree": _TREE,
                "report_measured_source_revision": _REVISION,
                "report_measured_source_tree": _TREE,
                "report_source_clean": True,
                "compiler": {
                    "identity": "gfortran",
                    "version": "GNU Fortran",
                    "flags": ["-O3"],
                    "target": "test-target",
                    "executable_sha256": _SHA,
                },
                "target_runtime_seconds": 5.0,
                "runtime_profile": {
                    "measurement": {"target_runtime_achieved": True}
                },
                "generation_timing_is_workload_specific": True,
                "row_selection_policy": "row-v1",
                "selector_color_word_policy": "selector-v1",
            },
        },
    )

    contract = _legacy_contract(
        record,
        requires_selector=False,
        expected_legacy_revision=_REVISION,
    )

    assert contract["selector_contract_sha256"] is None


def _lc_record(
    cell_id: str,
    *,
    point: str = "a" * 64,
    common_value: float = 1.0,
) -> SimpleNamespace:
    selector = {
        "selected_color_flow_ids": ["flow:2,1"],
        "selected_color_words": [[2, 1]],
        "all_flow_helicity_ids": ["h:-1,+1"],
        "all_flow_source_helicities": {"1": -1, "2": 1},
        "point_digest": point,
    }
    return SimpleNamespace(
        cell_id=cell_id,
        result={
            "selector_contract": selector,
            "resources": {"monitor": "test", "peak_rss_bytes": 1},
            "validation": {
                "status": "ok",
                "method": "independent-original-amplicol-oracle",
                "point_digest": point,
                "lc_common_component": {
                    "abi": "pyamplicol-report-lc-common-component-v1",
                    "cell_id": cell_id,
                    "value": common_value,
                    "point_digest": point,
                    "helicity_ids": ["h:-1,+1"],
                    "color_flow_ids": ["flow:2,1"],
                },
            },
            "provenance": {
                "method": "original-amplicol-generated-library",
                "revision": _REVISION,
                "report_source_revision": _REVISION,
                "report_source_tree": _TREE,
                "report_measured_source_revision": _REVISION,
                "report_measured_source_tree": _TREE,
                "report_source_clean": True,
                "compiler": {
                    "identity": "gfortran",
                    "version": "GNU Fortran",
                    "flags": ["-O3"],
                    "target": "test-target",
                    "executable_sha256": _SHA,
                },
                "target_runtime_seconds": 5.0,
                "runtime_profile": {
                    "measurement": {"target_runtime_achieved": True}
                },
                "generation_timing_is_workload_specific": True,
                "row_selection_policy": "row-v1",
                "selector_color_word_policy": "selector-v1",
            },
        },
    )


def test_lc_seed_rejects_stale_point_and_structural_zero() -> None:
    record = _lc_record("reference-amplicol-lc-example-selected-flow")

    contract = _legacy_contract(
        record,
        requires_selector=True,
        expected_legacy_revision=_REVISION,
        expected_point_digest="a" * 64,
    )
    assert contract["selector_contract_sha256"] is not None

    with pytest.raises(CampaignResetError, match="predates"):
        _legacy_contract(
            record,
            requires_selector=True,
            expected_legacy_revision=_REVISION,
            expected_point_digest="b" * 64,
        )
    with pytest.raises(CampaignResetError, match="structural zero"):
        _legacy_contract(
            _lc_record(record.cell_id, common_value=0.0),
            requires_selector=True,
            expected_legacy_revision=_REVISION,
            expected_point_digest="a" * 64,
        )


def test_lc_seed_family_requires_matching_selected_and_all_flow() -> None:
    selected = next(
        cell
        for cell in REPORT_CATALOG.measurement_cells()
        if cell.measurement.execution_mode is ExecutionMode.AMPLICOL
        and cell.measurement.accuracy is Accuracy.LC
        and cell.workload is Workload.SELECTED_FLOW
    )
    all_flow = next(
        cell
        for cell in REPORT_CATALOG.measurement_cells()
        if cell.dataset_id == selected.dataset_id
        and cell.process == selected.process
        and cell.n_final == selected.n_final
        and cell.measurement == selected.measurement
        and cell.workload is Workload.ALL_FLOW
    )
    cells = {
        selected.cell_id: selected,
        all_flow.cell_id: all_flow,
    }
    records = {
        selected.cell_id: _lc_record(selected.cell_id),
        all_flow.cell_id: _lc_record(all_flow.cell_id),
    }

    assert (
        _lc_seed_family_rejections(
            cells=cells,
            records_by_cell=records,
            admitted_cell_ids=set(cells),
        )
        == {}
    )

    records[all_flow.cell_id].result["selector_contract"][
        "selected_color_flow_ids"
    ] = ["flow:1,2"]
    rejected = _lc_seed_family_rejections(
        cells=cells,
        records_by_cell=records,
        admitted_cell_ids=set(cells),
    )
    assert set(rejected) == set(cells)
    assert all("contracts differ" in reason for reason in rejected.values())

    incomplete = _lc_seed_family_rejections(
        cells=cells,
        records_by_cell=records,
        admitted_cell_ids={selected.cell_id},
    )
    assert incomplete == {
        selected.cell_id: (
            "LC seed family is incomplete; selected/all-flow rows must be "
            "remeasured under one selector authority"
        )
    }


def test_seed_selects_only_ok_original_amplicol_and_preserves_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _store(tmp_path / "source")
    reference = _cell(ExecutionMode.AMPLICOL)
    candidate = _cell(ExecutionMode.RECURRENCE)
    original = _publish(source, reference.cell_id)
    _publish(source, candidate.cell_id)
    monkeypatch.setattr(
        "tools.performance_report.campaign_reset.validate_measurement",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "tools.performance_report.campaign_reset.validate_policy_measurement",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "tools.performance_report.campaign_reset._legacy_contract",
        lambda _record, **_kwargs: _contract(),
    )

    seed = build_seed_manifest(
        profile="macbook_M3",
        store=source,
        policy=MACBOOK_M3_POLICY,
        final_source_revision="9" * 40,
        final_source_tree="a" * 40,
        expected_legacy_revision="4" * 40,
    )
    assert seed["seed_count"] == 1
    assert [pin["cell_id"] for pin in seed["pins"]] == [reference.cell_id]

    destination = stage_seed_store(
        source_store=source,
        destination_root=tmp_path / "fresh-artifacts",
        destination_lock_root=tmp_path / "fresh-locks",
        seed_manifest=seed,
    )
    inherited = destination.load_current(reference.cell_id)
    assert inherited is not None
    assert inherited.attempt_id == original.attempt_id
    assert destination.load_current(candidate.cell_id, missing_ok=True) is None
    assert destination.cell_attempt_ids(reference.cell_id) == (
        original.attempt_id,
    )
    loaded = OriginalAmplicolSeed.load(
        destination.artifact_root / "original_amplicol_seed.json",
        profile="macbook_M3",
        store=destination,
    )
    assert loaded.source_for_current(
        inherited,
        active_revision="9" * 40,
        active_tree="a" * 40,
    ) == (_REVISION, _TREE)
    assert (
        loaded.source_for_current(
            inherited,
            active_revision="b" * 40,
            active_tree="c" * 40,
        )
        is None
    )


def test_seed_rejects_changed_current_pointer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _store(tmp_path / "source")
    reference = _cell(ExecutionMode.AMPLICOL)
    _publish(source, reference.cell_id)
    monkeypatch.setattr(
        "tools.performance_report.campaign_reset.validate_measurement",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "tools.performance_report.campaign_reset.validate_policy_measurement",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "tools.performance_report.campaign_reset._legacy_contract",
        lambda _record, **_kwargs: _contract(),
    )
    seed = build_seed_manifest(
        profile="macbook_M3",
        store=source,
        policy=MACBOOK_M3_POLICY,
        final_source_revision="9" * 40,
        final_source_tree="a" * 40,
        expected_legacy_revision="4" * 40,
    )
    destination = stage_seed_store(
        source_store=source,
        destination_root=tmp_path / "fresh-artifacts",
        destination_lock_root=tmp_path / "fresh-locks",
        seed_manifest=seed,
    )
    pointer = next(destination.cells_root.glob("*/current.json"))
    payload = json.loads(pointer.read_text())
    payload["manifest_sha256"] = "0" * 64
    pointer.write_text(json.dumps(payload))

    with pytest.raises(Exception, match="digest"):
        OriginalAmplicolSeed.load(
            destination.artifact_root / "original_amplicol_seed.json",
            profile="macbook_M3",
            store=destination,
        )


@pytest.mark.parametrize(
    ("catalog_count", "amplicol_count"),
    ((1646, 284), (1796, 314)),
)
def test_seed_load_accepts_only_supported_historical_catalog_cardinality(
    tmp_path: Path,
    catalog_count: int,
    amplicol_count: int,
) -> None:
    store = _store(tmp_path)
    seed_path = store.artifact_root / "original_amplicol_seed.json"
    seed_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "schema": "pyamplicol-original-amplicol-seed-v1",
        "profile": "x86_EPYC",
        "final_source_revision": _REVISION,
        "final_source_tree": _TREE,
        "expected_legacy_revision": "4" * 40,
        "catalog_cell_count": catalog_count,
        "amplicol_catalog_cell_count": amplicol_count,
        "seed_count": 0,
        "pins": [],
        "rejected": [],
    }
    payload["seed_manifest_sha256"] = sha256_payload(payload)
    seed_path.write_text(json.dumps(payload), encoding="ascii")

    loaded = OriginalAmplicolSeed.load(
        seed_path,
        profile="x86_EPYC",
        store=store,
    )
    assert loaded.pins_by_cell == {}

    payload.pop("seed_manifest_sha256")
    payload["catalog_cell_count"] = catalog_count - 1
    payload["seed_manifest_sha256"] = sha256_payload(payload)
    seed_path.write_text(json.dumps(payload), encoding="ascii")
    with pytest.raises(CampaignResetError, match="catalog coverage differs"):
        OriginalAmplicolSeed.load(
            seed_path,
            profile="x86_EPYC",
            store=store,
        )


def test_reset_transaction_moves_old_roots_and_is_idempotent(
    tmp_path: Path,
) -> None:
    source_publication = tmp_path / "old-publication"
    source_artifacts = tmp_path / "old-artifacts"
    source_coordination = tmp_path / "old-coordination"
    destination_publication = tmp_path / "new-publication"
    destination_artifacts = tmp_path / "new-artifacts"
    destination_coordination = tmp_path / "new-coordination"
    archive = tmp_path / "archive"
    staging = tmp_path / "staging"
    for path, name in (
        (source_publication, "old-doc"),
        (source_artifacts, "old-artifact"),
        (source_coordination, "old-lock"),
        (staging / "archive-publication", "archived-doc"),
        (staging / "fresh-artifacts", "seed"),
        (staging / "fresh-coordination", "fresh-lock"),
        (staging / "fresh-publication", "fresh-doc"),
    ):
        path.mkdir(parents=True)
        (path / name).write_text(name)
    (staging / "archive-manifest.json").write_text("{}")
    paths = ResetTransactionPaths(
        source_publication=source_publication,
        source_artifact_root=source_artifacts,
        source_coordination_root=source_coordination,
        destination_publication=destination_publication,
        destination_artifact_root=destination_artifacts,
        destination_coordination_root=destination_coordination,
        archive_root=archive,
        staging_root=staging,
        guard_path=tmp_path / "reset.guard",
    )
    write_prepared_journal(
        profile="macbook_M3",
        campaign_id="campaign-1",
        archive_id="archive-1",
        paths=paths,
        archive_manifest_sha256=_SHA,
        seed_manifest_sha256=_SHA,
        marker_sha256=_SHA,
    )

    first = commit_or_recover_reset(paths.journal_path)
    second = commit_or_recover_reset(paths.journal_path)

    assert first["state"] == second["state"] == "COMMITTED"
    assert (archive / "artifacts" / "old-artifact").is_file()
    assert (archive / "coordination" / "old-lock").is_file()
    assert (archive / "publication" / "archived-doc").is_file()
    assert (destination_artifacts / "seed").is_file()
    assert (destination_coordination / "fresh-lock").is_file()
    assert (destination_publication / "fresh-doc").is_file()


def test_campaign_marker_is_digest_bound(tmp_path: Path) -> None:
    marker = campaign_marker(
        campaign_id="campaign-1",
        profile="x86_EPYC",
        source_revision=_REVISION,
        source_tree=_TREE,
        policy_sha256=_SHA,
        seed_manifest_sha256=_SHA,
        archive_manifest_sha256=_SHA,
    )
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / CAMPAIGN_MARKER_FILENAME).write_text(json.dumps(marker))

    with pytest.raises(CampaignResetError, match="not READY"):
        assert_campaign_marker(
            root,
            campaign_id="campaign-1",
            profile="x86_EPYC",
            source_revision=_REVISION,
            source_tree=_TREE,
        )
    assert assert_campaign_marker(
        root,
        campaign_id="campaign-1",
        profile="x86_EPYC",
        source_revision=_REVISION,
        source_tree=_TREE,
        require_ready=False,
    )["marker_sha256"] == marker["marker_sha256"]

    marker["campaign_id"] = "different"
    (root / CAMPAIGN_MARKER_FILENAME).write_text(json.dumps(marker))
    with pytest.raises(CampaignResetError, match="differs"):
        assert_campaign_marker(
            root,
            campaign_id="campaign-1",
            profile="x86_EPYC",
            source_revision=_REVISION,
            source_tree=_TREE,
            require_ready=False,
        )


def test_seed_is_authorized_by_scheduler_and_report_merge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _store(tmp_path / "source")
    reference = _cell(ExecutionMode.AMPLICOL)
    _publish(source, reference.cell_id)
    for target in (
        "tools.performance_report.campaign_reset.validate_measurement",
        "tools.performance_report.campaign_reset.validate_policy_measurement",
    ):
        monkeypatch.setattr(target, lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "tools.performance_report.campaign_reset._legacy_contract",
        lambda _record, **_kwargs: _contract(),
    )
    seed = build_seed_manifest(
        profile="macbook_M3",
        store=source,
        policy=MACBOOK_M3_POLICY,
        final_source_revision="9" * 40,
        final_source_tree="a" * 40,
        expected_legacy_revision="4" * 40,
    )
    artifact_root = (
        tmp_path
        / "repo"
        / ".artifacts"
        / "performance-report"
        / "macbook_M3"
    )
    coordination_root = (
        tmp_path
        / "repo"
        / ".artifacts"
        / "performance-report-coordination"
        / "macbook_M3"
    )
    destination = stage_seed_store(
        source_store=source,
        destination_root=artifact_root,
        destination_lock_root=coordination_root,
        seed_manifest=seed,
    )
    _activate_marker(
        artifact_root,
        seed,
        profile="macbook_M3",
        revision="9" * 40,
        tree="a" * 40,
    )
    monkeypatch.setattr(
        "tools.performance_report.scheduler.validate_measurement",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "tools.performance_report.scheduler.validate_policy_measurement",
        lambda *_args, **_kwargs: None,
    )
    settings = CampaignSettings(
        workers=1,
        cell_cores=1,
        target_runtime_seconds=5.0,
        max_rss_bytes=30_000_000_000,
        campaign_policy=MACBOOK_M3_POLICY,
        report_profile="macbook_M3",
        missing_only=True,
    )
    assert plan_campaign(
        (reference,),
        store=destination,
        settings=settings,
        expected_revision="9" * 40,
        expected_tree="a" * 40,
    ) == ()

    repo = tmp_path / "repo"
    docs = repo / "docs" / "performance_reports" / "macbook_M3"
    docs.mkdir(parents=True)
    service = ReportService(
        ReportPaths(
            repo_root=repo,
            docs_dir=docs,
            results_dir=docs / "results",
            artifact_root=artifact_root,
            coordination_root=coordination_root,
        )
    )
    monkeypatch.setattr(
        "tools.performance_report.service.require_eligible_report_source",
        lambda _root: ReportSourceIdentity("9" * 40, "a" * 40, ()),
    )
    monkeypatch.setattr(
        "tools.performance_report.service.validate_measurement",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "tools.performance_report.cache.validate_measurement",
        lambda *_args, **_kwargs: None,
    )
    caches = service.reset_payloads()
    assert service.merge_current(caches) == 1
    measurement = next(
        entry["measurement"]
        for cache in caches.values()
        for entry in cache["entries"]
        if entry["cell_id"] == reference.cell_id
    )
    assert measurement["status"] == "ok"
