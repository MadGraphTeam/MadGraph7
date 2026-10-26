# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from tools.performance_report.agreements import (
    DIRECT_AGREEMENT_FIELD,
    LC_COMMON_COMPONENT_ABI,
    LC_COMMON_COMPONENT_FIELD,
)
from tools.performance_report.artifacts import ArtifactStore, CurrentRecord
from tools.performance_report.cache import empty_measurement
from tools.performance_report.campaign_holds import (
    DEPENDENCY_HOLD_REASON,
    PriorHeldDisposition,
    PriorHeldEligibilityError,
    active_prior_held_ids,
    catalog_prerequisite_closure,
    classify_prior_held_cells,
    prior_held_history_record,
)
from tools.performance_report.catalog import REPORT_CATALOG
from tools.performance_report.measurement import failure_measurement
from tools.performance_report.measurement_lineage import MeasurementLineage
from tools.performance_report.models import (
    Accuracy,
    ArtifactPolicy,
    CellSpec,
    ResultStatus,
    Workload,
)

_SOURCE_REVISION = "1" * 40
_SOURCE_TREE = "2" * 40


def _store(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(
        artifact_root=tmp_path / "artifacts",
        lock_root=tmp_path / "coordination",
    )


def _cell(*, workload: Workload = Workload.SELECTED_FLOW) -> CellSpec:
    return next(
        cell
        for cell in REPORT_CATALOG.measurement_cells()
        if cell.dataset_id == "matrix_compiled_builtin_sm_lc"
        and cell.process_key == "dd_z_jets"
        and cell.n_final == 1
        and cell.workload is workload
    )


def _hard_dependency_cell() -> CellSpec:
    return _cell(workload=Workload.ALL_FLOW)


def _ufo_recurrence_cell(
    *, workload: Workload = Workload.SELECTED_FLOW
) -> CellSpec:
    return next(
        cell
        for cell in REPORT_CATALOG.measurement_cells()
        if cell.dataset_id == "matrix_recurrence_ufo_sm_lc"
        and cell.process_key == "dd_z_jets"
        and cell.n_final == 1
        and cell.workload is workload
    )


def _ok_measurement(
    cell: CellSpec,
    *,
    source_revision: str | None = _SOURCE_REVISION,
    source_tree: str | None = _SOURCE_TREE,
) -> dict[str, object]:
    measurement = empty_measurement()
    selector = (
        None
        if cell.measurement.accuracy is not Accuracy.LC
        else {
            "selected_color_flow_ids": ["flow:2,1"],
            "selected_color_words": [[2, 1]],
            "all_flow_helicity_ids": ["h:-1,+1,-1"],
            "all_flow_source_helicities": {"1": -1, "2": 1, "3": -1},
            "point_digest": "a" * 64,
        }
    )
    validation: dict[str, object] = {
        "status": ResultStatus.OK.value,
        DIRECT_AGREEMENT_FIELD: [],
    }
    if selector is not None:
        validation[LC_COMMON_COMPONENT_FIELD] = {
            "abi": LC_COMMON_COMPONENT_ABI,
            "cell_id": cell.cell_id,
            "value": 1.0,
            "point_digest": selector["point_digest"],
            "helicity_ids": selector["all_flow_helicity_ids"],
            "color_flow_ids": selector["selected_color_flow_ids"],
        }
    measurement.update(
        {
            "status": ResultStatus.OK.value,
            "generation_seconds": 1.0,
            "wall_seconds_per_point": 1.0e-6,
            "execution_seconds_per_point": 8.0e-7,
            "matrix_element": 1.0,
            "sample_count": 5,
            "standard_error_seconds_per_point": 0.0,
            "relative_standard_error": 0.0,
            "artifact": {},
            "selector_contract": selector,
            "validation": validation,
            "resources": {},
            "provenance": (
                {}
                if source_revision is None
                else {
                    "report_measured_source_revision": source_revision,
                    "report_measured_source_tree": source_tree,
                    "report_source_revision": source_revision,
                    "report_source_tree": source_tree,
                }
            ),
        }
    )
    return measurement


def _publish_ok(store: ArtifactStore, cell: CellSpec) -> None:
    store.new_attempt(cell.cell_id, ArtifactPolicy.REGENERATE).publish(
        _ok_measurement(cell)
    )


def _publish_prerequisites(
    store: ArtifactStore,
    target: CellSpec,
) -> tuple[CellSpec, ...]:
    prerequisites = catalog_prerequisite_closure(target)
    for prerequisite in prerequisites:
        _publish_ok(store, prerequisite)
    return prerequisites


def _hold_observation(index: int, reason: str) -> dict[str, object]:
    summary_path = f"/summaries/{index:04d}.json"
    return {
        "reason": reason,
        "summary_path": summary_path,
        "summary_sha256": hashlib.sha256(summary_path.encode()).hexdigest(),
    }


def _prior(
    target: CellSpec,
    *,
    reasons: tuple[str, ...] = (DEPENDENCY_HOLD_REASON,),
) -> dict[str, object]:
    return {
        target.cell_id: prior_held_history_record(
            target.cell_id,
            (_hold_observation(index, reason) for index, reason in enumerate(reasons)),
        )
    }


def _classify(
    store: ArtifactStore,
    prior_held_records: dict[str, object],
    **kwargs: Any,
) -> dict[str, PriorHeldDisposition]:
    kwargs.setdefault("expected_source_revision", _SOURCE_REVISION)
    kwargs.setdefault("expected_source_tree", _SOURCE_TREE)
    return classify_prior_held_cells(
        store,
        prior_held_records,
        **kwargs,
    )


def _active_prior_held_ids(
    store: ArtifactStore,
    prior_held_records: dict[str, object],
) -> frozenset[str]:
    return active_prior_held_ids(
        store,
        prior_held_records,
        expected_source_revision=_SOURCE_REVISION,
        expected_source_tree=_SOURCE_TREE,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_dependency_hold_becomes_eligible_after_every_prerequisite_is_ok(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    target = _hard_dependency_cell()
    prerequisites = catalog_prerequisite_closure(target)
    assert len(prerequisites) == 1
    assert prerequisites[0] == _cell()

    before = _classify(store, _prior(target))[target.cell_id]
    assert before.eligible is False
    assert {state for _cell_id, state in before.blocking_prerequisites} == {"missing"}

    for prerequisite in prerequisites:
        _publish_ok(store, prerequisite)
    after = _classify(store, _prior(target))[target.cell_id]
    assert after.eligible is True
    assert after.reason == "eligible"
    assert _active_prior_held_ids(store, _prior(target)) == frozenset()


def test_compiled_recurrence_authority_is_not_a_hold_prerequisite(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    target = _cell()
    authority = REPORT_CATALOG.validation_baseline_cell(target)

    assert authority is not None
    assert authority.measurement.execution_mode.value == "recurrence"
    assert catalog_prerequisite_closure(target) == ()
    assert _classify(store, _prior(target))[target.cell_id].eligible is True


@pytest.mark.parametrize(
    "reason",
    (
        "exact one-cell plan expanded or changed",
        "signed-zero LC family awaits scoped continuity bridge",
        "future hold reason",
    ),
)
def test_non_dependency_historical_hold_is_never_readmitted(
    tmp_path: Path,
    reason: str,
) -> None:
    store = _store(tmp_path)
    target = _cell()
    _publish_prerequisites(store, target)

    disposition = _classify(
        store,
        _prior(target, reasons=(reason,)),
    )[target.cell_id]

    assert disposition.eligible is False
    assert disposition.reason == "historical-hold-not-dependency"
    assert _active_prior_held_ids(
        store,
        _prior(target, reasons=(reason,)),
    ) == frozenset((target.cell_id,))


def test_later_non_dependency_reason_overrides_earlier_dependency_hold(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    target = _cell()
    _publish_prerequisites(store, target)
    reasons = (
        DEPENDENCY_HOLD_REASON,
        "exact one-cell plan expanded or changed",
    )

    disposition = _classify(
        store,
        _prior(target, reasons=reasons),
    )[target.cell_id]

    assert disposition.eligible is False
    assert disposition.reason == "historical-hold-not-dependency"
    assert disposition.historical_reasons == reasons
    assert disposition.as_dict()["historical_reasons"] == list(reasons)
    assert len(disposition.historical_observations) == 2


@pytest.mark.parametrize(
    "record",
    (
        None,
        {},
        {"reason": None},
        {"reason": ""},
    ),
)
def test_malformed_historical_hold_record_aborts_classification(
    tmp_path: Path,
    record: object,
) -> None:
    store = _store(tmp_path)
    target = _cell()

    with pytest.raises(PriorHeldEligibilityError, match="historical hold"):
        _classify(store, {target.cell_id: record})


def test_historical_hold_history_rejects_bad_digest_and_reason_tampering(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    target = _cell()

    with pytest.raises(PriorHeldEligibilityError, match="observation"):
        prior_held_history_record(
            target.cell_id,
            (
                {
                    "reason": DEPENDENCY_HOLD_REASON,
                    "summary_path": "/summary.json",
                    "summary_sha256": "bad",
                },
            ),
        )
    history = _prior(target)
    raw_record = history[target.cell_id]
    assert isinstance(raw_record, dict)
    tampered = {**raw_record, "reasons": ["different"]}
    with pytest.raises(PriorHeldEligibilityError, match="inconsistent"):
        _classify(store, {target.cell_id: tampered})


def test_optional_amplicol_diagnostic_is_not_a_hold_prerequisite(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    target = _ufo_recurrence_cell()
    (direct,) = catalog_prerequisite_closure(target)
    optional_diagnostic = REPORT_CATALOG.validation_baseline_cell(target)
    assert optional_diagnostic is not None
    assert optional_diagnostic.measurement.execution_mode.value == "amplicol"

    disposition = _classify(
        store,
        _prior(target),
    )[target.cell_id]

    assert disposition.eligible is False
    assert disposition.prerequisite_ids == (direct.cell_id,)
    assert optional_diagnostic.cell_id not in disposition.prerequisite_ids
    assert disposition.blocking_prerequisites == ((direct.cell_id, "missing"),)


@pytest.mark.parametrize(
    "status",
    (
        ResultStatus.ERROR,
        ResultStatus.SKIP,
        ResultStatus.UNSUPPORTED,
    ),
)
def test_terminal_prerequisite_status_keeps_historical_hold(
    tmp_path: Path,
    status: ResultStatus,
) -> None:
    store = _store(tmp_path)
    target = _hard_dependency_cell()
    prerequisites = catalog_prerequisite_closure(target)
    failed = prerequisites[-1]
    for prerequisite in prerequisites:
        result = (
            failure_measurement(status, "terminal")
            if prerequisite == failed
            else _ok_measurement(prerequisite)
        )
        store.new_attempt(
            prerequisite.cell_id,
            ArtifactPolicy.REGENERATE,
        ).publish(result)

    disposition = _classify(
        store,
        _prior(target),
    )[target.cell_id]

    assert disposition.eligible is False
    assert (failed.cell_id, status.value) in disposition.blocking_prerequisites


@pytest.mark.parametrize("with_current", (False, True))
def test_target_is_never_reselected_after_any_attempt(
    tmp_path: Path,
    with_current: bool,
) -> None:
    store = _store(tmp_path)
    target = _cell()
    _publish_prerequisites(store, target)
    attempt = store.new_attempt(target.cell_id, ArtifactPolicy.REGENERATE)
    if with_current:
        attempt.publish(_ok_measurement(target))
    else:
        attempt.mark_failed("worker failed")

    disposition = _classify(
        store,
        _prior(target),
    )[target.cell_id]

    assert disposition.eligible is False
    assert disposition.reason == "target-already-attempted"
    assert disposition.target_attempt_ids == (attempt.attempt_id,)


def test_active_scoped_hold_blocks_target_and_dependency_descendants(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    target = _hard_dependency_cell()
    prerequisites = _publish_prerequisites(store, target)
    active_dependency = prerequisites[0]

    direct = _classify(
        store,
        _prior(target),
        active_scoped_hold_ids=(target.cell_id,),
    )[target.cell_id]
    descendant = _classify(
        store,
        _prior(target),
        active_scoped_hold_ids=(active_dependency.cell_id,),
    )[target.cell_id]

    assert direct.reason == "active-scoped-defect-hold"
    assert descendant.blocking_prerequisites == (
        (active_dependency.cell_id, "active-scoped-defect-hold"),
    )


def test_agreement_equivalence_peers_are_required_before_readmission(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    target = _cell(workload=Workload.ALL_FLOW)
    prerequisites = catalog_prerequisite_closure(target)
    selected_peer = _cell()
    assert selected_peer.cell_id in {
        prerequisite.cell_id for prerequisite in prerequisites
    }
    for prerequisite in prerequisites:
        if prerequisite != selected_peer:
            _publish_ok(store, prerequisite)

    blocked = _classify(
        store,
        _prior(target),
    )[target.cell_id]
    assert (selected_peer.cell_id, "missing") in blocked.blocking_prerequisites

    _publish_ok(store, selected_peer)
    assert _classify(
        store,
        _prior(target),
    )[target.cell_id].eligible


def test_unknown_or_invalid_current_fails_closed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    target = _hard_dependency_cell()
    prerequisite = catalog_prerequisite_closure(target)[0]
    store.new_attempt(
        prerequisite.cell_id,
        ArtifactPolicy.REGENERATE,
    ).publish({"status": "future"})

    with pytest.raises(
        PriorHeldEligibilityError,
        match="current measurement is invalid",
    ):
        _classify(store, _prior(target))
    with pytest.raises(PriorHeldEligibilityError, match="unknown catalog cells"):
        _classify(
            store,
            {"unknown-cell": {"reason": DEPENDENCY_HOLD_REASON}},
        )
    with pytest.raises(PriorHeldEligibilityError, match="invalid catalog cell"):
        _classify(
            store,
            {None: {"reason": DEPENDENCY_HOLD_REASON}},  # type: ignore[dict-item]
        )


@pytest.mark.parametrize("authenticated", (True, False, None, "yes"))
def test_additional_current_authenticator_is_strictly_boolean(
    tmp_path: Path,
    authenticated: object,
) -> None:
    store = _store(tmp_path)
    target = _hard_dependency_cell()
    _publish_prerequisites(store, target)

    disposition = classify_prior_held_cells(
        store,
        _prior(target),
        authenticate_current=lambda _cell, _current: authenticated,  # type: ignore[return-value]
    )[target.cell_id]

    assert disposition.eligible is (authenticated is True)
    states = {state for _cell_id, state in disposition.blocking_prerequisites}
    assert states == (set() if authenticated is True else {"unauthenticated"})


def test_measurement_lineage_authenticates_inherited_prerequisite_source(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    target = _hard_dependency_cell()
    prerequisites = catalog_prerequisite_closure(target)
    ancestor = prerequisites[0]
    ancestor_revision = "3" * 40
    ancestor_tree = "4" * 40
    for prerequisite in prerequisites:
        store.new_attempt(
            prerequisite.cell_id,
            ArtifactPolicy.REGENERATE,
        ).publish(
            _ok_measurement(
                prerequisite,
                source_revision=(
                    ancestor_revision if prerequisite == ancestor else _SOURCE_REVISION
                ),
                source_tree=(
                    ancestor_tree if prerequisite == ancestor else _SOURCE_TREE
                ),
            )
        )
    ancestor_current = store.load_current(ancestor.cell_id)
    assert ancestor_current is not None
    ancestor_pin = {
        "attempt_id": ancestor_current.attempt_id,
        "current_pointer_sha256": _sha256(
            ancestor_current.manifest_path.parent.parent.parent / "current.json"
        ),
        "manifest_sha256": ancestor_current.manifest_sha256,
        "result_sha256": _sha256(ancestor_current.result_path),
        "source_revision": ancestor_revision,
        "source_tree": ancestor_tree,
    }
    lineage = MeasurementLineage(
        payload={},
        retained_by_cell={ancestor.cell_id: ancestor_pin},
        invalidated_cell_ids=frozenset(),
        recompare_cell_ids=frozenset(),
        required_descendant_cell_ids=frozenset(),
        inherited_environments_by_revision={},
    )

    def authenticate(_cell: CellSpec, current: CurrentRecord) -> bool:
        return (
            lineage.source_for_current(
                current,
                active_revision=_SOURCE_REVISION,
                active_tree=_SOURCE_TREE,
            )
            is not None
        )

    accepted = classify_prior_held_cells(
        store,
        _prior(target),
        authenticate_current=authenticate,
    )[target.cell_id]
    rejected_lineage = MeasurementLineage(
        payload={},
        retained_by_cell={},
        invalidated_cell_ids=frozenset(),
        recompare_cell_ids=frozenset(),
        required_descendant_cell_ids=frozenset(),
        inherited_environments_by_revision={},
    )
    rejected = classify_prior_held_cells(
        store,
        _prior(target),
        authenticate_current=lambda _cell, current: (
            rejected_lineage.source_for_current(
                current,
                active_revision=_SOURCE_REVISION,
                active_tree=_SOURCE_TREE,
            )
            is not None
        ),
    )[target.cell_id]

    assert accepted.eligible is True
    assert rejected.eligible is False
    assert (ancestor.cell_id, "unauthenticated") in (rejected.blocking_prerequisites)


def test_source_authentication_policy_is_required(tmp_path: Path) -> None:
    store = _store(tmp_path)
    target = _cell()

    with pytest.raises(
        PriorHeldEligibilityError,
        match="source authentication policy",
    ):
        classify_prior_held_cells(store, _prior(target))


@pytest.mark.parametrize(
    ("revision", "tree"),
    (
        ("", ""),
        ("not-a-git-object", "2" * 40),
        ("1" * 39, "2" * 40),
        ("1" * 40, "z" * 40),
    ),
)
def test_single_source_authentication_rejects_malformed_git_identities(
    tmp_path: Path,
    revision: str,
    tree: str,
) -> None:
    store = _store(tmp_path)
    target = _cell()
    _publish_prerequisites(store, target)

    with pytest.raises(PriorHeldEligibilityError, match="Git object identities"):
        classify_prior_held_cells(
            store,
            _prior(target),
            expected_source_revision=revision,
            expected_source_tree=tree,
        )


def test_expected_source_identity_blocks_stale_prerequisites(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    target = _hard_dependency_cell()
    for prerequisite in catalog_prerequisite_closure(target):
        store.new_attempt(
            prerequisite.cell_id,
            ArtifactPolicy.REGENERATE,
        ).publish(
            _ok_measurement(
                prerequisite,
                source_revision="5" * 40,
                source_tree="6" * 40,
            )
        )

    disposition = classify_prior_held_cells(
        store,
        _prior(target),
        expected_source_revision=_SOURCE_REVISION,
        expected_source_tree=_SOURCE_TREE,
    )[target.cell_id]

    assert disposition.eligible is False
    assert {state for _cell_id, state in disposition.blocking_prerequisites} == {
        "source-stale"
    }
    with pytest.raises(PriorHeldEligibilityError, match="provided together"):
        classify_prior_held_cells(
            store,
            _prior(target),
            expected_source_tree=_SOURCE_TREE,
        )


def test_single_source_authentication_checks_measured_identity(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    target = _hard_dependency_cell()
    stale = catalog_prerequisite_closure(target)[0]
    for prerequisite in catalog_prerequisite_closure(target):
        measurement = _ok_measurement(prerequisite)
        if prerequisite == stale:
            provenance = measurement["provenance"]
            assert isinstance(provenance, dict)
            provenance["report_measured_source_revision"] = "5" * 40
        store.new_attempt(
            prerequisite.cell_id,
            ArtifactPolicy.REGENERATE,
        ).publish(measurement)

    disposition = _classify(store, _prior(target))[target.cell_id]

    assert disposition.eligible is False
    assert (stale.cell_id, "source-stale") in (disposition.blocking_prerequisites)
