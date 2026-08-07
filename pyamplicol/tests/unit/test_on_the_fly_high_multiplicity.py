# SPDX-License-Identifier: 0BSD
"""Focused contracts for the bounded OTF high-multiplicity driver."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from tools.developer import on_the_fly_high_multiplicity as study
from tools.performance_report.runner import SelectorContract
from tools.performance_report.selector_policy import fixed_selector_helicity


def _args(
    tmp_path: Path,
    n: int,
    workload: str,
    *,
    process_id: int = 7,
) -> argparse.Namespace:
    has_amplicol = workload == "selected" and process_id != 14 and n <= 7
    return argparse.Namespace(
        output=tmp_path / "out",
        process_id=process_id,
        multiplicity=n,
        workload=workload,
        prepared_model=None,
        candidate_artifact=None,
        selected_report=None,
        reference_repo_root=tmp_path if has_amplicol else None,
        reference_profile="reference" if has_amplicol else None,
        target_runtime=0.1,
        worker=False,
    )


def test_exact_cases_helicities_and_cli_routes(tmp_path: Path) -> None:
    assert study.SUPPORTED_MULTIPLICITIES == (5, 6, 7, 8, 9)
    assert study.SUPPORTED_PROCESS_MULTIPLICITIES == {
        7: (5, 6, 7, 8, 9),
        8: (5, 6, 7, 8, 9),
        11: (5, 6, 7),
        13: (5, 6, 7),
        14: (6, 7),
        15: (5, 6, 7),
    }
    bases = {
        7: (1, -1, 6, -6),
        8: (21, 21, 21, 21),
        11: (1, -1, 6, -6, 23, 25),
        13: (1, -1, 2, -2, 3, -3),
        14: (1, -1, 2, -2, 3, -3, 4, -4),
        15: (1, -1, 2, -2, 2, -2),
    }
    for process_id, multiplicities in study.SUPPORTED_PROCESS_MULTIPLICITIES.items():
        for n in multiplicities:
            case = study._case(process_id, n)
            base = bases[process_id]
            assert case.pdgs == (
                *base,
                *(21 for _ in range(n - (len(base) - 2))),
            )
            assert case.process_id == f"otf_p{process_id}_{case.process_key}_n{n}"
            assert fixed_selector_helicity(case.pdgs) == tuple(
                0 if abs(pdg) == 25 else (-1 if index % 2 else 1)
                for index, pdg in enumerate(case.pdgs, start=1)
            )
    assert study._case(7, 5).process == "d d~ > t t~ g g g"
    assert study._case(8, 5).process == "g g > g g g g g"
    assert study._case(11, 5).process == "d d~ > t t~ z h g"
    assert study._case(13, 5).process == "d d~ > u u~ s s~ g"
    assert study._case(14, 6).process == "d d~ > u u~ s s~ c c~"
    assert study._case(15, 5).process == "d d~ > u u~ u u~ g"
    for process_id, n in ((11, 8), (13, 8), (14, 5), (14, 8), (15, 8)):
        with pytest.raises(study.StudyError, match="planned high-multiplicity"):
            study._case(process_id, n)
    assert study._authority_kind(study._case(7, 5), "selected") == "amplicol"
    assert study._authority_kind(study._case(14, 6), "selected") == "recurrence"
    assert study._authority_kind(study._case(8, 9), "selected") == "otf-only"
    assert study._authority_kind(study._case(7, 5), "all-flow") == "otf-only"
    assert (
        study._parser()
        .parse_args(
            [
                "--output",
                "x",
                "--process-id",
                "7",
                "--multiplicity",
                "5",
                "--workload",
                "selected",
            ]
        )
        .target_runtime
        == 5.0
    )
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    prepared = tmp_path / "model"
    prepared.touch()
    invalid = (
        _args(tmp_path, 5, "selected"),
        _args(tmp_path, 8, "all-flow"),
        argparse.Namespace(
            **{**vars(_args(tmp_path, 5, "all-flow")), "candidate_artifact": candidate}
        ),
        argparse.Namespace(
            **{
                **vars(_args(tmp_path, 8, "all-flow")),
                "candidate_artifact": candidate,
                "prepared_model": prepared,
            }
        ),
        argparse.Namespace(
            **{
                **vars(_args(tmp_path, 8, "selected")),
                "prepared_model": prepared,
                "reference_repo_root": tmp_path,
                "reference_profile": "reference",
            }
        ),
    )
    for args in invalid:
        with pytest.raises(study.StudyError):
            study._validate_arguments(args)
    with pytest.raises(SystemExit):
        study._parser().parse_args(
            ["--output", "x", "--multiplicity", "5", "--recurrence-artifact", "x"]
        )

    selected = _args(tmp_path, 5, "selected")
    selected.prepared_model = prepared
    study._validate_arguments(selected)
    recurrence = _args(tmp_path, 6, "selected", process_id=14)
    recurrence.prepared_model = prepared
    study._validate_arguments(recurrence)
    otf_only = _args(tmp_path, 8, "selected")
    otf_only.prepared_model = prepared
    study._validate_arguments(otf_only)
    all_flow = _args(tmp_path, 5, "all-flow")
    all_flow.candidate_artifact = candidate
    all_flow.selected_report = prepared
    study._validate_arguments(all_flow)


def test_timing_batch_repeats_only_seed101_point() -> None:
    assert study.SEEDS == (101, 211, 307, 401, 503, 607, 709, 811)
    points = tuple(((float(index), 0.0, 0.0, 1.0),) for index in range(8))
    timing = study._timing_points(points)
    assert len(timing) == 128
    assert all(point == points[0] for point in timing)
    assert points[1] not in timing


def test_selector_derivation_and_compact_path_never_open_physics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = study._case(14, 6)
    states = fixed_selector_helicity(case.pdgs)
    word = study._colored_source_labels(case)
    contract = SelectorContract(
        ("flow:" + ",".join(map(str, word)),),
        (word,),
        ("h:" + ",".join(f"{state:+d}" for state in states),),
        tuple(enumerate(states, start=1)),
        "a" * 64,
    )
    validate = mock.Mock()
    derive = mock.Mock(return_value=contract)
    monkeypatch.setattr(study, "validate_selector_contract", validate)
    monkeypatch.setattr(study, "derive_selector_contract", derive)
    recurrence = SimpleNamespace(execution_mode="recurrence")
    selector, returned, payload = study._recurrence_authority_contract(
        case, recurrence, ()
    )
    assert selector.flow_word == word and selector.helicities == states
    assert returned is contract and payload == contract.as_dict()
    derive.assert_called_once_with(recurrence, ())
    validate.assert_called_once_with(recurrence, contract, ())

    case = study._case(8, 8)
    flow_id = "flow:1,2,3,4,5,6,7,8,9,10"
    points = (((1.0, 0.0, 0.0, 1.0),),)

    class Backend:
        @property
        def physics(self) -> object:
            raise AssertionError("dense physics opened")

        def _on_the_fly_benchmark_context(self, requested: object) -> dict[str, object]:
            assert requested == ("1",)
            return {
                "process_id": case.process_id,
                "process_expression": case.process,
                "color_accuracy": "lc",
                "helicity_count": 1,
                "color_count": 1,
                "selected_color_ids": [flow_id],
            }

    selector, compact_contract, context = study._selector_from_compact_ordinal_one(
        SimpleNamespace(_backend=Backend()), case, points
    )
    assert selector.flow_word == tuple(range(1, 11))
    assert compact_contract.selected_color_flow_ids == (flow_id,)
    assert context["requested_color_ids"] == ["1"]
    assert context["selector_source"] == "compact-seed-one-based-color-ordinal-1"


@pytest.mark.parametrize(
    ("process_id", "word"),
    (
        (7, (2, 5, 6, 7, 4, 3, 1)),
        (8, (1, 2, 3, 4, 5, 6, 7)),
        (11, (2, 7, 4, 3, 1)),
        (13, (2, 7, 1, 3, 4, 5, 6)),
        (15, (2, 7, 1, 3, 4, 5, 6)),
    ),
)
def test_amplicol_current_is_portably_loaded_and_only_digest_rebound(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    process_id: int,
    word: tuple[int, ...],
) -> None:
    case = study._case(process_id, 5)
    states = fixed_selector_helicity(case.pdgs)
    points = tuple(((float(index), 0.0, 0.0, 1.0),) for index in range(8))
    singleton_digest = study.point_digest((points[0],))
    active = study.ReportSourceIdentity("b" * 40, "d" * 40, ())
    stored = SelectorContract(
        ("flow:" + ",".join(map(str, word)),),
        (word,),
        ("h:" + ",".join(f"{state:+d}" for state in states),),
        tuple(enumerate(states, start=1)),
        singleton_digest,
    )
    docs = tmp_path / "docs" / "performance_reports" / "profile"
    paths = SimpleNamespace(
        repo_root=tmp_path,
        docs_dir=docs,
        artifact_root=docs / "campaign_artifacts",
        coordination_root=docs / "campaign_artifacts" / "coordination",
    )
    for path in (paths.docs_dir, paths.artifact_root, paths.coordination_root):
        path.mkdir(parents=True, exist_ok=True)
    attempt = tmp_path / "attempt"
    legacy_artifact = attempt / "artifact"
    library = legacy_artifact / "selected-flow-generated-library"
    (library / "Library").mkdir(parents=True)
    (library / "processes.txt").touch()
    (library / "libamp1.so").touch()
    executable = library / "amplicol_library_benchmark"
    executable.touch()
    executable.chmod(0o755)
    expected_cell = study._reference_cell(case, "selected")
    result = {
        "status": "ok",
        "selector_contract": stored.as_dict(),
        "matrix_element": 7.5,
        "generation_seconds": 3.0,
        "wall_seconds_per_point": 2.0e-6,
        "execution_seconds_per_point": 1.5e-6,
        "relative_standard_error": 0.0 if process_id == 8 else 0.01,
        "sample_count": 5,
        "artifact": {
            "path": str(legacy_artifact),
            "process_row": "group:1:integral:1",
        },
        "provenance": {
            "revision": study.PINNED_REFERENCE_REVISION,
            "report_source_revision": "b" * 40,
            "report_measured_source_revision": "b" * 40,
            "report_source_tree": "d" * 40,
            "report_measured_source_tree": "d" * 40,
            "report_source_clean": True,
            "generation_timing_is_workload_specific": True,
            "manual_campaign": {
                "batch_size": 128,
                "warmup_runs": 2,
                "minimum_samples": 5,
                "target_runtime_seconds": 0.1,
                "cell_identity": {
                    "accuracy": "lc",
                    "backend": "fortran",
                    "cell_id": expected_cell.cell_id,
                    "dataset_id": "reference_amplicol_lc",
                    "execution_mode": "amplicol",
                    "model": None,
                    "n_final": case.multiplicity,
                    "process": case.process,
                    "process_key": case.process_key,
                    "variant": None,
                    "workload": "selected-flow",
                },
            },
        },
    }
    current = SimpleNamespace(
        cell_id=expected_cell.cell_id,
        result=result,
        attempt_id="attempt",
        manifest_sha256="c" * 64,
        result_path=attempt / "result.json",
    )
    seen: dict[str, object] = {}

    def artifact_store(**kwargs: object) -> object:
        seen.update(kwargs)
        return SimpleNamespace(load_current=lambda cell_id: current)

    def report_paths(repo_root: Path, **kwargs: object) -> object:
        assert repo_root == tmp_path.resolve()
        assert kwargs == {
            "docs_dir": docs,
            "artifact_root": docs / "campaign_artifacts",
            "coordination_root": docs / "campaign_artifacts" / "coordination",
        }
        return paths

    validate = mock.Mock()
    monkeypatch.setattr(study.ReportPaths, "from_repo", report_paths)
    monkeypatch.setattr(study, "ArtifactStore", artifact_store)
    monkeypatch.setattr(study, "validate_measurement", validate)
    reference = study._load_amplicol_reference(
        tmp_path, "profile", case, "selected", points, 0.1, active
    )

    assert seen["current_publication_paths"] is paths
    validate.assert_called_once_with(result, expected_cell=expected_cell)
    assert reference.selector.flow_word == word
    assert reference.matrix_element == 7.5
    assert reference.library_path == library.resolve()
    assert reference.process_row == (1, 1)
    assert reference.lineage["stored_selector_point_digest"] == singleton_digest
    assert reference.timing["relative_standard_error"] == (
        0.0 if process_id == 8 else 0.01
    )
    assert reference.selector_contract == {
        **stored.as_dict(),
        "point_digest": study.point_digest(points),
    }
    if process_id == 7:
        outside = tmp_path / "outside-current-attempt"
        outside.mkdir()
        result["artifact"]["path"] = str(outside)
        with pytest.raises(study.StudyError, match="outside its current attempt"):
            study._load_amplicol_reference(
                tmp_path, "profile", case, "selected", points, 0.1, active
            )


def test_process7_rejects_process8_amplicol_current(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    case7 = study._case(7, 5)
    case8 = study._case(8, 5)
    cell8 = study._reference_cell(case8, "selected")
    paths = SimpleNamespace(
        repo_root=tmp_path,
        docs_dir=tmp_path / "docs",
        artifact_root=tmp_path / "artifacts",
        coordination_root=tmp_path / "coordination",
    )
    for path in (paths.docs_dir, paths.artifact_root, paths.coordination_root):
        path.mkdir()
    wrong = SimpleNamespace(cell_id=cell8.cell_id, result={"status": "ok"})
    monkeypatch.setattr(study.ReportPaths, "from_repo", lambda *_args, **_kwargs: paths)
    monkeypatch.setattr(
        study,
        "ArtifactStore",
        lambda **_kwargs: SimpleNamespace(load_current=lambda _cell_id: wrong),
    )
    monkeypatch.setattr(study, "validate_measurement", lambda *_args, **_kwargs: None)
    with pytest.raises(study.StudyError, match="workload identity"):
        study._load_amplicol_reference(
            tmp_path,
            "profile",
            case7,
            "selected",
            (((1.0, 0.0, 0.0, 1.0),),),
            0.1,
            study.ReportSourceIdentity("a" * 40, "b" * 40, ()),
        )


def test_amplicol_matrix_element_is_an_explicit_singleton_authority() -> None:
    case = study._case(7, 5)
    states = fixed_selector_helicity(case.pdgs)
    word = tuple(range(1, len(case.pdgs) + 1))
    selector = study.Selector(
        word,
        "flow:" + ",".join(map(str, word)),
        states,
        "h:" + ",".join(f"{state:+d}" for state in states),
    )
    points = tuple(((float(index), 0.0, 0.0, 1.0),) for index in range(8))
    contract = SelectorContract(
        (selector.flow_id,),
        (selector.flow_word,),
        (selector.helicity_id,),
        tuple(enumerate(selector.helicities, start=1)),
        study.point_digest(points),
    )
    reference = study.AmplicolReference(
        selector=selector,
        contract=contract,
        selector_contract=contract.as_dict(),
        matrix_element=3.0,
        timing={},
        lineage={},
    )
    check = study._amplicol_singleton_check(
        study.Evaluation((3.0, *(9.0 for _ in range(7))), None),
        reference,
        points,
    )
    assert check["seed"] == 101
    assert check["point_count"] == 1
    assert check["stored_amplicol_matrix_element"] == 3.0
    assert check["comparison"]["checks"] == 1
    within_amplicol_tolerance = study._amplicol_singleton_check(
        study.Evaluation((3.0 * (1.0 + 0.5e-8), *(9.0 for _ in range(7))), None),
        reference,
        points,
    )
    assert within_amplicol_tolerance["comparison"]["checks"] == 1
    with pytest.raises(study.StudyError, match="disagrees"):
        study._amplicol_singleton_check(
            study.Evaluation((3.0 * (1.0 + 2.0e-8), *(9.0 for _ in range(7))), None),
            reference,
            points,
        )


def test_amplicol_replays_all_eight_points_from_exact_recorded_row(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    case = study._case(8, 5)
    points = tuple(
        tuple(
            (float(index) + leg / 100.0, 0.0, 0.0, 1.0) for leg in range(len(case.pdgs))
        )
        for index in range(1, 9)
    )
    word = study._colored_source_labels(case)
    states = fixed_selector_helicity(case.pdgs)
    selector = study.Selector(
        word,
        "flow:" + ",".join(map(str, word)),
        states,
        "h:" + ",".join(f"{state:+d}" for state in states),
    )
    contract = SelectorContract(
        (selector.flow_id,),
        (selector.flow_word,),
        (selector.helicity_id,),
        tuple(enumerate(selector.helicities, start=1)),
        study.point_digest(points),
    )
    library = tmp_path / "selected-flow-generated-library"
    library.mkdir()
    entry = study.legacy_amplicol.ProcessEntry(3, 7, case.pdgs, word)
    reference = study.AmplicolReference(
        selector=selector,
        contract=contract,
        selector_contract=contract.as_dict(),
        matrix_element=1.0,
        timing={},
        lineage={},
        library_path=library,
        process_row=(3, 7),
    )
    monkeypatch.setattr(
        study.legacy_amplicol, "parse_process_file", lambda _path: (entry,)
    )
    monkeypatch.setattr(
        study.legacy_amplicol,
        "source_mapped_color_order",
        lambda entry, **_kwargs: entry.color_order,
    )
    monkeypatch.setenv("LD_LIBRARY_PATH", "/prior-ld")
    monkeypatch.setenv("DYLD_LIBRARY_PATH", "/prior-dyld")
    observed_environments: list[tuple[str | None, str | None]] = []

    def replay_probe(_library: Path, **kwargs: object) -> object:
        observed_environments.append(
            (os.environ.get("LD_LIBRARY_PATH"), os.environ.get("DYLD_LIBRARY_PATH"))
        )
        return SimpleNamespace(
            value=kwargs["momenta"][0][0],
            process_pdgs=case.pdgs,
            color_order=word,
        )

    probe = mock.Mock(side_effect=replay_probe)
    monkeypatch.setattr(study.legacy_amplicol, "run_selected_flow_library_probe", probe)

    values, evidence = study._replay_amplicol(reference, case, points)

    assert values == tuple(float(index) for index in range(1, 9))
    assert evidence["point_count"] == evidence["subprocess_launches"] == 8
    assert evidence["seed101_replay_vs_stored_matrix_element"]["checks"] == 1
    assert probe.call_count == 8
    root = str(library.resolve())
    assert (
        observed_environments
        == [(f"{root}{os.pathsep}/prior-ld", f"{root}{os.pathsep}/prior-dyld")] * 8
    )
    assert os.environ["LD_LIBRARY_PATH"] == "/prior-ld"
    assert os.environ["DYLD_LIBRARY_PATH"] == "/prior-dyld"
    for point, call in zip(points, probe.call_args_list, strict=True):
        assert call.args == (library,)
        assert call.kwargs["entry"] is entry
        assert call.kwargs["source_pdgs"] == case.pdgs
        assert call.kwargs["momenta"] == point
        assert call.kwargs["helicities"] is None
        assert call.kwargs["points"] == 1

    monkeypatch.setattr(
        study.legacy_amplicol,
        "run_selected_flow_library_probe",
        lambda _library, **kwargs: SimpleNamespace(
            value=kwargs["momenta"][0][0],
            process_pdgs=case.pdgs,
            color_order=tuple(reversed(word)),
        ),
    )
    with pytest.raises(study.StudyError, match="different semantic selector flow"):
        study._replay_amplicol(reference, case, points)


def test_process7_compact_context_receives_exact_amplicol_semantic_flow() -> None:
    case = study._case(7, 5)
    word = (2, 5, 6, 7, 4, 3, 1)
    flow_id = "flow:" + ",".join(map(str, word))
    selector = study.Selector(
        word,
        flow_id,
        fixed_selector_helicity(case.pdgs),
        "h:-1,+1,-1,+1,-1,+1,-1",
    )

    class Backend:
        @property
        def physics(self) -> object:
            raise AssertionError("dense physics opened")

        def _on_the_fly_benchmark_context(self, requested: object) -> dict[str, object]:
            assert requested == (flow_id, "1")
            return {
                "process_id": case.process_id,
                "process_expression": case.process,
                "color_accuracy": "lc",
                "helicity_count": 2 ** len(case.pdgs),
                "color_count": 10,
                "selected_color_ids": [flow_id, flow_id],
            }

    context = study._cross_check_compact_selector(
        SimpleNamespace(_backend=Backend()),
        case,
        selector,
        require_ordinal_one=True,
    )
    assert context["requested_color_ids"] == [flow_id, "1"]
    assert context["selected_color_ids"] == [flow_id, flow_id]
    assert context["ordinal_one_matches_authority"] is True


def test_reference_source_must_match_generated_candidate() -> None:
    revision = "a" * 40
    active = study.ReportSourceIdentity(revision, "c" * 40, ())
    assert study._candidate_source_binding(active, {"source_revision": revision}) == {
        "status": "passed",
        "policy": "exact-source-revision",
        "source_revision": revision,
        "source_tree": "c" * 40,
    }
    with pytest.raises(study.StudyError, match="active source"):
        study._candidate_source_binding(active, {"source_revision": "b" * 40})


def test_stale_reused_candidate_rejects_before_report_or_any_comparator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path, 5, "all-flow")
    candidate_path = tmp_path / "candidate"
    candidate_path.mkdir()
    selected_report = tmp_path / "selected-report.json"
    selected_report.touch()
    args.candidate_artifact = candidate_path
    args.selected_report = selected_report
    forbidden_recurrence = mock.Mock(
        side_effect=AssertionError("recurrence generation must not start")
    )
    forbidden_reference = mock.Mock(
        side_effect=AssertionError("AmpliCol reference must not load")
    )
    forbidden_report = mock.Mock(
        side_effect=AssertionError("selected report validation must not start")
    )
    monkeypatch.setattr(
        study,
        "_active_report_source",
        lambda: study.ReportSourceIdentity("a" * 40, "c" * 40, ()),
    )
    monkeypatch.setattr(study, "_load_amplicol_reference", forbidden_reference)
    monkeypatch.setattr(study.Runtime, "load", lambda *_, **__: SimpleNamespace())
    monkeypatch.setattr(
        study,
        "_artifact_identity",
        lambda *_: {
            "producer_identity": {
                "source_revision": "b" * 40,
                "native_build_inputs_sha256": "d" * 64,
            },
            "model_identity": {"content_sha256": "e" * 64},
        },
    )
    monkeypatch.setattr(study, "_selected_report_lineage", forbidden_report)
    monkeypatch.setattr(study, "_generate_recurrence", forbidden_recurrence)

    with pytest.raises(study.StudyError, match=r"generated candidate.*active source"):
        study._run_worker(args)
    forbidden_report.assert_not_called()
    forbidden_reference.assert_not_called()
    forbidden_recurrence.assert_not_called()


def test_ineligible_active_source_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        study,
        "require_eligible_report_source",
        mock.Mock(side_effect=study.ReportSourceIdentityError("dirty source")),
    )
    with pytest.raises(study.StudyError, match=r"active report source.*dirty source"):
        study._active_report_source()


def _reference_provenance(
    case: study.Case, workload: str, target: float = 0.1
) -> dict[str, object]:
    cell = study._reference_cell(case, workload)
    workload_id = "selected-flow" if workload == "selected" else "all-flow"
    return {
        "revision": study.PINNED_REFERENCE_REVISION,
        "generation_timing_is_workload_specific": True,
        "manual_campaign": {
            "batch_size": 128,
            "warmup_runs": 2,
            "minimum_samples": 5,
            "target_runtime_seconds": target,
            "cell_identity": {
                "accuracy": "lc",
                "backend": "fortran",
                "cell_id": cell.cell_id,
                "dataset_id": "reference_amplicol_lc",
                "execution_mode": "amplicol",
                "model": None,
                "n_final": case.multiplicity,
                "process": case.process,
                "process_key": case.process_key,
                "variant": None,
                "workload": workload_id,
            },
        },
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("batch_size", 64, "batch size"),
        ("warmup_runs", 1, "warm-up count"),
        ("minimum_samples", 4, "sample minimum"),
        ("target_runtime_seconds", 0.2, "target runtime"),
    ),
)
def test_reference_timing_contract_rejects_wrong_campaign_settings(
    field: str, value: object, message: str
) -> None:
    case = study._case(7, 5)
    provenance = _reference_provenance(case, "selected")
    manual = provenance["manual_campaign"]
    assert isinstance(manual, dict)
    manual[field] = value
    with pytest.raises(study.StudyError, match=message):
        study._reference_timing_contract(provenance, case, "selected", 0.1)


@pytest.mark.parametrize("value", (True, 4, 5.0, None))
def test_reference_rejects_insufficient_actual_sample_count(value: object) -> None:
    with pytest.raises(study.StudyError, match="actual samples"):
        study._required_reference_sample_count({"sample_count": value})
    assert study._required_reference_sample_count({"sample_count": 5}) == 5


def test_reference_selector_rejects_non_seed101_digest() -> None:
    case = study._case(7, 5)
    points = tuple(((float(index), 0.0, 0.0, 1.0),) for index in range(8))
    states = fixed_selector_helicity(case.pdgs)
    wrong = SelectorContract(
        ("flow:2,5,6,7,4,3,1",),
        ((2, 5, 6, 7, 4, 3, 1),),
        ("h:" + ",".join(f"{state:+d}" for state in states),),
        tuple(enumerate(states, start=1)),
        "a" * 64,
    )
    with pytest.raises(study.StudyError, match="seed-101 singleton"):
        study._rebind_reference_selector(wrong.as_dict(), case, points)


def test_generation_reporting_separates_selected_ratios_from_all_flow_absolute() -> (
    None
):
    selected, selected_ratios = study._generation_reporting(
        "selected",
        generation_seconds=2.0,
        warmup_seconds=0.5,
        amplicol_generation_seconds=1.0,
        source="this-selected-run",
    )
    assert selected["notation"] == "([xG] x(G+W))"
    assert selected["generation_only_over_amplicol"] == 2.0
    assert selected["generation_plus_warmup_over_amplicol"] == 2.5
    assert selected_ratios == {
        "generation_only": 2.0,
        "generation_plus_warmup": 2.5,
    }

    all_flow, all_flow_ratios = study._generation_reporting(
        "all-flow",
        generation_seconds=2.0,
        warmup_seconds=0.75,
        amplicol_generation_seconds=None,
        source="reused-selected-artifact-generation",
    )
    assert all_flow == {
        "notation": "[G] G+W",
        "reporting": "absolute-on-the-fly-seconds",
        "source": "reused-selected-artifact-generation",
        "warmup_source": "this-all-flow-cold-batch128-run",
        "on_the_fly_generation_seconds": 2.0,
        "on_the_fly_warmup_seconds": 0.75,
        "on_the_fly_generation_plus_warmup_seconds": 2.75,
    }
    assert all_flow_ratios == {}


def _active(queries: int, destinations: int) -> dict[str, object]:
    result = {
        "basis": "shared-query-family-union-v1",
        "scope": "active-family-union",
        "query_count": queries,
        "union_unique_current_count": 2,
        "union_unique_current_component_count": 3,
        "union_amplitude_destination_count": destinations,
    }
    for role in study.OPERATION_ROLES:
        result[f"union_{role}_rows"] = 2
        result[f"union_{role}_executor_call_groups"] = 1
    return result


def _state(families: int, active: dict[str, object] | None = None) -> dict[str, object]:
    warm = families > 0
    return {
        "process_preparation_count": int(warm),
        "retained_family_count": families,
        "pending_family_count": 0,
        "retained_selection_count": families,
        "retained_request_count": families * 2,
        "retained_amplitude_destination_count": families,
        "retained_executor_handle_count": families,
        "retained_query_local_trace_count": 0,
        "retained_embedded_lookup_key_count": 0,
        "semantic_executor_binding_count": families * 3,
        "active_family_union_census": active,
    }


@pytest.mark.parametrize(
    ("workload", "expected_kwargs"),
    (
        ("selected", {"color_flows": ("flow:1",)}),
        ("all-flow", {"helicities": ("h:-1",)}),
    ),
)
def test_requested_workload_cold_warmup_is_batch128_and_clears(
    monkeypatch: pytest.MonkeyPatch,
    workload: str,
    expected_kwargs: dict[str, tuple[str, ...]],
) -> None:
    cold = _state(0)
    warm = _state(1, _active(2, 1))
    states = iter((cold, warm, cold))
    monkeypatch.setattr(study, "_census", lambda *_: copy.deepcopy(next(states)))
    monkeypatch.setattr(
        study.time,
        "perf_counter_ns",
        mock.Mock(side_effect=(1_000_000, 4_000_000)),
    )
    points = tuple(((float(index), 0.0, 0.0, 1.0),) for index in range(8))
    captured: dict[str, object] = {}

    def evaluate(batch: object, **kwargs: object) -> tuple[complex, ...]:
        captured["batch"] = batch
        captured["kwargs"] = kwargs
        return (1.0 + 0.0j,) * 128

    runtime = SimpleNamespace(evaluate=evaluate, clear=mock.Mock())
    evidence = study._requested_workload_cold_warmup(
        runtime,
        study._case(7, 5),
        study.Selector((1,), "flow:1", (-1,), "h:-1"),
        points,
        workload,
        ratio_eligible=workload == "selected",
    )

    assert captured["kwargs"] == expected_kwargs
    assert captured["batch"] == (points[0],) * 128
    assert evidence["requested_workload"] == workload
    assert evidence["batch_size"] == evidence["point_count"] == 128
    assert evidence["seed"] == 101
    assert evidence["singleton_seed_point_digest"] == study.point_digest((points[0],))
    assert evidence["elapsed_nanoseconds"] == 3_000_000
    assert evidence["clear_restored_cold_state"] is True
    assert "Runtime.load" in evidence["excluded_from_elapsed"]
    assert "artifact generation" in evidence["excluded_from_elapsed"]
    assert evidence["ratio_eligible"] is (workload == "selected")
    runtime.clear.assert_called_once_with()


def test_a_b_repeat_revisit_clear_rebuild_and_census_invariants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cold, active_a, active_b = _state(0), _active(2, 1), _active(4, 2)
    state_a, state_b = _state(1, active_a), _state(2, active_b)
    revisit = copy.deepcopy(state_b)
    revisit["active_family_union_census"] = active_a
    values = iter((cold, state_a, state_b, state_b, revisit, cold, state_b))
    monkeypatch.setattr(study, "_census", lambda *_: copy.deepcopy(next(values)))
    monkeypatch.setattr(
        study, "_evaluate", lambda *_, **__: study.Evaluation((1.0 + 0.0j,), None)
    )
    monkeypatch.setattr(study, "_compare", lambda *_: {})
    runtime = SimpleNamespace(
        clear=mock.Mock(), evaluate=mock.Mock(return_value=(1.0 + 0.0j,))
    )
    lifecycle = study._lifecycle(
        runtime,
        study._case(7, 5),
        study.Selector((1,), "flow:1", (-1,), "h:-1"),
        ((((1.0, 0.0, 0.0, 1.0),),)),
        "all-flow",
        False,
    )[2]
    assert lifecycle["sequence"] == "A,B,B,A; clear; A,B"
    assert lifecycle["after_rebuild"] == state_b
    assert "cold_first_evaluation_timing" not in lifecycle
    runtime.evaluate.assert_called_once_with(
        ((((1.0, 0.0, 0.0, 1.0),),)), color_flows=("flow:1",)
    )
    runtime.clear.assert_called_once_with()
    for field, value in (
        ("retained_amplitude_destination_count", 0),
        ("retained_request_count", 0),
    ):
        broken = copy.deepcopy(state_a)
        broken[field] = value
        with pytest.raises(study.StudyError):
            study._assert_family_state(
                broken, "A", families=1, selections=1, handles=1, minimum_bindings=1
            )
    broken = copy.deepcopy(state_a)
    broken["active_family_union_census"]["union_source_executor_call_groups"] = 3
    with pytest.raises(study.StudyError):
        study._assert_family_state(
            broken, "A", families=1, selections=1, handles=1, minimum_bindings=1
        )


def test_selected_a_c_repeat_revisit_and_rebuild(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cold, active_a, active_c = _state(0), _active(2, 1), _active(1, 1)
    state_a, state_c = _state(1, active_a), _state(2, active_c)
    revisit = copy.deepcopy(state_c)
    revisit["active_family_union_census"] = active_a
    values = iter(
        (cold, state_a, state_c, state_c, revisit, cold, state_a, state_c, revisit)
    )
    calls: list[str] = []
    monkeypatch.setattr(study, "_census", lambda *_: copy.deepcopy(next(values)))

    def evaluate(
        _runtime: object,
        _points: object,
        workload: str,
        _selector: object,
        *,
        resolved: bool,
        _precomputed_total: tuple[object, ...] | None = None,
    ) -> study.Evaluation:
        assert resolved is False
        if not calls:
            assert _precomputed_total == (1.0 + 0.0j,)
        calls.append(workload)
        return study.Evaluation((1.0 + 0.0j,), None)

    monkeypatch.setattr(study, "_evaluate", evaluate)
    monkeypatch.setattr(study, "_compare", lambda *_: {})
    runtime = SimpleNamespace(
        clear=mock.Mock(), evaluate=mock.Mock(return_value=(1.0 + 0.0j,))
    )
    lifecycle = study._lifecycle(
        runtime,
        study._case(7, 8),
        study.Selector((1,), "flow:1", (-1,), "h:-1"),
        ((((1.0, 0.0, 0.0, 1.0),),)),
        "selected",
        False,
    )[2]

    assert calls == [
        "selected",
        "exact",
        "exact",
        "selected",
        "selected",
        "exact",
        "selected",
    ]
    assert lifecycle["sequence"] == "A,C,C,A; clear; A,C,A"
    assert lifecycle["requested_family"] == state_c
    assert lifecycle["after_rebuild"] == revisit
    assert "cold_first_evaluation_timing" not in lifecycle
    runtime.clear.assert_called_once_with()


def _worker_selector(
    case: study.Case, points: tuple[object, ...]
) -> tuple[study.Selector, SelectorContract]:
    word = study._colored_source_labels(case)
    states = fixed_selector_helicity(case.pdgs)
    selector = study.Selector(
        word,
        "flow:" + ",".join(map(str, word)),
        states,
        "h:" + ",".join(f"{state:+d}" for state in states),
    )
    return selector, SelectorContract(
        (selector.flow_id,),
        (selector.flow_word,),
        (selector.helicity_id,),
        tuple(enumerate(selector.helicities, start=1)),
        study.point_digest(points),
    )


def _patch_selected_worker_basics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    case: study.Case,
    points: tuple[object, ...],
    candidate: object,
) -> tuple[study.ReportSourceIdentity, dict[str, object]]:
    source = study.ReportSourceIdentity("b" * 40, "e" * 40, ())
    identity = {
        "path": str(tmp_path / "out" / "candidate-artifact"),
        "artifact_id": "a" * 64,
        "producer_identity": {
            "source_revision": source.revision,
            "native_build_inputs_sha256": "c" * 64,
        },
        "model_identity": {"content_sha256": "d" * 64},
    }
    monkeypatch.setattr(study, "_points", lambda _case: points)
    monkeypatch.setattr(study, "_active_report_source", lambda: source)
    monkeypatch.setattr(
        study.ModelSource,
        "from_path",
        lambda _path: SimpleNamespace(compile=lambda: object()),
    )
    monkeypatch.setattr(
        study.Runtime,
        "load",
        lambda path, **_kwargs: (
            candidate
            if "recurrence-authority" not in str(path)
            else SimpleNamespace(execution_mode="recurrence")
        ),
    )
    monkeypatch.setattr(study, "_artifact_identity", lambda *_args: dict(identity))
    monkeypatch.setattr(
        study,
        "_requested_workload_cold_warmup",
        lambda *_args, **_kwargs: {"seconds": 0.5},
    )
    monkeypatch.setattr(study, "_census", lambda *_args: {})
    return source, identity


@pytest.mark.parametrize("process_id", (7, 8, 11, 13, 15))
def test_amplicol_cells_forbid_recurrence_and_compiled_and_replay_eight_points(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, process_id: int
) -> None:
    args = _args(tmp_path, 5, "selected", process_id=process_id)
    args.prepared_model = tmp_path / "model"
    args.prepared_model.touch()
    case = study._case(process_id, 5)
    points = tuple(((float(index), 0.0, 0.0, 1.0),) for index in range(8))
    selector, contract = _worker_selector(case, points)
    candidate = SimpleNamespace()
    _patch_selected_worker_basics(monkeypatch, tmp_path, case, points, candidate)
    reference = study.AmplicolReference(
        selector=selector,
        contract=contract,
        selector_contract=contract.as_dict(),
        matrix_element=3.0,
        timing={
            "generation_seconds": 1.0,
            "wall_seconds_per_point": 0.5e-6,
            "evaluator_seconds_per_point": 0.4e-6,
        },
        lineage={"status": "authenticated"},
    )
    monkeypatch.setattr(study, "_load_amplicol_reference", lambda *_args: reference)
    monkeypatch.setattr(
        study,
        "_generate_otf",
        lambda *_args: {
            "seconds": 2.0,
            "seed_binding_calls": 1,
            "seed_count": 1,
            "materialized_lane_calls": 0,
        },
    )
    forbidden_recurrence = mock.Mock(
        side_effect=AssertionError("AmpliCol cell entered recurrence")
    )
    monkeypatch.setattr(study, "_generate_recurrence", forbidden_recurrence)
    compact = mock.Mock(
        return_value={
            "selected_color_ids": [selector.flow_id, selector.flow_id],
            "ordinal_one_matches_authority": True,
        }
    )
    monkeypatch.setattr(study, "_cross_check_compact_selector", compact)
    evaluation = study.Evaluation((3.0, *(4.0 for _ in range(7))), None)
    lifecycle = mock.Mock(return_value=(evaluation, evaluation, {"after_rebuild": {}}))
    monkeypatch.setattr(study, "_lifecycle", lifecycle)
    replay = mock.Mock(
        return_value=(
            evaluation.total,
            {"status": "passed", "point_count": 8, "subprocess_launches": 8},
        )
    )
    monkeypatch.setattr(study, "_replay_amplicol", replay)
    monkeypatch.setattr(study, "_compare", lambda *_args: {"checks": 8})
    monkeypatch.setattr(
        study,
        "_benchmark",
        lambda *_args: {
            "wall_seconds_per_point": 1.0e-6,
            "evaluator_seconds_per_point": 0.8e-6,
        },
    )

    result = study._run_worker(args)

    assert set(result["generation"]) == {"on_the_fly"}
    assert set(result["artifacts"]) == {"candidate"}
    assert set(result["timings"]) == {"on_the_fly", "amplicol"}
    assert (
        result["correctness"]["otf_seed101_vs_amplicol_matrix_element"]["status"]
        == "passed"
    )
    assert result["correctness"]["otf_before_vs_amplicol_replay"]["checks"] == 8
    assert result["correctness"]["claim_scope"]["external_amplicol_point_count"] == 8
    assert result["forbidden_paths"]["compiled_generation"] == "not-called"
    assert result["forbidden_paths"]["recurrence_generation"] == "not-called"
    compact.assert_called_once_with(candidate, case, selector, require_ordinal_one=True)
    replay.assert_called_once_with(reference, case, points)
    assert lifecycle.call_args.args[-1] is True
    forbidden_recurrence.assert_not_called()
    with pytest.raises(study.StudyError, match="forbids execution mode"):
        study._config("compiled", "topology-replay")


def test_id14_uses_recurrence_only_and_derives_its_selector(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    args = _args(tmp_path, 6, "selected", process_id=14)
    args.prepared_model = tmp_path / "model"
    args.prepared_model.touch()
    case = study._case(14, 6)
    points = tuple(((float(index), 0.0, 0.0, 1.0),) for index in range(8))
    selector, contract = _worker_selector(case, points)
    candidate = SimpleNamespace()
    source, identity = _patch_selected_worker_basics(
        monkeypatch, tmp_path, case, points, candidate
    )
    cold_warmup = mock.Mock(return_value={"seconds": 0.5})
    monkeypatch.setattr(study, "_requested_workload_cold_warmup", cold_warmup)
    recurrence = SimpleNamespace(execution_mode="recurrence")
    monkeypatch.setattr(
        study.Runtime,
        "load",
        lambda path, **_kwargs: (
            recurrence if "recurrence-authority" in str(path) else candidate
        ),
    )
    monkeypatch.setattr(
        study,
        "_artifact_identity",
        lambda path, *_args: {
            **identity,
            "path": str(path),
            "producer_identity": {
                "source_revision": source.revision,
                "native_build_inputs_sha256": "c" * 64,
            },
        },
    )
    forbidden_reference = mock.Mock(
        side_effect=AssertionError("ID14 must not load AmpliCol")
    )
    monkeypatch.setattr(study, "_load_amplicol_reference", forbidden_reference)
    monkeypatch.setattr(
        study,
        "_generate_otf",
        lambda *_args: {
            "seconds": 2.0,
            "seed_binding_calls": 1,
            "seed_count": 1,
            "materialized_lane_calls": 0,
        },
    )
    generate_recurrence = mock.Mock(
        return_value={
            "execution_mode": "recurrence",
            "layout": "topology-replay",
            "seconds": 4.0,
        }
    )
    monkeypatch.setattr(study, "_generate_recurrence", generate_recurrence)
    derive = mock.Mock(return_value=contract)
    validate = mock.Mock()
    monkeypatch.setattr(study, "derive_selector_contract", derive)
    monkeypatch.setattr(study, "validate_selector_contract", validate)
    compact = mock.Mock(
        return_value={
            "selected_color_ids": [selector.flow_id, selector.flow_id],
            "ordinal_one_required": True,
            "ordinal_one_matches_authority": True,
        }
    )
    monkeypatch.setattr(study, "_cross_check_compact_selector", compact)
    evaluation = study.Evaluation((3.0,) * 8, None)
    monkeypatch.setattr(
        study,
        "_lifecycle",
        lambda *_args: (evaluation, evaluation, {"after_rebuild": {}}),
    )
    evaluate = mock.Mock(return_value=evaluation)
    monkeypatch.setattr(study, "_evaluate", evaluate)
    monkeypatch.setattr(study, "_compare", lambda *_args: {"checks": 8})
    monkeypatch.setattr(
        study,
        "_benchmark",
        lambda _runtime, mode, *_args: {
            "wall_seconds_per_point": 1.0e-6 if mode == "on-the-fly" else 0.5e-6,
            "evaluator_seconds_per_point": 0.8e-6 if mode == "on-the-fly" else 0.4e-6,
        },
    )

    result = study._run_worker(args)

    assert result["authority"]["kind"] == "recurrence"
    assert set(result["generation"]) == {"on_the_fly", "recurrence"}
    assert set(result["artifacts"]) == {"candidate", "recurrence_authority"}
    assert set(result["timings"]) == {"on_the_fly", "recurrence"}
    assert result["generation_comparison"]["recurrence_generation_seconds"] == 4.0
    assert result["generation_comparison"]["generation_only_over_recurrence"] == 0.5
    assert (
        result["generation_comparison"]["generation_plus_warmup_over_recurrence"]
        == 0.625
    )
    assert result["descriptive_ratios"]["on_the_fly_over_recurrence"] == {
        "wall_seconds_per_point": 2.0,
        "evaluator_seconds_per_point": 2.0,
        "generation_only": 0.5,
        "generation_plus_warmup": 0.625,
    }
    derive.assert_called_once_with(recurrence, points)
    validate.assert_called_once_with(recurrence, contract, points)
    compact.assert_called_once_with(candidate, case, selector, require_ordinal_one=True)
    assert cold_warmup.call_args.kwargs["ratio_eligible"] is True
    evaluate.assert_called_once_with(
        recurrence, points, "selected", selector, resolved=True
    )
    generate_recurrence.assert_called_once()
    forbidden_reference.assert_not_called()


def test_n8_selected_is_compact_ordinal_one_and_forbids_comparators(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    args = _args(tmp_path, 8, "selected")
    args.prepared_model = tmp_path / "model"
    args.prepared_model.touch()
    case = study._case(8, 8)
    points = tuple(((float(index), 0.0, 0.0, 1.0),) for index in range(8))
    flow_id = "flow:1,2,3,4,5,6,7,8,9,10"

    class Backend:
        @property
        def physics(self) -> object:
            raise AssertionError("n8 dense physics opened")

        def _on_the_fly_benchmark_context(self, requested: object) -> dict[str, object]:
            assert requested == ("1",)
            return {
                "process_id": case.process_id,
                "process_expression": case.process,
                "color_accuracy": "lc",
                "helicity_count": 1024,
                "color_count": 100,
                "selected_color_ids": [flow_id],
            }

    candidate = SimpleNamespace(_backend=Backend())
    _patch_selected_worker_basics(monkeypatch, tmp_path, case, points, candidate)
    forbidden = mock.Mock(side_effect=AssertionError("n8 comparator called"))
    monkeypatch.setattr(study, "_load_amplicol_reference", forbidden)
    monkeypatch.setattr(study, "_generate_recurrence", forbidden)
    monkeypatch.setattr(study, "derive_selector_contract", forbidden)
    monkeypatch.setattr(
        study,
        "_generate_otf",
        lambda *_args: {
            "seconds": 2.0,
            "seed_binding_calls": 1,
            "seed_count": 1,
            "materialized_lane_calls": 0,
        },
    )
    evaluation = study.Evaluation((3.0,) * 8, None)
    lifecycle = mock.Mock(return_value=(evaluation, evaluation, {"after_rebuild": {}}))
    monkeypatch.setattr(study, "_lifecycle", lifecycle)
    monkeypatch.setattr(study, "_compare", lambda *_args: {"checks": 8})
    monkeypatch.setattr(
        study,
        "_benchmark",
        lambda *_args: {
            "wall_seconds_per_point": 1.0e-6,
            "evaluator_seconds_per_point": 0.8e-6,
        },
    )

    result = study._run_worker(args)

    assert result["authority"]["kind"] == "otf-only"
    assert result["selector_contract"]["selected_color_flow_ids"] == [flow_id]
    assert result["compact_selector_context"]["requested_color_ids"] == ["1"]
    assert result["correctness"]["status"] == "not-claimed"
    assert set(result["timings"]) == {"on_the_fly"}
    assert result["descriptive_ratios"] == {}
    assert lifecycle.call_args.args[-1] is False
    assert result["forbidden_paths"]["physics_enumeration"] == "not-called"
    assert result["forbidden_paths"]["resolved_output"] == "not-called"
    forbidden.assert_not_called()


def test_identity_n8_forbidden_route_and_worker_argv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    case, artifact_id = study._case(7, 8), "a" * 64
    manifest = SimpleNamespace(
        root=tmp_path,
        artifact_id=artifact_id,
        processes=(
            dict(
                id=case.process_id,
                expression=case.process,
                external_pdgs=case.pdgs,
                color_accuracy="lc",
            ),
        ),
        producer={"git_revision": "b" * 40, "native_build_inputs_sha256": "c" * 64},
        model={
            "name": "built-in-sm",
            "source_kind": "built-in-sm",
            "content_sha256": "d" * 64,
            "compiled_schema_version": 9,
        },
    )
    metadata = dict(
        execution_mode="on-the-fly",
        process=case.process,
        process_key=case.process_id,
        representative_process=case.process,
        representative_process_key=case.process_id,
        color_accuracy="lc",
        external_pdg_order=case.pdgs,
        external_count=len(case.pdgs),
    )
    runtime = SimpleNamespace(
        artifact_id=artifact_id,
        execution_mode="on-the-fly",
        representative_process_key=case.process_id,
        _backend=SimpleNamespace(
            _runtime=SimpleNamespace(artifact_id=artifact_id), _native_metadata=metadata
        ),
    )
    monkeypatch.setattr(study, "load_manifest", lambda *_, **__: manifest)
    identity = study._artifact_identity(tmp_path, runtime, case, "on-the-fly")
    assert identity["producer_identity"]["source_revision"] == "b" * 40
    mismatched = copy.deepcopy(identity)
    mismatched["model_identity"]["content_sha256"] = "e" * 64
    with pytest.raises(study.StudyError, match="different models"):
        study._common_model_identity({"candidate": identity, "authority": mismatched})
    wrong_report = tmp_path / "wrong-selected.json"
    wrong_report.write_text(json.dumps({"kind": "wrong"}), encoding="utf-8")
    with pytest.raises(study.StudyError, match="wrong study identity"):
        study._selected_report_lineage(
            wrong_report,
            case,
            identity,
            study.ReportSourceIdentity("b" * 40, "e" * 40, ()),
        )

    candidate = tmp_path / "candidate with spaces; no shell"
    candidate.mkdir()
    selected_report = tmp_path / "selected-report.json"
    selected_report.touch()
    args = _args(tmp_path, 8, "all-flow")
    args.candidate_artifact = candidate
    args.selected_report = selected_report
    command = study._worker_command(args, args.output)
    assert isinstance(command, list) and command[0] == sys.executable
    assert str(candidate.resolve()) in command
    assert command[-1] == str(selected_report.resolve())

    selector = study.Selector(
        tuple(range(1, 11)),
        "flow:" + ",".join(map(str, range(1, 11))),
        fixed_selector_helicity(case.pdgs),
        "h:-1,+1,-1,+1,-1,+1,-1,+1,-1,+1",
    )
    contract = SelectorContract(
        (selector.flow_id,),
        (selector.flow_word,),
        (selector.helicity_id,),
        tuple(enumerate(selector.helicities, start=1)),
        study.point_digest(study._points(case)),
    )
    forbidden = mock.Mock(side_effect=AssertionError("forbidden n>=8 route"))
    active_source = study.ReportSourceIdentity("b" * 40, "e" * 40, ())
    candidate_identity = {
        **identity,
        "path": str(candidate.resolve()),
        "artifact_id": artifact_id,
    }
    selected_report.write_text(
        json.dumps(
            {
                "kind": f"{study.KIND}-run",
                "schema_version": study.SCHEMA_VERSION,
                "status": "passed",
                "watchdog": {
                    "passes": True,
                    "limit_bytes": study.WATCHDOG_BYTES,
                },
                "study": {
                    "kind": study.KIND,
                    "schema_version": study.SCHEMA_VERSION,
                    "status": "passed",
                    "workload": "selected",
                    "process": study.dataclass_payload(case),
                    "authority": {"kind": study.AUTHORITY_OTF_ONLY},
                    "correctness": {
                        "status": "not-claimed",
                        "pre_post_clear_self_consistency": {"total": {"checks": 8}},
                    },
                    "compact_selector_context": {
                        "selector_source": "compact-seed-one-based-color-ordinal-1",
                        "requested_color_ids": ["1"],
                        "selected_color_ids": [selector.flow_id],
                        "resolved_semantic_flow_id": selector.flow_id,
                    },
                    "producer_identity": identity["producer_identity"],
                    "model_identity": identity["model_identity"],
                    "active_source": active_source.provenance(),
                    "candidate_source_binding": study._candidate_source_binding(
                        active_source, identity["producer_identity"]
                    ),
                    "generation": {
                        "on_the_fly": {
                            "seconds": 1.0,
                            "seed_binding_calls": 1,
                            "seed_count": 1,
                            "materialized_lane_calls": 0,
                        }
                    },
                    "selector_contract": contract.as_dict(),
                    "artifacts": {"candidate": candidate_identity},
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(study, "_active_report_source", lambda: active_source)
    monkeypatch.setattr(study, "_load_amplicol_reference", forbidden)
    loaded: list[tuple[Path, str]] = []

    def load(path: Path, *, process: str) -> object:
        loaded.append((Path(path), process))
        return runtime

    monkeypatch.setattr(study.Runtime, "load", load)
    monkeypatch.setattr(
        study,
        "_artifact_identity",
        lambda *_: candidate_identity,
    )
    monkeypatch.setattr(
        study,
        "_cross_check_compact_selector",
        lambda *_, **__: {"selected_color_ids": [selector.flow_id]},
    )
    monkeypatch.setattr(
        study,
        "_requested_workload_cold_warmup",
        lambda *_, **__: {
            "requested_workload": "all-flow",
            "batch_size": 128,
            "point_count": 128,
            "seconds": 0.5,
        },
    )
    monkeypatch.setattr(
        study,
        "_lifecycle",
        lambda *args: (
            study.Evaluation((1j,), None),
            study.Evaluation((1j,), None),
            {"after_rebuild": {}},
        ),
    )
    monkeypatch.setattr(
        study,
        "_benchmark",
        lambda *_: {
            "wall_seconds_per_point": 1e-6,
            "evaluator_seconds_per_point": 0.8e-6,
        },
    )
    monkeypatch.setattr(study, "_census", lambda *_: {})
    monkeypatch.setattr(study, "_compare", lambda *_: {})
    monkeypatch.setattr(study, "_generate_recurrence", forbidden)
    monkeypatch.setattr(study, "derive_selector_contract", forbidden)
    monkeypatch.setattr(study.ModelSource, "from_path", forbidden)
    result = study._run_worker(args)
    assert loaded == [(candidate.resolve(), case.process_id)]
    assert result["candidate_reuse"]["status"] == "actual-reuse"
    assert (
        result["generation"] == {} and result["correctness"]["status"] == "not-claimed"
    )
    assert result["candidate_source_binding"]["status"] == "passed"
    assert result["amplicol_reference"] is None
    assert set(result["timings"]) == {"on_the_fly"}
    assert result["generation_comparison"] == {
        "notation": "[G] G+W",
        "reporting": "absolute-on-the-fly-seconds",
        "source": "reused-selected-artifact-generation",
        "warmup_source": "this-all-flow-cold-batch128-run",
        "on_the_fly_generation_seconds": 1.0,
        "on_the_fly_warmup_seconds": 0.5,
        "on_the_fly_generation_plus_warmup_seconds": 1.5,
    }
    assert result["descriptive_ratios"] == {}
    assert result["forbidden_paths"]["external_comparator"] == "not-created"
    assert result["forbidden_paths"]["physics_enumeration"] == "not-called"
    forbidden.assert_not_called()


def test_resolved_total_must_match_public_total() -> None:
    selector = study.Selector((1,), "flow:1", (-1,), "h:-1")

    class Resolved:
        color_ids = ("flow:1",)

        @staticmethod
        def total() -> tuple[complex, ...]:
            return (2.0 + 0.0j,)

    runtime = SimpleNamespace(
        evaluate=lambda *_args, **_kwargs: (1.0 + 0.0j,),
        evaluate_resolved=lambda *_args, **_kwargs: Resolved(),
    )
    with pytest.raises(study.StudyError):
        study._evaluate(
            runtime,
            ((((1.0, 0.0, 0.0, 1.0),),)),
            "selected",
            selector,
            resolved=True,
        )


@pytest.mark.parametrize(
    "mutation",
    (
        None,
        "artifact-id",
        "canonical-path",
        "authority",
        "source-lineage",
        "selector-proof",
        "contract-flow",
    ),
)
def test_selected_report_binds_exact_candidate(
    mutation: str | None, tmp_path: Path
) -> None:
    case = study._case(7, 5)
    candidate_path = tmp_path / "candidate"
    candidate_path.mkdir()
    canonical = str(candidate_path.resolve())
    artifact_id = "a" * 64
    source = study.ReportSourceIdentity("b" * 40, "e" * 40, ())
    producer = {
        "source_revision": source.revision,
        "native_build_inputs_sha256": "c" * 64,
    }
    model = {"content_sha256": "d" * 64}
    recorded_path = canonical
    recorded_id = artifact_id
    if mutation == "artifact-id":
        recorded_id = "b" * 64
    elif mutation == "canonical-path":
        recorded_path = (
            f"{candidate_path.parent}/../{candidate_path.parent.name}/candidate"
        )
    payload = {
        "kind": f"{study.KIND}-run",
        "schema_version": study.SCHEMA_VERSION,
        "status": "passed",
        "watchdog": {"passes": True, "limit_bytes": study.WATCHDOG_BYTES},
        "study": {
            "kind": study.KIND,
            "schema_version": study.SCHEMA_VERSION,
            "status": "passed",
            "workload": "selected",
            "process": study.dataclass_payload(case),
            "authority": {"kind": study.AUTHORITY_AMPLICOL},
            "correctness": {
                "status": "passed",
                "otf_before_vs_amplicol_replay": {"checks": 8},
                "otf_after_vs_amplicol_replay": {"checks": 8},
            },
            "compact_selector_context": {
                "ordinal_one_required": True,
                "ordinal_one_matches_authority": True,
                "requested_color_ids": [
                    "flow:2,5,6,7,4,3,1",
                    "1",
                ],
                "selected_color_ids": [
                    "flow:2,5,6,7,4,3,1",
                    "flow:2,5,6,7,4,3,1",
                ],
                "semantic_authority_flow_id": "flow:2,5,6,7,4,3,1",
            },
            "producer_identity": producer,
            "model_identity": model,
            "active_source": source.provenance(),
            "candidate_source_binding": study._candidate_source_binding(
                source, producer
            ),
            "generation": {
                "on_the_fly": {
                    "seconds": 1.0,
                    "seed_binding_calls": 1,
                    "seed_count": 1,
                    "materialized_lane_calls": 0,
                }
            },
            "selector_contract": SelectorContract(
                ("flow:2,5,6,7,4,3,1",),
                ((2, 5, 6, 7, 4, 3, 1),),
                ("h:-1,+1,-1,+1,-1,+1,-1",),
                tuple(enumerate(fixed_selector_helicity(case.pdgs), start=1)),
                "f" * 64,
            ).as_dict(),
            "artifacts": {
                "candidate": {"path": recorded_path, "artifact_id": recorded_id}
            },
        },
    }
    if mutation == "authority":
        payload["study"]["authority"]["kind"] = study.AUTHORITY_OTF_ONLY
    elif mutation == "source-lineage":
        payload["study"]["producer_identity"]["source_revision"] = "e" * 40
    elif mutation == "selector-proof":
        payload["study"]["compact_selector_context"][
            "ordinal_one_matches_authority"
        ] = False
    elif mutation == "contract-flow":
        payload["study"]["compact_selector_context"]["semantic_authority_flow_id"] = (
            "flow:1,2,3,4,5,6,7"
        )
    report = tmp_path / f"selected-{mutation or 'valid'}.json"
    report.write_text(json.dumps(payload), encoding="utf-8")
    candidate = {
        "path": canonical,
        "artifact_id": artifact_id,
        "producer_identity": producer,
        "model_identity": model,
    }
    if mutation is None:
        lineage = study._selected_report_lineage(report, case, candidate, source)
        assert lineage["candidate_path"] == canonical
        assert lineage["candidate_artifact_id"] == artifact_id
    else:
        with pytest.raises(study.StudyError):
            study._selected_report_lineage(report, case, candidate, source)


@pytest.mark.parametrize(
    ("interrupted", "target", "batch", "warmups", "minimum", "samples"),
    (
        (True, 0.1, 128, 2, 5, 5),
        (False, 0.2, 128, 2, 5, 5),
        (False, 0.1, 64, 2, 5, 5),
        (False, 0.1, 128, 1, 5, 5),
        (False, 0.1, 128, 2, 4, 5),
        (False, 0.1, 128, 2, 5, 4),
    ),
)
def test_benchmark_rejects_partial_or_wrong_effective_contract(
    monkeypatch: pytest.MonkeyPatch,
    interrupted: bool,
    target: float,
    batch: int,
    warmups: int,
    minimum: int,
    samples: int,
) -> None:
    result = SimpleNamespace(
        interrupted=interrupted,
        sample_count=samples,
        effective_config=SimpleNamespace(
            target_runtime=target,
            precision=16,
            batch_size=batch,
            warmup_runs=warmups,
            minimum_samples=minimum,
            helicity_ids=(),
            color_flow_ids=("flow:1",),
        ),
    )
    monkeypatch.setattr(
        study,
        "BenchmarkRunner",
        lambda _config: SimpleNamespace(run=lambda *_args, **_kwargs: result),
    )
    with pytest.raises(study.StudyError):
        study._benchmark(
            SimpleNamespace(),
            "on-the-fly",
            ((((1.0, 0.0, 0.0, 1.0),),)),
            "selected",
            study.Selector((1,), "flow:1", (-1,), "h:-1"),
            0.1,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("precision", 8),
        ("helicity_ids", ("h:-1",)),
        ("color_flow_ids", ("flow:2",)),
    ),
)
def test_benchmark_rejects_wrong_precision_or_selector(
    monkeypatch: pytest.MonkeyPatch, field: str, value: object
) -> None:
    effective = SimpleNamespace(
        target_runtime=0.1,
        precision=16,
        batch_size=128,
        warmup_runs=2,
        minimum_samples=5,
        helicity_ids=(),
        color_flow_ids=("flow:1",),
    )
    setattr(effective, field, value)
    result = SimpleNamespace(
        interrupted=False,
        sample_count=5,
        effective_config=effective,
    )
    monkeypatch.setattr(
        study,
        "BenchmarkRunner",
        lambda _config: SimpleNamespace(run=lambda *_args, **_kwargs: result),
    )
    with pytest.raises(study.StudyError, match="effective timing contract"):
        study._benchmark(
            SimpleNamespace(),
            "on-the-fly",
            ((((1.0, 0.0, 0.0, 1.0),),)),
            "selected",
            study.Selector((1,), "flow:1", (-1,), "h:-1"),
            0.1,
        )
