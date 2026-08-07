# SPDX-License-Identifier: 0BSD
"""Focused contracts for the bounded OTF high-multiplicity driver."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from tools.developer import on_the_fly_high_multiplicity as study
from tools.performance_report.runner import SelectorContract
from tools.performance_report.selector_policy import fixed_selector_helicity


def _args(tmp_path: Path, n: int, workload: str) -> argparse.Namespace:
    return argparse.Namespace(
        output=tmp_path / "out",
        multiplicity=n,
        workload=workload,
        prepared_model=None,
        candidate_artifact=None,
        selected_report=None,
        target_runtime=0.1,
        worker=False,
    )


def test_exact_cases_helicities_and_cli_routes(tmp_path: Path) -> None:
    for n in study.SUPPORTED_MULTIPLICITIES:
        case = study._case(n)
        gluons = n - 2
        assert case.process == "d d~ > t t~ " + " ".join("g" for _ in range(gluons))
        assert case.process_id == f"otf_dd_tt_{gluons}g"
        assert case.pdgs == (1, -1, 6, -6, *(21 for _ in range(gluons)))
        assert fixed_selector_helicity(case.pdgs) == tuple(
            -1 if index % 2 else 1 for index in range(1, n + 3)
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
    )
    for args in invalid:
        with pytest.raises(study.StudyError):
            study._validate_arguments(args)
    with pytest.raises(SystemExit):
        study._parser().parse_args(
            ["--output", "x", "--multiplicity", "5", "--recurrence-artifact", "x"]
        )


def test_selector_derivation_and_compact_path_never_open_physics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = study._case(5)
    states = fixed_selector_helicity(case.pdgs)
    word = tuple(range(1, len(case.pdgs) + 1))
    contract = SelectorContract(
        ("flow:" + ",".join(map(str, word)),),
        (word,),
        ("h:" + ",".join(f"{state:+d}" for state in states),),
        tuple(enumerate(states, start=1)),
        "a" * 64,
    )
    derive, validate = mock.Mock(return_value=contract), mock.Mock()
    monkeypatch.setattr(study, "derive_selector_contract", derive)
    monkeypatch.setattr(study, "validate_selector_contract", validate)
    recurrence, compiled = (
        SimpleNamespace(execution_mode="recurrence"),
        SimpleNamespace(execution_mode="compiled"),
    )
    selector, _payload = study._authority_contract(case, recurrence, compiled, ())
    assert selector.flow_word == word and selector.helicities == states
    assert validate.call_args_list == [
        mock.call(recurrence, contract, ()),
        mock.call(compiled, contract, ()),
    ]

    case = study._case(8)

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
                "selected_color_ids": ["flow:1,2,3,4,5,6,7,8,9,10"],
            }

    selector, context = study._compact_reference_selector(
        SimpleNamespace(_backend=Backend()), case
    )
    assert selector.flow_word == tuple(range(1, 11))
    assert context["requested_color_ids"] == ["1"]


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
    monkeypatch.setattr(
        study.time,
        "perf_counter_ns",
        mock.Mock(side_effect=(10_000_000, 14_000_000)),
    )
    runtime = SimpleNamespace(
        clear=mock.Mock(), evaluate=mock.Mock(return_value=(1.0 + 0.0j,))
    )
    lifecycle = study._lifecycle(
        runtime,
        study._case(5),
        study.Selector((1,), "flow:1", (-1,), "h:-1"),
        ((((1.0, 0.0, 0.0, 1.0),),)),
        "all-flow",
        False,
    )[2]
    assert lifecycle["sequence"] == "A,B,B,A; clear; A,B"
    assert lifecycle["after_rebuild"] == state_b
    cold_timing = lifecycle["cold_first_evaluation_timing"]
    assert cold_timing == {
        "kind": "on-the-fly-cold-first-evaluation-timing-v1",
        "timer": "time.perf_counter_ns",
        "runtime_state": "census-proven-cold",
        "family": "selected-A",
        "workload": "selected",
        "requested_study_workload": "all-flow",
        "requested_study_workload_cold_timed": False,
        "requested_all_flow_family_preparation": "not-independently-cold-timed",
        "elapsed_nanoseconds": 4_000_000,
        "seconds": 0.004,
        "point_count": 1,
        "seconds_per_point": 0.004,
        "excluded_from_elapsed": [
            "Runtime.load",
            "artifact generation",
            "resolved-output follow-up",
            "warmed BenchmarkRunner",
        ],
        "ratio_eligible": False,
        "acceptance_eligible": False,
    }
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
    monkeypatch.setattr(
        study.time,
        "perf_counter_ns",
        mock.Mock(side_effect=(20_000_000, 25_000_000)),
    )
    runtime = SimpleNamespace(
        clear=mock.Mock(), evaluate=mock.Mock(return_value=(1.0 + 0.0j,))
    )
    lifecycle = study._lifecycle(
        runtime,
        study._case(8),
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
    cold_timing = lifecycle["cold_first_evaluation_timing"]
    assert cold_timing["requested_study_workload"] == "selected"
    assert cold_timing["requested_study_workload_cold_timed"] is True
    assert cold_timing["requested_all_flow_family_preparation"] == "not-applicable"
    assert cold_timing["seconds"] == 0.005
    assert cold_timing["point_count"] == 1
    runtime.clear.assert_called_once_with()


def test_identity_n8_forbidden_route_and_worker_argv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    case, artifact_id = study._case(8), "a" * 64
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
        study._selected_report_lineage(wrong_report, case, identity)

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
    forbidden = mock.Mock(side_effect=AssertionError("forbidden n>=8 route"))
    monkeypatch.setattr(study.Runtime, "load", lambda *_, **__: runtime)
    monkeypatch.setattr(
        study,
        "_artifact_identity",
        lambda *_: {
            "producer_identity": identity["producer_identity"],
            "model_identity": identity["model_identity"],
        },
    )
    monkeypatch.setattr(
        study, "_selected_report_lineage", lambda *_: {"status": "passed"}
    )
    monkeypatch.setattr(
        study,
        "_compact_reference_selector",
        lambda *_: (selector, {"selected_color_ids": [selector.flow_id]}),
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
        study, "_benchmark", lambda *_: {"wall_seconds_per_point": 1e-6}
    )
    monkeypatch.setattr(study, "_census", lambda *_: {})
    monkeypatch.setattr(study, "_compare", lambda *_: {})
    monkeypatch.setattr(study, "_generate_authority", forbidden)
    monkeypatch.setattr(study.ModelSource, "from_path", forbidden)
    result = study._run_worker(args)
    assert result["candidate_reuse"]["status"] == "actual-reuse"
    assert (
        result["generation"] == {} and result["correctness"]["status"] == "not-claimed"
    )


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


@pytest.mark.parametrize("mutation", (None, "artifact-id", "canonical-path"))
def test_selected_report_binds_exact_candidate(
    mutation: str | None, tmp_path: Path
) -> None:
    case = study._case(5)
    candidate_path = tmp_path / "candidate"
    candidate_path.mkdir()
    canonical = str(candidate_path.resolve())
    artifact_id = "a" * 64
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
        "study": {
            "kind": study.KIND,
            "schema_version": study.SCHEMA_VERSION,
            "status": "passed",
            "workload": "selected",
            "process": study.dataclass_payload(case),
            "generation": {
                "on_the_fly": {
                    "seed_binding_calls": 1,
                    "seed_count": 1,
                    "materialized_lane_calls": 0,
                }
            },
            "artifacts": {
                "candidate": {"path": recorded_path, "artifact_id": recorded_id}
            },
        },
    }
    report = tmp_path / f"selected-{mutation or 'valid'}.json"
    report.write_text(json.dumps(payload), encoding="utf-8")
    candidate = {"path": canonical, "artifact_id": artifact_id}
    if mutation is None:
        lineage = study._selected_report_lineage(report, case, candidate)
        assert lineage["candidate_path"] == canonical
        assert lineage["candidate_artifact_id"] == artifact_id
    else:
        with pytest.raises(study.StudyError, match="disagree"):
            study._selected_report_lineage(report, case, candidate)


@pytest.mark.parametrize(
    ("interrupted", "batch", "warmups", "minimum", "samples"),
    (
        (True, 128, 2, 5, 5),
        (False, 64, 2, 5, 5),
        (False, 128, 1, 5, 5),
        (False, 128, 2, 4, 5),
        (False, 128, 2, 5, 4),
    ),
)
def test_benchmark_rejects_partial_or_wrong_effective_contract(
    monkeypatch: pytest.MonkeyPatch,
    interrupted: bool,
    batch: int,
    warmups: int,
    minimum: int,
    samples: int,
) -> None:
    result = SimpleNamespace(
        interrupted=interrupted,
        sample_count=samples,
        effective_config=SimpleNamespace(
            batch_size=batch,
            warmup_runs=warmups,
            minimum_samples=minimum,
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
