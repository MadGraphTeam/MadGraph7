# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
import sys
from decimal import Decimal
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from tools.developer import fft_numerical_acceptance as gate


def test_catalog_and_comparison_matrix_are_derived_at_locked_bounds() -> None:
    selected = gate.catalog_cases(gate.SELECTED_MAX_N_FINAL)
    totals = gate.catalog_cases(gate.TOTAL_MAX_N_FINAL)

    assert len(selected) == 47
    assert len(totals) == 33
    assert {case.family_id for case in selected}.isdisjoint({14})
    assert all(case.n_final <= 5 for case in selected)
    assert all(case.n_final <= 4 for case in totals)

    specs = gate.comparison_specs()
    direct_fft = tuple(spec for spec in specs if spec.authority == "direct-vs-fft")
    frozen = tuple(spec for spec in specs if spec.authority == "frozen-madgraph")
    assert len(direct_fft) == 508
    assert len(frozen) == 66
    assert len(specs) == 574
    assert all(spec.model == "ufo-sm" for spec in frozen)
    assert all(spec.accuracy == "full" for spec in frozen)
    assert all(spec.helicity_scope == "total" for spec in frozen)


def test_dry_run_authenticates_frozen_fixture_and_declares_cache_policy() -> None:
    payload = gate.dry_run_payload()

    assert payload["dry_run"] is True
    assert payload["point_policy"] == {
        "generator": "generic_validation_point",
        "seed": 101,
    }
    assert payload["comparison_counts"] == {
        "direct_vs_fft": 508,
        "frozen_madgraph": 66,
        "total": 574,
    }
    cache = payload["process_set_cache_policy"]
    assert cache["per_process_generation"] is False
    assert cache["base_key_axes"] == [
        "method",
        "model",
        "mode",
        "accuracy",
        "helicity_scope",
    ]
    assert cache["selected_execution"] == {
        "recurrence": "generation-specialized",
        "on-the-fly": "runtime-query-complete-coverage",
    }
    assert "shared by selected runtime queries" in cache["on_the_fly_complete_coverage"]
    assert payload["frozen_authority"]["madgraph_rerun"] is False


def test_dry_run_cli_does_not_create_output_root(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_root = tmp_path / "must-not-exist"
    assert gate.main(["--dry-run", "--output-root", str(output_root)]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["kind"] == gate.GATE_KIND
    assert payload["comparison_counts"]["total"] == 574
    assert not output_root.exists()


def test_scale_relative_comparison_has_no_absolute_tolerance_escape() -> None:
    assert gate.strict_relative_compare(0, 0).passed
    assert gate.strict_relative_compare(Decimal("1.00000000005"), Decimal("1")).passed
    assert not gate.strict_relative_compare(Decimal("1e-300"), Decimal("2e-300")).passed
    assert not gate.strict_relative_compare(1 + 1e-8j, 1).passed


def test_nonzero_discovery_prefers_structural_selector_then_stable_fallback() -> None:
    rows = (
        gate.HelicityObservation("h:-1,-1", (-1, -1), False, 5.0),
        gate.HelicityObservation("h:-1,+1", (-1, 1), False, 2.0),
        gate.HelicityObservation("h:+1,-1", (1, -1), True, 0.0),
        gate.HelicityObservation("h:+1,+1", (1, 1), False, 0.0),
    )

    preferred = gate.choose_nonzero_helicity(rows, preferred=(-1, 1))
    assert preferred.observation.values == (-1, 1)
    assert preferred.source == "preferred-structural-selector"
    assert preferred.domains == ((-1, 1), (-1, 1))

    fallback = gate.choose_nonzero_helicity(rows, preferred=(1, 1))
    assert fallback.observation.values == (-1, -1)
    assert fallback.source == "largest-nonzero-fallback"


def test_nonzero_discovery_rejects_an_all_zero_or_structural_axis() -> None:
    with pytest.raises(gate.FFTAcceptanceError, match="no nonzero"):
        gate.choose_nonzero_helicity(
            (
                gate.HelicityObservation("h:-1", (-1,), False, 0.0),
                gate.HelicityObservation("h:+1", (1,), True, 3.0),
            ),
            preferred=(-1,),
        )


def test_selected_groups_include_domain_and_exact_tuple_in_scope() -> None:
    cases = gate.catalog_cases(2)[:3]

    def selection(
        case: gate.CatalogCase,
        values: tuple[int, ...],
        domains: tuple[tuple[int, ...], ...],
    ) -> gate.SelectionRecord:
        return gate.SelectionRecord(
            model="built-in-sm",
            accuracy="full",
            case=case,
            helicity_id="h:" + ",".join(f"{value:+d}" for value in values),
            values=values,
            domains=domains,
            source="unit-test",
            discovery_value=1.0,
        )

    first_values = (-1, 1, -1)
    first_domains = ((-1, 1), (-1, 1), (-1, 0, 1))
    second_domains = ((-1, 1), (-1, 1), (-1, 1))
    selections = {
        cases[0].case_id: selection(cases[0], first_values, first_domains),
        cases[1].case_id: selection(cases[1], first_values, first_domains),
        cases[2].case_id: selection(cases[2], first_values, second_domains),
    }

    groups = gate.group_selected_cases(cases, selections)
    assert tuple(len(group.cases) for group in groups) == (2, 1)
    assert groups[0].scope != groups[1].scope
    assert groups[0].source_mapping == ((1, -1), (2, 1), (3, -1))


def test_process_set_cache_builds_once_per_exact_boundary() -> None:
    calls: list[gate.GroupRequest] = []

    def build(request: gate.GroupRequest) -> object:
        calls.append(request)
        return object()

    cases = gate.catalog_cases(1)
    request = gate.GroupRequest(
        gate.ArtifactKey(
            "direct",
            "built-in-sm",
            "recurrence",
            "full",
            "all-helicity-total",
        ),
        cases,
    )
    cache: gate.ProcessSetCache[object] = gate.ProcessSetCache(build)

    first = cache.get(request)
    second = cache.get(request)
    assert first is second
    assert calls == [request]
    assert cache.generation_count == 1

    incompatible = gate.GroupRequest(request.key, cases[:1])
    with pytest.raises(gate.FFTAcceptanceError, match="different case"):
        cache.get(incompatible)


def test_group_request_rejects_selection_outside_recorded_domain() -> None:
    case = gate.catalog_cases(1)[0]
    with pytest.raises(ValueError, match="outside its domain"):
        gate.GroupRequest(
            gate.ArtifactKey(
                "direct",
                "built-in-sm",
                "recurrence",
                "full",
                "selected-invalid",
            ),
            (case,),
            selected_source_helicities=((1, 0),),
            external_helicity_domains=((-1, 1),),
        )


def test_otf_group_request_is_complete_and_rejects_generation_selection() -> None:
    cases = gate.catalog_cases(gate.SELECTED_MAX_N_FINAL)
    request = gate.otf_complete_group_request(
        method="symmetric-group-fft",
        model="ufo-sm",
        accuracy="nlc",
        cases=cases,
    )

    assert request.key.mode == "on-the-fly"
    assert request.key.helicity_scope == "complete-helicity-runtime-query-and-total"
    assert request.cases == cases
    assert request.selected_source_helicities is None
    assert request.external_helicity_domains is None
    assert request.as_payload()["helicity_generation_contract"] == "complete-coverage"

    with pytest.raises(ValueError, match="must retain complete helicity coverage"):
        gate.GroupRequest(
            gate.ArtifactKey(
                "direct",
                "built-in-sm",
                "on-the-fly",
                "full",
                "invalid-selected-otf",
            ),
            cases[:1],
            selected_source_helicities=((1, -1),),
            external_helicity_domains=((-1, 1),),
        )


def test_selected_artifact_plan_specializes_recurrence_but_groups_complete_otf(
    tmp_path: Path,
) -> None:
    cases = gate.catalog_cases(1)
    harness = object.__new__(gate.NativeAcceptanceHarness)
    harness.selected_cases = cases
    harness.total_cases = cases
    harness.selections = {
        (model, accuracy, case.case_id): gate.SelectionRecord(
            model=model,
            accuracy=accuracy,
            case=case,
            helicity_id="h:-1,+1,-1",
            values=(-1, 1, -1),
            domains=((-1, 1), (-1, 1), (-1, 1)),
            source="unit-test",
            discovery_value=3.0,
        )
        for model in gate.MODELS
        for accuracy in gate.ACCURACIES
        for case in cases
    }

    class RecordingCache:
        def __init__(self) -> None:
            self.requests: list[gate.GroupRequest] = []

        def get(self, request: gate.GroupRequest) -> gate.ArtifactRecord:
            self.requests.append(request)
            return gate.ArtifactRecord(request, tmp_path / request.key.slug)

    cache = RecordingCache()
    harness.cache = cache
    fallback_scopes = frozenset(
        (model, accuracy) for model in gate.MODELS for accuracy in gate.ACCURACIES
    )
    records = harness._selected_artifacts({}, fallback_scopes)

    recurrence = tuple(
        request for request in cache.requests if request.key.mode == "recurrence"
    )
    otf = tuple(
        request for request in cache.requests if request.key.mode == "on-the-fly"
    )
    assert len(recurrence) == 8
    assert all(request.selected_source_helicities for request in recurrence)
    assert len(otf) == 8
    assert all(request.cases == cases for request in otf)
    assert all(request.selected_source_helicities is None for request in otf)
    for model in gate.MODELS:
        for accuracy in gate.ACCURACIES:
            for method in gate.METHODS:
                lane_paths = {
                    records[(model, "on-the-fly", accuracy, method, case.case_id)].path
                    for case in cases
                }
                assert len(lane_paths) == 1

    request_count = len(cache.requests)
    totals = harness._total_artifacts(records)
    assert len(cache.requests) == request_count + 4
    for model in gate.MODELS:
        for method in gate.METHODS:
            selected_path = records[
                (model, "on-the-fly", "full", method, cases[0].case_id)
            ].path
            assert totals[(model, "on-the-fly", method)].path == selected_path


def test_otf_selected_evaluation_uses_exact_runtime_helicity_query(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    case = gate.catalog_cases(1)[0]
    values = (-1, 1, -1)
    helicity_id = "h:-1,+1,-1"
    selection = gate.SelectionRecord(
        model="built-in-sm",
        accuracy="full",
        case=case,
        helicity_id=helicity_id,
        values=values,
        domains=((-1, 1), (-1, 1), (-1, 1)),
        source="unit-test",
        discovery_value=3.0,
    )
    request = gate.otf_complete_group_request(
        method="direct",
        model="built-in-sm",
        accuracy="full",
        cases=(case,),
    )
    record = gate.ArtifactRecord(request, tmp_path / "artifact")
    calls: list[tuple[str, ...] | None] = []

    class FakeRuntime:
        execution_mode = "on-the-fly"
        physics = SimpleNamespace(
            color_accuracy="full",
            helicity_coverage="complete",
            helicities=(
                SimpleNamespace(id="h:-1,-1,-1", values=(-1, -1, -1)),
                SimpleNamespace(id=helicity_id, values=values),
            ),
        )

        def evaluate(
            self,
            _momenta: object,
            *,
            helicities: tuple[str, ...] | None = None,
        ) -> tuple[complex, ...]:
            calls.append(helicities)
            return (3.0 + 0.0j,)

        def clear(self) -> None:
            pass

    runtime = FakeRuntime()
    fake_pyamplicol = ModuleType("pyamplicol")
    fake_pyamplicol.Runtime = SimpleNamespace(  # type: ignore[attr-defined]
        load=lambda _path, *, process: runtime
    )
    monkeypatch.setitem(sys.modules, "pyamplicol", fake_pyamplicol)
    monkeypatch.setattr(
        gate,
        "_runtime_points",
        lambda _case: (((1.0, 0.0, 0.0, 1.0),),),
    )

    assert gate._evaluate_scalar(record, case, selection=selection) == 3.0 + 0.0j
    assert calls == [(helicity_id,)]


def test_shared_complete_artifact_caches_selected_and_total_values_separately(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    case = gate.catalog_cases(1)[0]
    request = gate.otf_complete_group_request(
        method="direct",
        model="built-in-sm",
        accuracy="full",
        cases=(case,),
    )
    record = gate.ArtifactRecord(request, tmp_path / "artifact")
    selection = gate.SelectionRecord(
        model="built-in-sm",
        accuracy="full",
        case=case,
        helicity_id="h:-1,+1,-1",
        values=(-1, 1, -1),
        domains=((-1, 1), (-1, 1), (-1, 1)),
        source="unit-test",
        discovery_value=3.0,
    )
    calls: list[gate.SelectionRecord | None] = []

    def evaluate(
        _record: gate.ArtifactRecord,
        _case: gate.CatalogCase,
        *,
        selection: gate.SelectionRecord | None,
    ) -> float:
        calls.append(selection)
        return 3.0 if selection is not None else 7.0

    monkeypatch.setattr(gate, "_evaluate_scalar", evaluate)
    harness = object.__new__(gate.NativeAcceptanceHarness)
    harness._values = {}

    assert harness._value(record, case, selection=selection) == 3.0
    assert harness._value(record, case, selection=None) == 7.0
    assert harness._value(record, case, selection=selection) == 3.0
    assert calls == [selection, None]
