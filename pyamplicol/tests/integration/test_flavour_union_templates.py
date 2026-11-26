# SPDX-License-Identifier: 0BSD
"""Flavour-union artifacts against separately generated subprocesses.

With ``process.flavor_unions`` the subprocesses that differ only in union-quark
flavours are evaluated through one pattern-resolved template: a member sums the
template amplitudes of its pairing patterns before squaring. Every subprocess,
loaded by its process key, must match the artifact that generates each
subprocess on its own, per helicity and LC flow.

``p p > j j j`` and ``p p > j j j j`` run only with
``PYAMPLICOL_RUN_FLAVOUR_UNION=pp3j,pp4j`` (either or both).
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import time
from collections.abc import Iterator, Mapping
from pathlib import Path

import pytest

from pyamplicol import Generator, ProcessSet, Runtime
from pyamplicol.api.errors import CompatibilityError
from pyamplicol.artifacts.process_groups import read_process_groups
from pyamplicol.config import (
    ColorConfig,
    EvaluatorConfig,
    EvaluatorOptimizationConfig,
    GenerationConfig,
    GenerationRelationDiscoveryConfig,
    GenerationValidationConfig,
    JITConfig,
    ProcessConfig,
    RunConfig,
)
from pyamplicol.generation.process_grouping import GroupingBasis
from pyamplicol.models.builtin.validation import generic_validation_point

_EXPLICIT = (
    "d d~ > d d d~ d~",
    "d d~ > d u d~ u~",
    "d d~ > u s u~ s~",
    "d d~ > u u u~ u~",
    "u u~ > u u u~ u~",
    "d d > d d g g",
    "d u > d u g g",
    "d u > u d g g",
    "u u > u u g g",
    "d d~ > d d~ g g",
    "d d~ > u u~ g g",
    "d u~ > d u~ g g",
)
_CASES = {
    "explicit": _EXPLICIT,
    "pp2j": ("p p > j j",),
    # No all-d process: the union template is the requested 'u u~ > u u~'.
    "uu2j": ("u u~ > j j",),
    "pp3j": ("p p > j j j",),
    "pp4j": ("p p > j j j j",),
}
_OPT_IN = {"pp3j", "pp4j"}
_OPT_IN_ENV = "PYAMPLICOL_RUN_FLAVOUR_UNION"


def _require_native_runtime() -> None:
    if importlib.util.find_spec("pyamplicol._rusticol") is None:
        pytest.skip("the Rusticol extension has not been built")
    if importlib.util.find_spec("symbolica") is None:
        pytest.skip("Symbolica is unavailable")


def _config(*, unions: bool) -> RunConfig:
    return RunConfig(
        action="generate",
        color=ColorConfig(accuracy="lc"),
        process=ProcessConfig(
            flavor_scheme=4,
            flavor_blind_strict=unions,
            flavor_unions=unions,
            max_coupling_orders={"QED": 0},
        ),
        generation=GenerationConfig(
            workers=1,
            emit_api_bundle=False,
            relation_discovery=GenerationRelationDiscoveryConfig(mode="off"),
            validation=GenerationValidationConfig(
                enabled=False, post_build_validation=False
            ),
        ),
        evaluator=EvaluatorConfig(
            execution_mode="recurrence",
            optimization=EvaluatorOptimizationConfig(cores=1),
            jit=JITConfig(optimization_level=2),
        ),
    )


def _row_count(artifact: Path) -> int:
    raw = json.loads((artifact / "artifact.json").read_text(encoding="utf-8"))
    rows = 0
    for process in raw["processes"]:
        execution = json.loads(
            (artifact / "processes" / process["id"] / "execution.json").read_text(
                encoding="utf-8"
            )
        )
        certificate = execution["plan"]["inspection_summary"].get(
            "selector_work_certificate"
        )
        rows += int((certificate or {}).get("persisted_union", {}).get("row_count", 0))
    return rows


@pytest.fixture(scope="module", params=tuple(_CASES))
def union_artifacts(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[dict[str, Path]]:
    if request.param in _OPT_IN and request.param not in os.environ.get(
        _OPT_IN_ENV, ""
    ).split(","):
        pytest.skip(f"set {_OPT_IN_ENV}={request.param} to run it")
    _require_native_runtime()
    root = tmp_path_factory.mktemp(f"flavour-union-{request.param}")
    artifacts = {"union": root / "union", "reference": root / "reference"}
    processes = ProcessSet.from_expressions(_CASES[request.param])
    try:
        for kind in ("union", "reference"):
            started = time.perf_counter()
            Generator(_config(unions=kind == "union")).generate(
                processes, artifacts[kind]
            )
            print(
                f"{request.param} {kind}: {time.perf_counter() - started:.1f} s, "
                f"{_row_count(artifacts[kind])} rows"
            )
        yield artifacts
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _pdgs(expression: str) -> tuple[int, ...]:
    return tuple(int(particle.pdg) for particle in generic_validation_point(expression))


def _point(expression: str) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(float(component) for component in particle.momentum)  # type: ignore[misc]
        for particle in generic_validation_point(expression)
    )


def _canonical_flow(expression: str, color: str) -> tuple[tuple[str, ...], ...]:
    """LC flow as its set of open strings; relabelled members list them unordered."""

    pdgs = _pdgs(expression)
    initial = len(expression.split(">")[0].split())
    fundamental = {
        str(index + 1)
        for index, pdg in enumerate(pdgs)
        if abs(pdg) <= 6 and (pdg > 0) == (index >= initial)
    }
    strings: list[list[str]] = []
    for label in color.removeprefix("flow:").split(","):
        if label in fundamental or not strings:
            strings.append([])
        strings[-1].append(label)
    return tuple(sorted(tuple(string) for string in strings))


def _flow_values(runtime: Runtime, point: object) -> dict[tuple[str, object], float]:
    resolved = runtime.evaluate_resolved((point,))  # type: ignore[arg-type]
    expression = runtime.physics.process
    return {
        (helicity, _canonical_flow(expression, color)): float(
            complex(resolved.values[0][h][c]).real
        )
        for h, helicity in enumerate(resolved.helicity_ids)
        for c, color in enumerate(resolved.color_ids)
    }


def _assert_flows_match(
    actual: Mapping[tuple[str, object], float],
    expected: Mapping[tuple[str, object], float],
    context: str,
) -> None:
    assert actual.keys() == expected.keys(), context
    scale = max(abs(value) for value in expected.values())
    assert scale > 0.0, context
    for key, value in expected.items():
        assert actual[key] == pytest.approx(value, rel=1.0e-12, abs=1.0e-12 * scale), (
            context,
            key,
        )


def _process_keys(artifact: Path) -> dict[str, str]:
    raw = json.loads((artifact / "artifact.json").read_text(encoding="utf-8"))
    return {
        str(process["id"]): str(process["expression"]) for process in raw["processes"]
    }


def test_unions_route_members_through_pattern_resolved_templates(
    union_artifacts: dict[str, Path],
) -> None:
    groups = read_process_groups(union_artifacts["union"])
    assert groups is not None
    union_members = [
        member
        for member in groups.members()
        if member.basis is GroupingBasis.FLAVOUR_UNION
    ]
    assert union_members
    # In template slot order every pattern enters with sign +1.
    assert all(set(member.pattern_signs or ()) == {1} for member in union_members)
    assert len(_process_keys(union_artifacts["union"])) < len(
        _process_keys(union_artifacts["reference"])
    )


def test_every_subprocess_matches_its_separately_generated_program(
    union_artifacts: dict[str, Path],
) -> None:
    reference_keys = _process_keys(union_artifacts["reference"])
    for key, expression in reference_keys.items():
        routed = Runtime.load(union_artifacts["union"], process=key)
        reference = Runtime.load(union_artifacts["reference"], process=key)
        assert routed.physics.process == expression
        point = _point(expression)
        _assert_flows_match(
            _flow_values(routed, point), _flow_values(reference, point), expression
        )
        assert routed.evaluate((point,) * 2) == pytest.approx(
            reference.evaluate((point,) * 2), rel=1.0e-12
        ), expression


def test_side_permuted_expressions_select_union_members(
    union_artifacts: dict[str, Path],
) -> None:
    for expression in _process_keys(union_artifacts["reference"]).values():
        initial, final = expression.split(">")
        legs = initial.split()
        if len(set(legs)) < 2:
            continue
        swapped = f"{' '.join(reversed(legs))} >{final}"
        routed = Runtime.load(union_artifacts["union"], process=swapped)
        reference = Runtime.load(union_artifacts["reference"], process=swapped)
        point = _point(swapped)
        _assert_flows_match(
            _flow_values(routed, point), _flow_values(reference, point), swapped
        )


def test_high_precision_refuses_union_members(
    union_artifacts: dict[str, Path],
) -> None:
    groups = read_process_groups(union_artifacts["union"])
    assert groups is not None
    member = next(
        member
        for member in groups.members()
        if member.basis is GroupingBasis.FLAVOUR_UNION
    )
    routed = Runtime.load(union_artifacts["union"], process=member.process_id)
    with pytest.raises(CompatibilityError, match="pattern-resolved"):
        routed.evaluate((_point(member.expression),), precision=32)


def test_recurrence_export_sums_union_patterns(
    union_artifacts: dict[str, Path], tmp_path: Path
) -> None:
    """Members export their pattern signs; the Rust parity test checks the bits."""
    from pyamplicol.cli.recurrence_export import parse_export_recurrence

    groups = read_process_groups(union_artifacts["union"])
    assert groups is not None
    members = [
        member
        for member in groups.members()
        if member.basis is GroupingBasis.FLAVOUR_UNION
    ]
    member = next(
        (member for member in members if set(member.pattern_signs or ()) != {1}),
        members[0],
    )
    routed = Runtime.load(union_artifacts["union"], process=member.process_id)
    export = routed.export_recurrence()
    assert export["schema_version"] == 3
    programs = export["programs"]
    assert programs
    for program in programs:
        signs = program["pattern_signs"]
        assert set(signs) <= {-1, 0, 1}
        selected = dict(
            zip(member.patterns or (), member.pattern_signs or (), strict=True)
        )
        assert {index: sign for index, sign in enumerate(signs) if sign} == selected
        reductions = [term for run in program["runs"] for term in run["reduction"]]
        assert reductions
        assert all("pattern_terms" in term for term in reductions)
        assert all(
            term["amplitude"] == term["pattern_terms"][0]["amplitude"]
            for term in reductions
        )
    kept = os.environ.get("PYAMPLICOL_RECURRENCE_EXPORT_ARTIFACTS")
    if kept and union_artifacts["union"].parent.name.startswith("flavour-union-uu2j"):
        # CI's Rust parity test interprets this export bit for bit as well.
        destination = Path(kept) / "flavour-union"
        shutil.rmtree(destination, ignore_errors=True)
        shutil.copytree(union_artifacts["union"], destination)
    output = tmp_path / "export.json"
    parse_export_recurrence([str(union_artifacts["union"]), "-o", str(output)]).run()
    combined = json.loads(output.read_text(encoding="utf-8"))
    assert combined["schema_version"] == 3
    entries = {entry["process_key"]: entry for entry in combined["programs"]}
    entry = entries[member.process_id]
    if "program_reference" in entry:
        assert entry["pattern_signs"] == programs[0]["pattern_signs"]
        template = next(
            candidate
            for candidate in combined["programs"]
            if "program_reference" not in candidate
            and candidate["process_key"] == entry["program_reference"]["process_key"]
        )
        assert set(template["pattern_signs"]) == {1}


def test_plan_with_a_compiled_model_keeps_electroweak_flows_routed() -> None:
    """Minimal coupling orders: W-only flavour changes stay out of the unions."""

    _require_native_runtime()
    from pyamplicol import ModelSource

    def concrete(unions: bool) -> list[str]:
        config = RunConfig(
            action="generate",
            color=ColorConfig(accuracy="lc"),
            process=ProcessConfig(
                flavor_scheme=4, flavor_blind_strict=True, flavor_unions=unions
            ),
            generation=GenerationConfig(workers=1, emit_api_bundle=False),
            evaluator=EvaluatorConfig(execution_mode="recurrence"),
        )
        plan = Generator(config).plan("p p > j j", model=ModelSource.built_in_sm_heft())
        return [process.expression for process in plan.concrete_processes]

    blind, unions = concrete(False), concrete(True)
    assert "u c~ > d s~" in blind and "u c~ > d s~" in unions
    assert len(unions) < len(blind)
