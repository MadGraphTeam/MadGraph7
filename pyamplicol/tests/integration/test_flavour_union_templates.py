# SPDX-License-Identifier: 0BSD
"""Flavour-union members evaluated through one pattern-resolved template.

A template is the all-``d`` subprocess of a union generated with one amplitude
per fermion pairing rule (pattern). A member selects the patterns that join
equal flavours and sums them, with sign +1, before squaring. Every member must
match its separately generated program per helicity and LC flow.
"""

from __future__ import annotations

import importlib.util
import json
import math
import shutil
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from fractions import Fraction
from functools import cache
from pathlib import Path

import pytest

from pyamplicol import Generator, ProcessSet, Runtime
from pyamplicol.api.errors import CompatibilityError
from pyamplicol.artifacts.process_groups import (
    initial_process_groups,
    write_process_groups,
)
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
from pyamplicol.generation.process_grouping import GroupingBasis, GroupMember
from pyamplicol.generation.service import flavour_union_template_generation
from pyamplicol.models.builtin.validation import generic_validation_point

# Template, member, member relation (template slot -> member leg).
_MEMBERS = (
    ("d d~ > d d d~ d~", "d d~ > d u d~ u~", (0, 1, 2, 3, 4, 5)),
    ("d d~ > d d d~ d~", "d d~ > u s u~ s~", (0, 1, 2, 3, 4, 5)),
    ("d d~ > d d d~ d~", "d d~ > u u u~ u~", (0, 1, 2, 3, 4, 5)),
    ("d d~ > d d d~ d~", "u u~ > u u u~ u~", (0, 1, 2, 3, 4, 5)),
    ("d d > d d g g", "d u > d u g g", (0, 1, 2, 3, 4, 5)),
    ("d d > d d g g", "d u > u d g g", (0, 1, 3, 2, 4, 5)),
    ("d d > d d g g", "u u > u u g g", (0, 1, 2, 3, 4, 5)),
    ("d d~ > d d~ g g", "d d~ > u u~ g g", (0, 1, 2, 3, 4, 5)),
    ("d d~ > d d~ g g", "d u~ > d u~ g g", (0, 1, 2, 3, 4, 5)),
)
_TEMPLATES = tuple(dict.fromkeys(template for template, _, _ in _MEMBERS))
_REFERENCES = _TEMPLATES + tuple(member for _, member, _ in _MEMBERS)


def _require_native_runtime() -> None:
    if importlib.util.find_spec("pyamplicol._rusticol") is None:
        pytest.skip("the Rusticol extension has not been built")
    if importlib.util.find_spec("symbolica") is None:
        pytest.skip("Symbolica is unavailable")


def _config() -> RunConfig:
    return RunConfig(
        action="generate",
        color=ColorConfig(accuracy="lc"),
        process=ProcessConfig(max_coupling_orders={"QED": 0}),
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


def _member_id(expression: str) -> str:
    initial, final = expression.split(">")
    return "union_" + "_".join(
        token.replace("~", "bar") for token in (*initial.split(), "to", *final.split())
    )


def _pdgs(expression: str) -> tuple[int, ...]:
    return tuple(int(particle.pdg) for particle in generic_validation_point(expression))


@cache
def _pairing_rules(template: str) -> dict[int, tuple[tuple[int, int], ...]]:
    """Pairing rules (fundamental slot, antifundamental slot) of the template."""

    from pyamplicol.generation.recurrence_fermion_pairing import (
        build_recurrence_fermion_pairing_catalog_v1,
    )
    from pyamplicol.models import BuiltinSMModel
    from pyamplicol.models.builtin.process_ir import build_process_ir
    from pyamplicol.models.prepared_catalog import build_prepared_kernel_catalog
    from pyamplicol.models.recurrence_catalog_builder import (
        build_recurrence_template_catalog,
    )

    model = BuiltinSMModel()
    catalog = build_recurrence_template_catalog(
        model,
        build_prepared_kernel_catalog(model),
        compiled_model_digest="a" * 64,
        prepared_kernel_pack_digest="b" * 64,
    )
    pairing = build_recurrence_fermion_pairing_catalog_v1(
        build_process_ir(template, color_accuracy="lc"),
        catalog.current_states,
        quantum_flows=catalog.quantum_flows,
    )
    return {rule.rule_id: tuple(rule.endpoint_pairings) for rule in pairing.rules}


def _member_patterns(
    template: str, member: str, permutation: Sequence[int]
) -> tuple[int, ...]:
    """Template rules whose every endpoint pair joins equal member flavours."""

    flavours = [abs(pdg) for pdg in _pdgs(member)]
    return tuple(
        rule_id
        for rule_id, pairs in sorted(_pairing_rules(template).items())
        if all(
            flavours[permutation[left]] == flavours[permutation[right]]
            for left, right in pairs
        )
    )


def _symmetry(expression: str) -> int:
    final = expression.split(">")[1].split()
    return math.prod(math.factorial(count) for count in Counter(final).values())


def _attach_members(artifact: Path) -> None:
    raw = json.loads((artifact / "artifact.json").read_text(encoding="utf-8"))
    ids = {str(process["expression"]): str(process["id"]) for process in raw["processes"]}
    members: dict[str, list[GroupMember]] = {}
    for template, member, permutation in _MEMBERS:
        patterns = _member_patterns(template, member, permutation)
        members.setdefault(ids[template], []).append(
            GroupMember(
                process_id=_member_id(member),
                expression=member,
                pdgs=_pdgs(member),
                initial_count=2,
                factor=Fraction(_symmetry(template), _symmetry(member)),
                permutation=permutation,
                basis=GroupingBasis.FLAVOUR_UNION,
                generated=False,
                patterns=patterns,
                pattern_signs=(1,) * len(patterns),
            )
        )
    write_process_groups(
        artifact,
        initial_process_groups(
            (
                (str(process["id"]), str(process["expression"]), process["external_pdgs"])
                for process in raw["processes"]
            ),
            members,
        ),
    )


@pytest.fixture(scope="module")
def union_artifacts(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[dict[str, Path]]:
    _require_native_runtime()
    root = tmp_path_factory.mktemp("flavour-union-templates")
    artifacts = {"template": root / "template", "reference": root / "reference"}
    try:
        with flavour_union_template_generation():
            Generator(_config()).generate(
                ProcessSet.from_expressions(_TEMPLATES), artifacts["template"]
            )
        _attach_members(artifacts["template"])
        Generator(_config()).generate(
            ProcessSet.from_expressions(_REFERENCES), artifacts["reference"]
        )
        yield artifacts
    finally:
        shutil.rmtree(root, ignore_errors=True)


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
) -> None:
    assert actual.keys() == expected.keys()
    scale = max(abs(value) for value in expected.values())
    assert scale > 0.0
    for key, value in expected.items():
        assert actual[key] == pytest.approx(value, rel=1.0e-12, abs=1.0e-12 * scale), key


def test_template_pattern_counts_follow_the_single_flavour_catalog() -> None:
    assert len(_pairing_rules("d d~ > d d d~ d~")) == 6
    assert len(_pairing_rules("d d > d d g g")) == 2
    assert len(_member_patterns("d d~ > d d d~ d~", "d d~ > u s u~ s~", range(6))) == 1
    assert len(_member_patterns("d d~ > d d d~ d~", "d d~ > d u d~ u~", range(6))) == 2
    assert len(_member_patterns("d d~ > d d d~ d~", "d d~ > u u u~ u~", range(6))) == 2
    assert len(_member_patterns("d d~ > d d d~ d~", "u u~ > u u u~ u~", range(6))) == 6


@pytest.mark.parametrize("template", _TEMPLATES)
def test_template_without_member_matches_the_normal_program(
    union_artifacts: dict[str, Path], template: str
) -> None:
    resolved = Runtime.load(union_artifacts["template"], process=template)
    normal = Runtime.load(union_artifacts["reference"], process=template)
    point = _point(template)
    _assert_flows_match(_flow_values(resolved, point), _flow_values(normal, point))
    assert resolved.evaluate((point,) * 3) == pytest.approx(
        normal.evaluate((point,) * 3), rel=1.0e-12
    )


@pytest.mark.parametrize(("template", "member", "permutation"), _MEMBERS)
def test_member_matches_its_separately_generated_program(
    union_artifacts: dict[str, Path],
    template: str,
    member: str,
    permutation: tuple[int, ...],
) -> None:
    routed = Runtime.load(union_artifacts["template"], process=_member_id(member))
    reference = Runtime.load(union_artifacts["reference"], process=member)
    assert routed.physics.process == member
    assert routed.process_group_factor == Fraction(_symmetry(template), _symmetry(member))
    point = _point(member)
    _assert_flows_match(_flow_values(routed, point), _flow_values(reference, point))
    assert routed.evaluate((point,) * 2) == pytest.approx(
        reference.evaluate((point,) * 2), rel=1.0e-12
    )
    helicity = next(h for h in routed.physics.helicities if h.computed)
    assert routed.evaluate((point,), helicities=(helicity.id,)) == pytest.approx(
        reference.evaluate((point,), helicities=(helicity.id,)), rel=1.0e-12
    )


def test_high_precision_refuses_pattern_resolved_templates(
    union_artifacts: dict[str, Path],
) -> None:
    routed = Runtime.load(
        union_artifacts["template"], process=_member_id("d u > d u g g")
    )
    with pytest.raises(CompatibilityError, match="pattern-resolved"):
        routed.evaluate((_point("d u > d u g g"),), precision=32)
