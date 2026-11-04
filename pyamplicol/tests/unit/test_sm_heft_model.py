# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from decimal import Decimal, localcontext
from pathlib import Path

import pytest

from pyamplicol import (
    CompiledModel,
    Generator,
    ModelError,
    ModelSource,
    ProcessSet,
    Runtime,
)
from pyamplicol._internal.sm_heft import (
    CANONICAL_SM_HEFT_SOURCE,
    SM_HEFT_ALIASES,
    extend_sm_model_payload,
    packaged_sm_source_path,
    source_digest,
)
from pyamplicol.api.models import _compiled_model_payload
from pyamplicol.color.plan import build_color_plan
from pyamplicol.config import (
    ColorConfig,
    EvaluatorConfig,
    GenerationConfig,
    ModelConfig,
    ProcessConfig,
    RunConfig,
)
from pyamplicol.generation.dag_compiler import compile_generic_dag
from pyamplicol.models import CompiledUFOModel
from pyamplicol.processes.model import build_model_process_ir

_HEFT_ORDERS = {"QCD": 1, "QED": 0, "HIG": 1}
_PI = Decimal("3.14159265358979323846264338327950288419716939937510582097494")


@pytest.fixture(scope="module")
def compiled_sm_heft() -> CompiledModel:
    return ModelSource.built_in_sm_heft().compile(use_cache=False)


def _named_records(
    payload: Mapping[str, object], field: str
) -> dict[str, Mapping[str, object]]:
    records = payload[field]
    assert isinstance(records, list)
    return {
        str(record["name"]): record for record in records if isinstance(record, Mapping)
    }


def test_packaged_sm_heft_has_canonical_aliases() -> None:
    assert CANONICAL_SM_HEFT_SOURCE in SM_HEFT_ALIASES


def test_sm_heft_extension_is_independent_and_has_constant_coupling() -> None:
    base = json.loads(packaged_sm_source_path().read_text(encoding="utf-8"))
    extended = extend_sm_model_payload(base)

    assert extended is not base
    assert extended["name"] == CANONICAL_SM_HEFT_SOURCE
    assert "GH" not in _named_records(base, "parameters")
    assert not any(
        name.startswith("V_HEFT") for name in _named_records(base, "vertex_rules")
    )

    parameters = _named_records(extended, "parameters")
    couplings = _named_records(extended, "couplings")
    gh_expression = str(parameters["GH"]["expression"])
    assert "P(" not in gh_expression
    assert "Momentum(" not in gh_expression
    assert {name for name in couplings if name.startswith("GC_HEFT_")} == {
        "GC_HEFT_HGG",
        "GC_HEFT_HGGG",
        "GC_HEFT_HGGGG",
    }
    assert all(
        "P(" not in str(couplings[name]["expression"])
        for name in ("GC_HEFT_HGG", "GC_HEFT_HGGG", "GC_HEFT_HGGGG")
    )


def test_compiled_sm_heft_contains_all_scalar_gluon_vertices(
    compiled_sm_heft: CompiledModel,
) -> None:
    private = _compiled_model_payload(compiled_sm_heft)
    heft_terms = tuple(
        term for term in private.ir.vertex_terms if term.vertex.startswith("V_HEFT_")
    )

    assert compiled_sm_heft.name == CANONICAL_SM_HEFT_SOURCE
    assert compiled_sm_heft.source.kind == "built-in-sm-heft"
    assert compiled_sm_heft.supported
    assert compiled_sm_heft.capabilities.form_factor_count == 0
    assert compiled_sm_heft.capabilities.maximum_valence == 5
    observed_terms = [
        (term.vertex, term.color_index, term.lorentz_index) for term in heft_terms
    ]
    assert observed_terms == [
        ("V_HEFT_HGG", 0, 0),
        ("V_HEFT_HGGG", 0, 0),
        ("V_HEFT_HGGGG", 0, 0),
        ("V_HEFT_HGGGG", 1, 1),
        ("V_HEFT_HGGGG", 2, 2),
    ]
    contact_ids = {term.id for term in heft_terms if len(term.particles) > 3}
    lowered_ids = {
        term_id
        for kernel in private.ir.oriented_kernels
        if "::contact-" in kernel.vertex and "final" in kernel.vertex
        for term_id in (kernel.term_ids or (kernel.term_id,))
    }
    assert contact_ids <= lowered_ids


def test_sm_heft_extension_rejects_record_collisions() -> None:
    base = json.loads(packaged_sm_source_path().read_text(encoding="utf-8"))
    base["parameters"].append({"name": "GH"})

    with pytest.raises(ValueError, match="already defines HEFT parameters"):
        extend_sm_model_payload(base)


def test_default_restriction_is_the_plain_selector() -> None:
    plain = ModelSource.built_in_sm_heft()

    assert plain.restriction is None
    assert ModelSource.built_in_sm_heft(restriction="default") == plain
    for config in (
        ModelConfig(source="built-in-sm-heft"),
        ModelConfig(source="sm_heft", restriction="default"),
    ):
        assert ModelSource.from_config(config) == plain
    assert source_digest() == source_digest("default")


@pytest.mark.parametrize(
    "restriction",
    (
        "no_such_card",
        "restrict_no_b_mass.json",
        "../sm/restrict_no_b_mass",
        packaged_sm_source_path().with_name("restrict_no_b_mass.json"),
    ),
)
def test_sm_heft_restrictions_name_packaged_cards(restriction: str | Path) -> None:
    with pytest.raises(ModelError, match="restrict"):
        ModelSource(kind="built-in-sm-heft", restriction=restriction)


def _b_mass(compiled: CompiledModel) -> tuple[object, object]:
    parameters = {
        parameter.name: parameter
        for parameter in _compiled_model_payload(compiled).ir.parameters
    }
    return parameters["MB"].nature, parameters["MB"].value


def test_restricted_sm_heft_has_its_own_identity_and_a_massless_b(
    compiled_sm_heft: CompiledModel,
) -> None:
    restricted = ModelSource.from_config(
        ModelConfig(source="built-in-sm-heft", restriction="no_b_mass")
    ).compile(use_cache=False)

    assert restricted.name == CANONICAL_SM_HEFT_SOURCE
    assert restricted.source.kind == "built-in-sm-heft"
    assert compiled_sm_heft.source.restriction == "default"
    assert restricted.source.restriction == "no_b_mass"
    assert compiled_sm_heft.source.digest == source_digest()
    assert restricted.source.digest == source_digest("no_b_mass") != source_digest()
    assert _b_mass(compiled_sm_heft) == ("external", (4.7, 0.0))
    assert _b_mass(restricted) == ("internal", (0.0, 0.0))


def _heft_config(execution_mode: str = "compiled") -> RunConfig:
    return RunConfig(
        action="generate",
        color=ColorConfig(accuracy="full"),
        evaluator=EvaluatorConfig(backend="jit", execution_mode=execution_mode),
        generation=GenerationConfig(workers=1, emit_api_bundle=False),
        process=ProcessConfig(
            coupling_order_policy="explicit",
            max_coupling_orders={"QCD": 1, "QED": 0, "HIG": 1},
        ),
    )


def test_no_b_mass_heft_b_quark_scatters_like_a_light_quark(tmp_path: Path) -> None:
    """Without its mass and Yukawa coupling, the b couples to H only through
    the effective gluon vertex, like the d quark: b g > H b = d g > H d."""

    output = tmp_path / "heft_no_b_mass"
    Generator(_heft_config()).generate(
        ProcessSet.from_expressions(
            ("b g > h b", "d g > h d"), names=("bg_hb", "dg_hd")
        ),
        output,
        model=ModelSource.built_in_sm_heft(restriction="no_b_mass"),
    )
    manifest = json.loads((output / "artifact.json").read_text(encoding="utf-8"))
    assert manifest["model"]["restriction"] == "no_b_mass"
    assert manifest["model"]["content_sha256"] == source_digest("no_b_mass")
    # sqrt(s) = 1000 GeV, m_H = 125 GeV: |p| = 492.1875 along two exact directions.
    beams = (
        (Decimal(500), Decimal(0), Decimal(0), Decimal(500)),
        (Decimal(500), Decimal(0), Decimal(0), Decimal(-500)),
    )
    points = tuple(
        (
            *beams,
            (Decimal("507.8125"), -x, -y, -z),
            (Decimal("492.1875"), x, y, z),
        )
        for x, y, z in (
            (Decimal("295.3125"), Decimal(0), Decimal("393.75")),
            (Decimal("-236.25"), Decimal("177.1875"), Decimal("-393.75")),
            (Decimal(0), Decimal("492.1875"), Decimal(0)),
        )
    )
    b, d = (
        Runtime.load(output, process=name).evaluate(points, precision=61)
        for name in ("bg_hb", "dg_hd")
    )
    for b_value, d_value in zip(b, d, strict=True):
        assert abs(Decimal(b_value) / Decimal(d_value) - 1) <= Decimal("1e-30")


def test_prepared_execution_refuses_a_restricted_sm_heft(tmp_path: Path) -> None:
    with pytest.raises(ModelError, match="uses the default restriction"):
        Generator(_heft_config("recurrence")).generate(
            ProcessSet.from_expressions(("g g > h",), names=("gg_h",)),
            tmp_path / "heft_recurrence",
            model=ModelSource.built_in_sm_heft(restriction="no_b_mass"),
        )


def test_prepared_execution_refuses_a_restricted_built_in_sm(tmp_path: Path) -> None:
    """The built-in SM takes no restriction (as from a run card); the refusal
    does not point to the SM-HEFT's restricted bundles."""

    with pytest.raises(ModelError, match="only be applied to external models"):
        Generator(_heft_config("recurrence")).generate(
            ProcessSet.from_expressions(("g g > g g",), names=("gg_gg",)),
            tmp_path / "sm_recurrence",
            model=ModelSource(kind="built-in-sm", restriction="no_b_mass"),
        )


@pytest.mark.parametrize(
    ("expression", "accuracy"),
    (("d d~ > h g", "full"), ("g g > h", "lc"), ("g g > h g", "lc")),
)
def test_colour_singlet_merges_enter_the_dag_in_one_input_order(
    compiled_sm_heft: CompiledModel, expression: str, accuracy: str
) -> None:
    """The H g -> g identity vertex combines a gluon with a colour singlet;
    both input orders are one Feynman vertex and must not both be merged."""

    private = _compiled_model_payload(compiled_sm_heft)
    process = build_model_process_ir(expression, private.ir, color_accuracy=accuracy)
    dag = compile_generic_dag(
        process,
        model=CompiledUFOModel(private),
        color_plan=build_color_plan(process, color_accuracy=accuracy),
        max_coupling_orders=_HEFT_ORDERS,
    )
    merges = Counter(
        (frozenset((interaction.left_id, interaction.right_id)), interaction.result_id)
        for interaction in dag.interactions
    )
    assert merges
    assert max(merges.values()) == 1


def test_quark_initiated_heft_trees_match_the_effective_lagrangian(
    tmp_path: Path,
) -> None:
    """L = -(C/4) H G^a G^a: summed over colours and helicities,
    |M(g g > H)|^2 = 4 C^2 m_H^4, |M(q q~ > H g)|^2 = 4 g^2 C^2 (t^2 + u^2) / s
    and by crossing |M(q~ g > H q~)|^2 = 4 g^2 C^2 (s^2 + u^2) / (-t), with
    t = (p_q~,in - p_q~,out)^2. Ratios to g g > H cancel C."""

    output = tmp_path / "heft_quarks"
    config = RunConfig(
        action="generate",
        color=ColorConfig(accuracy="full"),
        evaluator=EvaluatorConfig(backend="jit", execution_mode="compiled"),
        generation=GenerationConfig(workers=1, emit_api_bundle=False),
        process=ProcessConfig(
            coupling_order_policy="explicit", max_coupling_orders=_HEFT_ORDERS
        ),
    )
    Generator(config).generate(
        ProcessSet.from_expressions(
            ("g g > h", "d d~ > h g", "d~ g > h d~"),
            names=("gg_h", "ddx_hg", "dxg_hdx"),
        ),
        output,
        model=ModelSource.built_in_sm_heft(),
    )
    alpha_s = Decimal("0.125")

    def evaluate(name: str, point: tuple[tuple[Decimal, ...], ...]) -> Decimal:
        runtime = Runtime.load(output, process=name, model_parameters={"aS": 0.125})
        return Decimal(str(runtime.evaluate((point,), precision=40)[0]))

    def dot(p: tuple[Decimal, ...], q: tuple[Decimal, ...]) -> Decimal:
        return p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3]

    half_mass = Decimal("62.5")
    zero = Decimal(0)
    gg_h = evaluate(
        "gg_h",
        (
            (half_mass, zero, zero, half_mass),
            (half_mass, zero, zero, -half_mass),
            (2 * half_mass, zero, zero, zero),
        ),
    )
    # sqrt(s) = 1000 GeV, m_H = 125 GeV: |p| = 492.1875 along two exact directions.
    beams = (
        (Decimal(500), zero, zero, Decimal(500)),
        (Decimal(500), zero, zero, Decimal(-500)),
    )
    for x, y, z in (
        (Decimal("295.3125"), zero, Decimal("393.75")),
        (Decimal("-236.25"), Decimal("177.1875"), Decimal("-393.75")),
    ):
        point = (
            *beams,
            (Decimal("507.8125"), -x, -y, -z),
            (Decimal("492.1875"), x, y, z),
        )
        with localcontext() as context:
            context.prec = 50
            s = 2 * dot(point[0], point[1])
            t = -2 * dot(point[0], point[3])
            u = -2 * dot(point[1], point[3])
            g_squared = 4 * _PI * alpha_s
            m_h4 = (2 * half_mass) ** 4
            # Averages 1/256 (g g), 1/36 (q q~) and 1/96 (q~ g).
            expected = {
                "ddx_hg": Decimal(256) / 36 * g_squared * (t * t + u * u) / s / m_h4,
                "dxg_hdx": Decimal(256) / 96 * g_squared * (s * s + u * u) / -t / m_h4,
            }
            for name, ratio in expected.items():
                observed = evaluate(name, point) / gg_h
                assert abs(observed / ratio - 1) <= Decimal("1e-30"), name


def test_quark_initiated_heft_trees_agree_across_execution_modes(
    tmp_path: Path,
) -> None:
    """The compiled tree (checked against the effective Lagrangian above),
    the eager lane and the recurrence lane give one value for d d~ > h g and
    d~ g > h d~. Before the single-input-order rule held on the reflection
    path, the eager lane refused these processes (its DAG attached the
    H g -> g merge twice: overlapping attachment destinations) and the
    compiled one was 4 times too large; recurrence was right."""

    expressions = ("d d~ > h g", "d~ g > h d~")
    names = ("ddx_hg", "dxg_hdx")
    point = (
        (500.0, 0.0, 0.0, 500.0),
        (500.0, 0.0, 0.0, -500.0),
        (507.8125, -295.3125, 0.0, -393.75),
        (492.1875, 295.3125, 0.0, 393.75),
    )
    values: dict[str, dict[str, complex]] = {}
    for mode in ("compiled", "eager", "recurrence"):
        output = tmp_path / mode
        config = RunConfig(
            action="generate",
            color=ColorConfig(accuracy="full"),
            evaluator=EvaluatorConfig(backend="jit", execution_mode=mode),
            generation=GenerationConfig(workers=1, emit_api_bundle=False),
            process=ProcessConfig(
                coupling_order_policy="explicit", max_coupling_orders=_HEFT_ORDERS
            ),
        )
        Generator(config).generate(
            ProcessSet.from_expressions(expressions, names=names),
            output,
            model=ModelSource.built_in_sm_heft(),
        )
        values[mode] = {
            name: complex(Runtime.load(output, process=name).evaluate((point,))[0])
            for name in names
        }
    for mode in ("eager", "recurrence"):
        for name in names:
            compiled = values["compiled"][name]
            assert abs(values[mode][name] - compiled) <= 1e-12 * abs(compiled), (
                mode,
                name,
            )
