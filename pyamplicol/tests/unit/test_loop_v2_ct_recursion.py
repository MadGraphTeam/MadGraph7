# SPDX-License-Identifier: 0BSD
"""Counterterms inside the tree recursion (design §6.1, WP2.2).

Oracles:

* O6: per component, squared order and Laurent coefficient, the Born x CT
  partials of the tree runtime on the CT-lowered model equal the site closure
  (``loops.v2.ct_sites``) of the same rules at every fixture point;
* O1: the recursion plus the sites of the rules it leaves to site closure
  reproduces the frozen legacy UV_local and R2 components.

The CT program is an ordinary tree artifact of the lowered model, generated
through the public API in both tree lanes: the Rust recurrence and the
Python eager compiler.
"""

from __future__ import annotations

from decimal import Decimal, localcontext
from pathlib import Path

import pytest

from pyamplicol.loops.v2.ct_sites import LAURENT_POWERS
from tools.loop_reference import legacy_capture

from .test_loop_v2_counterterms import (
    CONTROLS,
    TOLERANCE,
    WORKING_DIGITS,
    _components,
    _sector_orders,
    counterterm_slices,
)

LANES = ("recurrence", "eager")
CT_COMPONENTS = ("uv_local", "r2")


@pytest.fixture(scope="module")
def ct_slice(loop_sm_compiled_model):
    return counterterm_slices(loop_sm_compiled_model)


@pytest.fixture(scope="module")
def lowered_ct_model(loop_sm_compiled_model, tmp_path_factory):
    """``forbidden |PDG| codes -> LoweredSource``: loop_sm with every
    lowerable CT rule in its tree model, once per veto; the prepared model
    source of the tree generator is built on first use."""

    from pyamplicol.loops.compiler import prepare_loop_request_model
    from pyamplicol.loops.v2.ct_recursion import lower_model_counterterms
    from pyamplicol.models.loading import load_compiled_model

    compiled = load_compiled_model(loop_sm_compiled_model)
    prepared = prepare_loop_request_model(
        compiled, fixed_parameters=legacy_capture.FIXED_PARAMETERS
    )
    built = {}

    def build(forbidden=()):
        key = tuple(sorted(forbidden))
        if key not in built:
            model = lower_model_counterterms(
                compiled,
                prepared.counterterms.rules,
                fixed_parameters=prepared.structure.fixed_parameters,
                forbidden=key,
            )
            built[key] = LoweredSource(model, tmp_path_factory)
        return built[key]

    return build


class LoweredSource:
    """A ``CountertermModel`` and, lazily, its prepared tree-model source."""

    def __init__(self, model, tmp_path_factory):
        self.model = model
        self._factory = tmp_path_factory
        self._source = None

    @property
    def source(self):
        from pyamplicol import ModelSource
        from pyamplicol.config import EvaluatorConfig

        if self._source is None:
            root = self._factory.mktemp("ct-lowered")
            path = self.model.lowered.compiled.write(
                root / "loop-sm-ct.pyamplicol-model.json"
            )
            bundle = root / "loop-sm-ct.pyamplicol-model"
            ModelSource.from_path(path).compile(
                cache_dir=root / "cache",
                prepared_output=bundle,
                evaluator=EvaluatorConfig(),
            )
            self._source = ModelSource.from_path(bundle)
        return self._source


def _recursion(ct, lowered):
    from pyamplicol.loops.v2.ct_recursion import CountertermRecursion

    return CountertermRecursion.build(
        lowered,
        ct.rules,
        ct.families,
        ct.data,
        ct.selection,
        ct.blocks.born_orders,
        forbidden=ct.vetoed,
    )


def _program(ct, recursion, source, artifact: Path, lane: str):
    from pyamplicol import Generator, Runtime
    from pyamplicol.config import (
        ColorConfig,
        EvaluatorConfig,
        EvaluatorOptimizationConfig,
        GenerationConfig,
        GenerationRelationDiscoveryConfig,
        GenerationValidationConfig,
        ModelConfig,
        ProcessConfig,
        RunConfig,
    )

    config = RunConfig(
        action="generate",
        model=ModelConfig(cache=False),
        color=ColorConfig(accuracy="full"),
        process=ProcessConfig(
            coupling_order_policy="explicit",
            max_coupling_orders=recursion.coupling_order_limits(),
        ),
        generation=GenerationConfig(
            workers=1,
            emit_api_bundle=False,
            relation_discovery=GenerationRelationDiscoveryConfig(mode="off"),
            validation=GenerationValidationConfig(
                enabled=False, samples=1, seed=7, post_build_validation=False
            ),
        ),
        evaluator=EvaluatorConfig(
            execution_mode=lane,
            optimization=EvaluatorOptimizationConfig(cores=1),
        ),
    )
    process = ct.control.process.split(" / ")[0]  # vetoes: in the lowered model
    Generator(config).generate(process, artifact, model=source)
    return Runtime.load(artifact)


def _model_parameters(ct, index: int) -> dict[str, str]:
    fixture = ct.fixture
    scale = Decimal(fixture["scales_squared"][index]).sqrt()
    parameters = {**legacy_capture.PARAMETERS, "MU_R": str(scale)}
    parameters.update(legacy_capture.FIXED_PARAMETERS)
    if ct.control.drell_yan:
        parameters["aEWM1"] = legacy_capture.DRELL_YAN_MODEL_VALUES[1]
    return parameters


def _inputs(ct, index: int, label: str):
    """Model parameters and momenta of one fixture point in a carrier."""

    parameters = _model_parameters(ct, index)
    point = ct.fixture["points"][index]
    if label == "double":
        model = {name: float(value) for name, value in parameters.items()}
        return model, [[float(x) for x in momentum] for momentum in point]
    model = {name: Decimal(value) for name, value in parameters.items()}
    return model, [list(momentum) for momentum in point]


def _set_parameters(recursion, runtime, model, **options) -> None:
    recursion.model.check_parameters(model)
    runtime.set_model_parameters({**model, **recursion.parameter_values(**options)})


def _recursion_components(
    ct, recursion, runtime, index: int, label: str = "double", *, poles: bool = True
):
    """Born and the recursion's CT partials at one fixture point.

    ``label`` is the fixture carrier: binary64, or 200 bits evaluated by the
    tree runtime at ``WORKING_DIGITS`` decimal digits with exact inputs.
    """

    model, momenta = _inputs(ct, index, label)
    _set_parameters(recursion, runtime, model, poles=poles)
    evaluation = runtime.evaluate_orders(
        [momenta], precision=None if label == "double" else WORKING_DIGITS[label]
    )
    born = {
        order: evaluation.for_order(order)[0]
        for order in evaluation.orders
        if not any(name.startswith("PYAMPLICOL_CT_") for name, _ in order)
    }
    parts = {key: values[0] for key, values in recursion.components(evaluation).items()}
    return born, parts


def _eager_slot_totals(ct, recursion, runtime, index: int):
    """Born and Born x CT per slot from totals of an eager (Python) artifact.

    Eager artifacts have no order partials. The CT couplings are linear in the
    slot switches, so with one slot switched to +1 and to -1 the totals
    differ by 4 Re(Born* CT_slot); 32 digits keep the subtraction exact to
    binary64.
    """

    model, momenta = _inputs(ct, index, "double")

    def total(**options):
        _set_parameters(recursion, runtime, model, **options)
        return _real(runtime.evaluate([momenta], precision=32)[0])

    born = total(slots=())
    parts = {}
    for slot in recursion.slots:
        plus = total(slots=(slot,))
        minus = total(slots=(slot,), sign=-1)
        parts[slot] = (plus - minus) / 2
    return born, parts


def _real(value) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, complex):
        if value.imag:
            raise AssertionError(f"Born x CT partial {value} is not real")
        return Decimal(value.real)
    return Decimal(value.real)


def _site_components(ct, label: str, index: int, rules) -> dict:
    """CT components of the sites of ``rules`` (empty when they have none)."""

    key = ("sites", frozenset(rules))
    if key not in ct._caches:
        ct._caches[key] = bool(ct.sites(closed_rules=rules).sites)
    if not ct._caches[key]:
        return {}
    return _components(ct, label, index, closed_rules=rules)[1]


def _site_value(rows, key) -> Decimal:
    return Decimal(rows[key].real) if key in rows else Decimal(0)


def _assert_equal_sites_and_legacy(ct, recursion, runtime, label: str) -> None:
    """Per (component, squared order, power): recursion = sites of its rules,
    and recursion + remaining sites = O1 (Decimal arithmetic at 120 digits,
    above every carrier)."""

    with localcontext() as context:
        context.prec = 120
        _compare_with_sites_and_legacy(ct, recursion, runtime, label)


def _compare_with_sites_and_legacy(ct, recursion, runtime, label: str) -> None:
    fixture = ct.fixture
    results = fixture["results"][label]
    sectors = _sector_orders(fixture, "virtual")
    born_sectors = _sector_orders(fixture, "born")
    tolerance = TOLERANCE[label]
    for index in range(len(fixture["points"])):
        born, parts = _recursion_components(ct, recursion, runtime, index, label)
        sites = _site_components(ct, label, index, recursion.rules)
        residual = _site_components(ct, label, index, recursion.residual)
        scale_born = max(abs(Decimal(row[0])) for row in results["born"][index])
        for b, order in enumerate(born_sectors):
            reference = Decimal(results["born"][index][b][0])
            assert abs(_real(born[order]) - reference) <= tolerance * abs(reference)
        assert {key[0] for key in parts} <= set(CT_COMPONENTS)
        for component in CT_COMPONENTS:
            for s, order in enumerate(sectors):
                for k, power in enumerate(LAURENT_POWERS):
                    key = (component, order, power)
                    reference = Decimal(results[component][index][s][k][0])
                    scale = abs(reference) or scale_born
                    mine = _real(parts.get(key, Decimal(0)))
                    site = _site_value(sites, key)
                    total = mine + _site_value(residual, key)
                    assert abs(mine - site) <= tolerance * scale, (key, index)
                    assert abs(total - reference) <= tolerance * scale, (key, index)


def _assert_eager_equal_sites_and_legacy(ct, recursion, runtime) -> None:
    """The same per slot, summed over squared orders (binary64)."""

    fixture = ct.fixture
    results = fixture["results"]["double"]
    sectors = _sector_orders(fixture, "virtual")
    tolerance = TOLERANCE["double"]
    for index in range(len(fixture["points"])):
        born, parts = _eager_slot_totals(ct, recursion, runtime, index)
        sites = _site_components(ct, "double", index, recursion.rules)
        residual = _site_components(ct, "double", index, recursion.residual)
        references = [Decimal(row[0]) for row in results["born"][index]]
        assert abs(born - sum(references)) <= tolerance * max(map(abs, references))
        for component in CT_COMPONENTS:
            for k, power in enumerate(LAURENT_POWERS):
                keys = [(component, order, power) for order in sectors]
                reference = sum(
                    Decimal(results[component][index][s][k][0])
                    for s in range(len(sectors))
                )
                scale = abs(reference) or max(map(abs, references))
                # Slots are (component, power, N_c grade); grade 0 without a
                # colour continuation.
                mine = sum(
                    (v for slot, v in parts.items() if slot[:2] == (component, power)),
                    Decimal(0),
                )
                site = sum((_site_value(sites, key) for key in keys), Decimal(0))
                rest = sum((_site_value(residual, key) for key in keys), Decimal(0))
                assert abs(mine - site) <= tolerance * scale, (component, power, index)
                assert abs(mine + rest - reference) <= tolerance * scale, (
                    component,
                    power,
                    index,
                )


@pytest.mark.loop_fast
def test_tree_colour_vocabulary():
    """CT colours lower into the tree colour flow or are refused whole."""

    from pyamplicol.models.ct_color_flow import (
        UnloweredCountertermColour,
        tree_colour_source,
    )

    assert tree_colour_source("T(3,2,1)", (-3, 3, 8)) == "UFO::{}::T(3,2,1)"
    assert tree_colour_source("Tr(1,2)", (8, 8, 1)) == ("1/2*UFO::{}::Identity(1,2)")
    assert tree_colour_source("f(-1,1,2)*f(3,4,-1)", (8, 8, 8, 8)) == (
        "UFO::{}::f(-1,1,2)*UFO::{}::f(3,4,-1)"
    )
    for source, representations in (
        ("d(-1,1,3)*d(-1,2,4)", (8, 8, 8, 8)),
        ("Identity(1,2)*Identity(3,4)", (8, 8, 8, 8)),
        ("d(1,2,3)", (8, 8, 8, 1)),
        ("Tr(1,2,3)", (8, 8, 8)),
        ("Tr(1,2)", (3, -3)),
    ):
        with pytest.raises(UnloweredCountertermColour):
            tree_colour_source(source, representations)


@pytest.mark.loop_fast
def test_exact_numeric_literals():
    from fractions import Fraction

    from pyamplicol.loops.v2.ct_recursion import _exact_literal

    assert _exact_literal("3.0") == 3
    assert _exact_literal("(4.0/3.0)") == Fraction(4, 3)
    assert _exact_literal("-1.5e-3") == Fraction(-3, 2000)
    assert _exact_literal("2*cmath.sqrt(aS)") is None


@pytest.mark.loop_fast
def test_lowered_couplings_are_exact_model_expressions(lowered_ct_model):
    """The lowered couplings are S_slot * sum M_rule * c_rule: the top-loop UV
    coefficients keep log(MT^2/MU_R^2) under the assumption MT != 0, the
    massless b (MB fixed to 0) contributes no finite UV term, and parameters
    that break an assumption are refused."""

    model = lowered_ct_model().model
    assert model.lowered.slot_orders == (
        "PYAMPLICOL_CT_R2_EPS0",
        "PYAMPLICOL_CT_UV_LOCAL_EPS0",
        "PYAMPLICOL_CT_UV_LOCAL_EPS1",
    )
    assert set(model.switches.values()) == {
        "PYAMPLICOL_CT_S_R2_EPS0",
        "PYAMPLICOL_CT_S_UV_LOCAL_EPS0",
        "PYAMPLICOL_CT_S_UV_LOCAL_EPS1",
    }
    assert model.assumptions == (
        "UFO::{}::MT",
        "UFO::{}::MT^2*UFO::{}::MU_R^(-2)",
    )
    couplings = {
        record["name"]: record["expression"]
        for record in model.lowered.compiled.model["couplings"]
    }
    lowered = {couplings[name] for name in model.lowered.couplings.values()}
    assert any("log(UFO::{}::MT^2*UFO::{}::MU_R^(-2))" in e for e in lowered)
    assert not any("MB" in e or "value_" in e for e in lowered)
    assert all("PYAMPLICOL_CT_S_" in e and "PYAMPLICOL_CT_M" in e for e in lowered)
    reasons = set(model.residual.values())
    assert not any("two-point" in reason for reason in reasons)
    # The ggVV and ggSS contacts lower (``compiler_contacts`` splits); the
    # tree vocabulary leaves the d-symbols and colour products to the
    # multi-trace state of the closure (``ct_closure``). The Levi-Civita
    # cells of the R2 of the Z with gluons cancel between the quark doublets
    # (``test_loop_v2_ct_cancellation``): only the g g g Z rules, whose vector
    # cell is a d-symbol, stay with the four-gluon R2.
    assert not any("contact decomposition" in reason for reason in reasons)
    assert reasons and all(
        "multi-trace colour state" in reason for reason in reasons
    ), reasons
    model.check_parameters({"MT": 173.0, "MU_R": 91.188})
    with pytest.raises(ValueError, match="vanish"):
        model.check_parameters({"MT": 0.0})


@pytest.mark.loop_fast
def test_constant_factors_are_decided_not_assumed():
    """Helper arguments free of model parameters are decided when the CT
    couplings are lowered: loop_qcd_qed_sm's reglog(2 pi) is log(2 pi) with no
    assumption, and only the parameter factors of reglog(4 pi MU_R^2/MW^2)
    are assumed non-zero. The branch a cond does not take adds nothing. The
    runtime check decides every assumption, builtin constants included, and
    still refuses parameters that make one vanish or singular."""

    import re
    from dataclasses import replace
    from types import SimpleNamespace

    from symbolica import E

    from pyamplicol._internal.physics.symbols import symbols
    from pyamplicol.loops.v2.ct_recursion import (
        CoefficientLowering,
        CountertermCouplingError,
        CountertermModel,
    )

    # The helper resolution alone (UFO function bindings value_0 = cond,
    # value_1 = reglog), without a compiled model.
    lowering = CoefficientLowering.__new__(CoefficientLowering)
    lowering._helpers = {0: "cond", 1: "reglog"}
    lowering._zero = E("0")
    lowering._log = symbols.qualified("symbolica::log")
    lowering.assumptions = set()
    helper = "pyamplicol_scalar_model_test::value_"
    assert lowering._resolve(E(f"{helper}1(2*𝜋)")) == E("log(2*𝜋)")
    assert not lowering.assumptions
    lowering._resolve(E(f"{helper}1(4*𝜋*UFO::{{}}::MU_R^2/UFO::{{}}::MW^2)"))
    assert lowering.assumptions == {"UFO::{}::MU_R^2*UFO::{}::MW^(-2)"}
    # cond(0, 1, reglog(MZ)): the untaken branch's argument is no assumption.
    assert lowering._resolve(E(f"{helper}0(0, 1, {helper}1(UFO::{{}}::MZ))")) == E("1")
    assert lowering.assumptions == {"UFO::{}::MU_R^2*UFO::{}::MW^(-2)"}
    assert lowering._resolve(E(f"{helper}1(0)")) == E("0")
    with pytest.raises(CountertermCouplingError, match="undecided"):
        lowering._resolve(E(f"{helper}1(sin(𝜋)*UFO::{{}}::MW)"))

    # The runtime check decides assumptions with builtin constants too (with
    # or without model parameters next to them).
    defaults = {"MT": (173.0, 0.0), "MU_R": (91.188, 0.0)}
    model = CountertermModel(
        lowered=SimpleNamespace(compiled=SimpleNamespace(parameter_defaults=defaults)),
        multiplicities={},
        switches={},
        assumptions=(
            "2*𝜋",
            "UFO::{}::MT",
            "UFO::{}::MT^(-2)*UFO::{}::MU_R^2*𝜋",
            "UFO::{}::MT^2*UFO::{}::MU_R^(-2)",
        ),
        residual={},
    )
    model.check_parameters({})
    model.check_parameters({"MT": 4.5, "MU_R": 1000.0})
    with pytest.raises(ValueError, match=re.escape("make UFO::{}::MT vanish")):
        model.check_parameters({"MT": 0.0})
    with pytest.raises(ValueError, match="singular"):
        replace(model, assumptions=model.assumptions[2:3]).check_parameters({"MT": 0.0})
    undecided = replace(model, assumptions=("sin(𝜋)*UFO::{}::MT",))
    with pytest.raises(ValueError, match="cannot decide"):
        undecided.check_parameters({})

    from pyamplicol.loops.v2.ct_recursion import decide_constant

    assert [
        decide_constant(E(value))
        for value in ("2*𝜋", "1/(4*𝜋)", "0", "0.0", "𝜋-𝜋", "0^(-2)", "sin(𝜋)")
    ] == ["non-zero"] * 2 + ["zero"] * 3 + ["singular", "undecided"]


@pytest.mark.loop_fast
def test_rules_split_between_recursion_and_sites(ct_slice, lowered_ct_model):
    """Only the fixed-SU(3) four-gluon R2 stays a site; every two-point,
    three-gluon, quark-gluon and four-gluon rule is in the recursion with its
    admitted alternatives."""

    ct = ct_slice("gg_gg")
    recursion = _recursion(ct, lowered_ct_model().model)
    assert set(recursion.rules) | set(recursion.residual) == set(ct.rules)
    for name, reason in recursion.residual.items():
        rule = ct.rules[name]
        assert {e.vertex_name for e in rule.entries} == {"V_R2RGA"}, reason
        assert "multi-trace" in reason
    assert any(len(rule.particle_pdgs) == 2 for rule in recursion.rules.values())
    # Pure-gluon rules are admitted with all their alternatives (every quark
    # flavour, the top and the gluon run in the loops of g g > g g); rules
    # with quark legs have no witness in this process.
    for name, rule in recursion.rules.items():
        gluons = set(rule.particle_pdgs) == {21}
        expected = tuple(range(len(rule.alternatives))) if gluons else ()
        assert recursion.admission[name] == expected, name
    assert recursion.slots == (("r2", 0, 0), ("uv_local", -1, 0), ("uv_local", 0, 0))
    assert recursion.coupling_order_limits() == {
        "QCD": 4,
        "QED": 0,
        "PYAMPLICOL_CT_INSERTION": 2,
    }


@pytest.mark.loop_fast
def test_two_point_insertions_are_clone_lines(lowered_ct_model):
    """Two-point CTs become clone lines: the gluon and the massive top keep
    P Pi P as a model-supplied propagator (the gluon's U(1)-subtraction
    auxiliary with it), a massless (Weyl) quark keeps its propagator and
    carries kappa on the copies of its antiparticle leg only, and the
    structure-constant colour of a copy follows its orientation's parity."""

    from pyamplicol.models.ct_line_insertions import _engine_denominator

    lowered = lowered_ct_model().model.lowered
    payload = lowered.compiled.model
    ir = lowered.compiled.ir
    clones = {s.particles[0].rstrip("~"): c for s, c in lowered.insertions.items()}
    propagators = {item["particle"]: item for item in payload["propagators"]}
    particles = {item["name"]: item for item in payload["particles"]}
    for field in ("G", "t"):
        line = propagators[clones[field]]
        assert line["pyamplicol_source"] == "model-supplied"
        # D^2 with the denominator the engine applies (the top's width).
        engine = _engine_denominator(propagators[field], particles[field])
        assert line["denominator"] == f"({engine})*({engine})"
        assert lowered.line_fields[clones[field]] == field
    assert "WT" in _engine_denominator(propagators["t"], particles["t"])
    assert not lowered.dirac_fields
    auxiliary = f"__pyamplicol_u1_subtraction_{clones['G']}"
    custom = {record.particle for record in ir.propagators if record.custom}
    assert auxiliary in custom and clones["G"] in custom

    weyl = clones["d"]
    assert propagators[weyl] == {
        **propagators["d"],
        "name": f"{weyl}_prop",
        "particle": weyl,
    }
    couplings = {item["name"]: item["expression"] for item in payload["couplings"]}
    for rule in payload["vertex_rules"]:
        legs = rule["particles"]
        if weyl not in legs and f"{weyl}~" not in legs:
            continue
        original = next(
            r
            for r in payload["vertex_rules"]
            if r["name"] == rule["name"].split("__", 1)[1]
        )
        scaled = [
            couplings[mine] != couplings[theirs]
            for row, rows in zip(rule["couplings"], original["couplings"], strict=True)
            for mine, theirs in zip(row, rows, strict=True)
            if mine is not None
        ]
        assert set(scaled) == {f"{weyl}~" in legs}, rule["name"]

    signs = {
        kernel.source_particle_legs: kernel.color_projection_coefficient
        for kernel in ir.oriented_kernels
        if kernel.vertex == f"{clones['G']}__V_3"
    }
    assert signs == {
        (1, 2, 0): (1.0, 0.0),
        (0, 2, 1): (-1.0, -0.0),
        (2, 0, 1): (1.0, 0.0),
    }


# The tree route (form 1) is a test of the lowering; the closure, the
# production form (design §6.1), runs on the same lowered couplings in the
# default and physics tiers (test_loop_v2_ct_closure, test_loop_v2_slice).
# Each form-1 id takes 5-7 minutes: all are nightly (HANDOFF §0.3 item 6).
@pytest.mark.loop_nightly
@pytest.mark.parametrize(
    ("name", "lane"), [(n, lane) for n in CONTROLS for lane in LANES]
)
def test_recursion_equals_site_closure(
    ct_slice, lowered_ct_model, loop_native, tmp_path, name, lane
):
    """O6: Born x CT per slot equals the sites of the same rules, and with the
    remaining sites the UV_local and R2 components equal O1 (binary64). The
    Rust recurrence lane is checked per squared order, the Python eager lane
    per slot summed over squared orders (it has no order partials)."""

    ct = ct_slice(name)
    lowered = lowered_ct_model(ct.vetoed)
    recursion = _recursion(ct, lowered.model)
    runtime = _program(ct, recursion, lowered.source, tmp_path / "artifact", lane)
    if lane == "eager":
        _assert_eager_equal_sites_and_legacy(ct, recursion, runtime)
    else:
        _assert_equal_sites_and_legacy(ct, recursion, runtime, "double")


# The tree runtime keeps the U(1)-subtraction colour weight 1/N_c of
# processes with several quark lines in binary64 (``GenericDAG.color_weight``,
# wave01-open-minors), so their 200-bit Born is exact to ~1e-17 only.
_BINARY64_U1_WEIGHT = pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="tree runtime U(1)-subtraction colour weight is binary64 "
    "(several quark lines): the 200-bit Born is exact to ~1e-17",
)


@pytest.mark.loop_nightly
@pytest.mark.parametrize(
    "name",
    (
        "gg_gg",
        "uux_ggg",
        "dy",
        pytest.param("uux_ddxg", marks=_BINARY64_U1_WEIGHT),
    ),
)
def test_recursion_equals_site_closure_at_200_bits(
    ct_slice, lowered_ct_model, loop_native, tmp_path, name
):
    """O6 and O1 in the 200-bit carrier, with exact inputs (recurrence lane)."""

    ct = ct_slice(name)
    lowered = lowered_ct_model(ct.vetoed)
    recursion = _recursion(ct, lowered.model)
    runtime = _program(
        ct, recursion, lowered.source, tmp_path / "artifact", "recurrence"
    )
    _assert_equal_sites_and_legacy(ct, recursion, runtime, "arbitrary-200")


@pytest.mark.loop_nightly
def test_finite_only_binds_no_pole_slot(ct_slice, lowered_ct_model, tmp_path):
    """Finite-only evaluation of the same form-1 program: pole slots are
    exactly zero and the finite slots are unchanged (the production closure:
    test_loop_v2_slice.py::test_finite_only_equals_the_finite_part)."""

    lowered = lowered_ct_model()
    ct = ct_slice("uux_ddxg")
    recursion = _recursion(ct, lowered.model)
    runtime = _program(
        ct, recursion, lowered.source, tmp_path / "artifact", "recurrence"
    )
    _, full = _recursion_components(ct, recursion, runtime, 0)
    _, finite = _recursion_components(ct, recursion, runtime, 0, poles=False)
    assert any(power and value for (_c, _o, power), value in full.items())
    for (component, order, power), value in finite.items():
        if power:
            assert value == 0
        else:
            assert value == full[component, order, power]
