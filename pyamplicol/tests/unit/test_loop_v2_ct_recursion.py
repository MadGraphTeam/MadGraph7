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
    """``vetoed |PDG| codes -> LoweredSource``: loop_sm with every
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

    def build(veto=()):
        key = tuple(sorted(veto))
        if key not in built:
            model = lower_model_counterterms(
                compiled,
                prepared.counterterms.rules,
                fixed_parameters=prepared.structure.fixed_parameters,
                veto=key,
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
        lowered, ct.rules, ct.selection, ct.blocks.born_orders
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
def test_lowered_model_compile_is_cached_by_content(
    loop_sm_compiled_model, tmp_path, monkeypatch
):
    """The compile of a CT-lowered payload (most of the lowering) is cached
    by content like a compiled model: a warm entry gives the identical IR
    without compiling, a damaged entry is recompiled, and another payload or
    kernel variant is another entry."""

    import json

    import pyamplicol.models.compiler_entry as compiler_entry
    from pyamplicol.models.loading import (
        COUNTERTERM_IR_CACHE_SUFFIX,
        compile_counterterm_model_ir,
        load_compiled_model,
    )

    model = load_compiled_model(loop_sm_compiled_model).model
    payload = json.loads(json.dumps(model))
    keep = {"G", "u", "u~"}
    payload["vertex_rules"] = [
        rule for rule in payload["vertex_rules"] if set(rule["particles"]) <= keep
    ]

    def compile_ir(variant="full", data=payload):
        return compile_counterterm_model_ir(
            data,
            tree_kernel_variant=variant,
            u1_projected_vertices=frozenset(),
            cache_dir=tmp_path,
        )

    cold = compile_ir()
    (entry,) = tmp_path.glob(f"*{COUNTERTERM_IR_CACHE_SUFFIX}")
    assert cold.oriented_kernels and cold.tree_kernel_variant == "full"

    calls = []
    original = compiler_entry.compile_ufo_model_ir

    def counted(*args, **kwargs):
        calls.append(kwargs.get("tree_kernel_variant"))
        return original(*args, **kwargs)

    monkeypatch.setattr(compiler_entry, "compile_ufo_model_ir", counted)
    assert compile_ir() == cold and not calls
    entry.write_text("{", encoding="utf-8")
    assert compile_ir() == cold and calls == ["full"]
    assert compile_ir() == cold and calls == ["full"]
    assert compile_ir("transverse").tree_kernel_variant == "transverse"
    fewer = {**payload, "vertex_rules": payload["vertex_rules"][:-1]}
    assert len(compile_ir(data=fewer).vertex_terms) < len(cold.vertex_terms)
    assert calls == ["full", "transverse", "full"]
    assert len(list(tmp_path.glob(f"*{COUNTERTERM_IR_CACHE_SUFFIX}"))) == 3


@pytest.mark.loop_fast
def test_lowered_model_cache_keeps_the_most_recently_used(tmp_path):
    """The lowering cache is bounded: beyond ``COUNTERTERM_IR_CACHE_KEPT``
    entries the least recently used go, each with its Weyl table, and
    nothing else in the directory is touched."""

    import os

    from pyamplicol.models.loading import (
        COUNTERTERM_IR_CACHE_SUFFIX,
        COUNTERTERM_WEYL_CACHE_SUFFIX,
        _evict_counterterm_entries,
    )

    for age in range(5):
        for suffix in (COUNTERTERM_IR_CACHE_SUFFIX, COUNTERTERM_WEYL_CACHE_SUFFIX):
            path = tmp_path / f"k{age}{suffix}"
            path.write_text("{}", encoding="utf-8")
            os.utime(path, ns=(10**18 - age * 10**9,) * 2)  # k0 newest
    other = tmp_path / "loop-sm.json.pyAmplicol-model.json"
    other.write_text("{}", encoding="utf-8")
    os.utime(tmp_path / f"k4{COUNTERTERM_IR_CACHE_SUFFIX}", ns=(10**18 + 1,) * 2)
    _evict_counterterm_entries(tmp_path, kept=3)
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        [
            other.name,
            *(
                f"k{k}{s}"
                for k in (0, 1, 4)
                for s in (COUNTERTERM_IR_CACHE_SUFFIX, COUNTERTERM_WEYL_CACHE_SUFFIX)
            ),
        ]
    )


@pytest.mark.loop_fast
def test_lowered_model_weyl_support_is_cached(
    loop_sm_compiled_model, tmp_path, monkeypatch
):
    """The Weyl-projection support of a lowered model (every decision
    ``allowed_quantum_flows`` can ask) is probed once and kept next to its
    lowering-cache entry: the warm table is the probed one and a tree model
    seeded with it answers like an unseeded one without a probe; a damaged
    table, or other parameter defaults, is probed again."""

    from dataclasses import replace
    from types import SimpleNamespace

    from pyamplicol.models.external import CompiledUFOModel
    from pyamplicol.models.external_kernels import ExternalModelKernelMixin
    from pyamplicol.models.loading import (
        COUNTERTERM_IR_CACHE_SUFFIX,
        COUNTERTERM_WEYL_CACHE_SUFFIX,
        CountertermLoweredModel,
        counterterm_weyl_support,
        load_compiled_model,
    )

    compiled = load_compiled_model(loop_sm_compiled_model)
    entry = (tmp_path / f"key{COUNTERTERM_IR_CACHE_SUFFIX}", "key")

    def flows(model):
        def index(pdg, chirality):
            return SimpleNamespace(
                particle_id=pdg,
                chirality=chirality,
                flavour_flow=(pdg,),
                coupling_orders=(),
            )

        def chiralities(pdg):
            return (-1, 0, 1) if model.is_chiral_eligible(pdg) else (0,)

        return [
            model.allowed_quantum_flows(vertex, index(left, a), index(right, b))
            for vertex in model.vertices
            for left, right, _ in (vertex.particles,)
            for a in chiralities(left)
            for b in chiralities(right)
        ]

    unseeded = CompiledUFOModel(compiled)
    expected = flows(unseeded)
    probed = unseeded.weyl_projection_support()
    # Discriminating: some chiral projections vanish, others do not.
    assert any(probed.values()) and not all(probed.values())
    assert counterterm_weyl_support(CompiledUFOModel(compiled), entry) == probed
    (table,) = tmp_path.glob(f"*{COUNTERTERM_WEYL_CACHE_SUFFIX}")

    probes = []
    original = ExternalModelKernelMixin._projected_kernel_components

    def counted(self, *args, **kwargs):
        probes.append(kwargs.get("result_chirality"))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(
        ExternalModelKernelMixin, "_projected_kernel_components", counted
    )
    assert counterterm_weyl_support(CompiledUFOModel(compiled), entry) == probed
    lowered = CountertermLoweredModel(compiled, {}, (), cache_entry=entry)
    seeded = lowered.tree_model()
    assert flows(seeded) == expected and not probes
    assert flows(lowered.tree_model()) == expected and not probes

    table.write_text("{", encoding="utf-8")
    assert counterterm_weyl_support(CompiledUFOModel(compiled), entry) == probed
    assert probes
    probes.clear()
    assert counterterm_weyl_support(CompiledUFOModel(compiled), entry) == probed
    assert not probes
    name, (real, imag) = next(
        (name, value)
        for name, value in sorted(compiled.parameter_defaults.items())
        if value[0]
    )
    other = replace(
        compiled,
        parameter_defaults={**compiled.parameter_defaults, name: (2 * real, imag)},
    )
    counterterm_weyl_support(CompiledUFOModel(other), entry)
    assert probes


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

    # The helper resolution alone (UFO-standard helpers: scalar-library
    # records value_0 = cond, value_1 = reglog), without a compiled model.
    lowering = CoefficientLowering.__new__(CoefficientLowering)
    lowering._helpers = lowering._standard = {0: "cond", 1: "reglog"}
    lowering._fixed, lowering._substitution = [], {}
    lowering._zero = E("0")
    lowering._log, lowering._conj = symbols.qualified(
        "symbolica::log", "symbolica::conj"
    )
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
    """Only the fixed-SU(3) four-gluon R2 stays out of the tree-route program
    (its multi-trace colour); every two-point, three-gluon, quark-gluon and
    four-gluon rule is in the recursion with every loop-content alternative
    (the plain rule: no veto, no matching to the process's loops)."""

    ct = ct_slice("gg_gg")
    recursion = _recursion(ct, lowered_ct_model().model)
    assert set(recursion.rules) | set(recursion.residual) == set(ct.rules)
    for name, reason in recursion.residual.items():
        rule = ct.rules[name]
        assert {e.vertex_name for e in rule.entries} == {"V_R2RGA"}, reason
        assert "multi-trace" in reason
    assert any(len(rule.particle_pdgs) == 2 for rule in recursion.rules.values())
    # Every rule is admitted with all its alternatives, also the rules with
    # quark legs, which no tree of g g > g g holds.
    for name, rule in recursion.rules.items():
        assert recursion.admission[name] == tuple(range(len(rule.alternatives)))
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


# Vertices the plain rule admits that no tree of the request holds (by their
# UFO vertex names): REQUIREMENTS_ADDENDUM J's inert rules when unlowerable.
UNREACHABLE_VERTICES = {
    "gg_gg": {"V_R2GUU", "V_R2UU"},  # g g > g g: no quark current
    # u u~ > e- e+ / z: no gluon current, and the veto removes the Z.
    "dy": {"V_R2GG", "V_R2uuZ"},
}


@pytest.mark.loop_fast
@pytest.mark.parametrize("name", ("gg_gg", "uux_ddxg", "dy"))
def test_insertion_reachability_is_sound_against_the_site_oracle(ct_slice, name):
    """The reachability that makes an unlowerable CT vertex inert
    (``ct_closure.InsertionReachability``, REQUIREMENTS_ADDENDUM J) is sound:
    every rule the site oracle closes on the slice's exact tree currents
    (every partition of the legs, the plain admission, no witness) enters a
    tree for it. It also discriminates: the quark vertices enter no tree of
    g g > g g, nor the gluon and (vetoed) Z vertices a tree of Drell-Yan.
    Were every admitted rule unlowerable, the reachable ones would be refused
    and the others inert (``CountertermRecursion.with_inert``)."""

    from pyamplicol.loops.v2.blocks import vetoed_particle_ids
    from pyamplicol.loops.v2.ct_closure import (
        InsertionReachability,
        program_order_limits,
    )
    from pyamplicol.loops.v2.ct_recursion import (
        CountertermRecursion,
        admitted_alternatives,
    )
    from pyamplicol.loops.v2.ct_sites import generate_counterterm_sites
    from pyamplicol.models.external import CompiledUFOModel
    from pyamplicol.models.loading import COUNTERTERM_INSERTION_ORDER
    from pyamplicol.processes.model import build_model_process_ir

    ct = ct_slice(name)
    text = ct.control.process.split(" / ")[0]
    limits = program_order_limits(
        ct.compiled, ct.blocks.born_orders, ct.selection.loop_orders
    )
    reachability = InsertionReachability.build(
        CompiledUFOModel(ct.compiled),
        build_model_process_ir(text, ct.compiled.ir, color_accuracy="full"),
        {**limits, COUNTERTERM_INSERTION_ORDER: 2},
        ignored_particle_ids=vetoed_particle_ids(ct.vetoed),
    )
    admission = {
        rule_name: admitted_alternatives(rule, ct.vetoed, ct.selection.corrections)
        for rule_name, rule in ct.rules.items()
    }
    sites = generate_counterterm_sites(
        None,
        ct.table,
        ct.rules,
        ct.data,
        ct.selection,
        forbidden=ct.vetoed,
        whole_rules=True,
        admission=admission,
    ).sites
    closed = {site.rule_name for site in sites}
    assert closed
    reachable = {
        rule_name
        for rule_name, rule in ct.rules.items()
        if admission[rule_name] and reachability.admits(rule)
    }
    assert closed <= reachable, sorted(closed - reachable)
    unreachable = {
        entry.vertex_name
        for rule_name, rule in ct.rules.items()
        if admission[rule_name] and rule_name not in reachable
        for entry in rule.entries
    }
    assert UNREACHABLE_VERTICES.get(name, set()) <= unreachable, sorted(unreachable)
    assert (
        not {entry.vertex_name for site in sites for entry in site.rule.entries}
        & unreachable
    )
    # The inert rule itself, as if every admitted rule could not be lowered:
    # the reachable ones stay refused, the others become inert.
    admitted = {rule_name for rule_name in ct.rules if admission[rule_name]}
    every = CountertermRecursion(
        None, {}, {}, {}, dict.fromkeys(admitted, "not lowered (test)"), (), ()
    )
    split = every.with_inert(ct.rules, reachability.admits)
    assert set(split.residual) == reachable
    assert set(split.inert) == admitted - reachable
    assert all("enters no tree" in reason for reason in split.inert.values())


@pytest.mark.loop_fast
def test_insertion_reachability_tells_a_vertex_from_its_conjugate(
    loop_sm_compiled_model,
):
    """Charged currents tell a CT vertex from its conjugate
    (``ct_closure.InsertionReachability``). At a CT root (leg 1 alone) the
    current entering a leg of field f is anti(f); inside a current the tree
    engine puts the current of particle p on the leg of p. In u d~ > e+ ve
    the R2 of [d~ u W-] enters at the root and the lepton vertex [e+ ve W-]
    inside the W+ current of legs 3 and 4; their conjugates [u~ d W+] and
    [ve~ e- W+] and the vertices of other quark flavours enter no tree. With
    a gluon, [u~ d W+] enters the current of legs 1 and 5 as well. A
    two-point vertex enters internal lines only (an external line takes the
    field factor): in u u~ > d d~ the s-channel gluon holds [g g], and no
    tree has an internal quark line for [u~ u] or [d~ d]."""

    from types import SimpleNamespace

    from pyamplicol.loops.compiler import prepare_loop_request_model
    from pyamplicol.loops.v2.ct_closure import (
        InsertionReachability,
        program_order_limits,
    )
    from pyamplicol.models.external import CompiledUFOModel
    from pyamplicol.models.loading import (
        COUNTERTERM_INSERTION_ORDER,
        load_compiled_model,
    )
    from pyamplicol.processes.model import build_model_process_ir

    compiled = load_compiled_model(loop_sm_compiled_model)
    rules = prepare_loop_request_model(compiled).counterterms.rules
    model = CompiledUFOModel(compiled)

    def reachability(text, born, loop):
        limits = program_order_limits(compiled, (born,), (loop,))
        return InsertionReachability.build(
            model,
            build_model_process_ir(text, compiled.ir, color_accuracy="full"),
            {**limits, COUNTERTERM_INSERTION_ORDER: 2},
        )

    def admitted(reach, *pdgs):
        """Whether the loop_sm rules of exactly these fields are admitted
        (all of them alike; at least one exists)."""

        found = {
            reach.admits(rule)
            for rule in rules.values()
            if sorted(rule.particle_pdgs) == sorted(pdgs)
        }
        assert len(found) == 1, (pdgs, found)
        return found.pop()

    def vertex(*pdgs):
        return SimpleNamespace(particle_pdgs=pdgs, entries=(), order=(("QED", 1),))

    charged = reachability("u d~ > e+ ve", (("QED", 2),), (("QCD", 2), ("QED", 2)))
    assert admitted(charged, -1, 2, -24)  # V_R2dxuW, at the root
    assert not admitted(charged, -2, 1, 24)  # V_R2uxdW, its conjugate
    assert not admitted(charged, -3, 2, -24)  # V_R2sxuW: no s quark
    assert not admitted(charged, -1, 4, -24)  # V_R2dxcW: no c quark
    assert charged.admits(vertex(-11, 12, -24))  # inside the W+ of e+ ve
    assert not charged.admits(vertex(-12, 11, 24))

    gluon = reachability(
        "u d~ > e+ ve g", (("QCD", 1), ("QED", 2)), (("QCD", 3), ("QED", 2))
    )
    assert admitted(gluon, -1, 2, -24)  # at the root
    assert admitted(gluon, -2, 1, 24)  # inside the current of u~ and g
    assert not admitted(gluon, -3, 2, -24)

    neutral = reachability("u u~ > d d~", (("QCD", 2),), (("QCD", 4),))
    assert admitted(neutral, 21, 21)
    assert not admitted(neutral, -2, 2)
    assert not admitted(neutral, -1, 1)


def _program_and_rules(compiled_path, text, born, loop, veto=(), *, u1_grades=False):
    """The production CT program (``ct_closure.counterterm_program_blocks``,
    as ``LoopSlice.generate`` builds it; ``u1_grades`` as for a leading- or
    next-to-leading-colour request) of ``text`` on a compiled model, with the
    request's rules, native model, selection and the CT-lowered model's
    process IR."""

    import json

    from pyamplicol.loops.v2.assemble import (
        SliceRequest,
        prepare_request_model,
        resolve_request,
    )
    from pyamplicol.loops.v2.blocks import (
        build_tree_blocks,
        vetoed_particle_ids,
        with_ward_kernels,
    )
    from pyamplicol.loops.v2.ct_closure import counterterm_program_blocks
    from pyamplicol.loops.v2.families import LoopOrderSelection, add_orders
    from pyamplicol.models.external import CompiledUFOModel
    from pyamplicol.models.loading import load_compiled_model
    from pyamplicol.processes.model import build_model_process_ir

    compiled = load_compiled_model(compiled_path)
    # The model's strong correction (``resolve_request``: SliceRequest
    # names no correction by default).
    request = resolve_request(
        compiled,
        SliceRequest(
            text,
            born,
            loop,
            fixed_parameters=legacy_capture.FIXED_PARAMETERS,
            particle_veto=tuple(veto),
        ),
    )
    prepared, continuation, fixed, _scale, _alpha = prepare_request_model(
        compiled, request
    )
    data = json.loads(prepared.structure.native.to_json())
    selection = LoopOrderSelection(
        frozenset(request.corrections),
        frozenset(add_orders(tuple(o.items())) for o in request.exact_loop_orders),
    )
    blocks = build_tree_blocks(
        build_model_process_ir(text, compiled.ir, color_accuracy="full"),
        CompiledUFOModel(compiled),
        born_coupling_orders=dict(request.born_orders),
        block_coupling_orders=request.loop_order_bounds,
        ignored_particle_ids=vetoed_particle_ids(request.particle_veto),
    )
    blocks, compiled = with_ward_kernels(blocks, compiled, fixed)
    program = counterterm_program_blocks(
        compiled,
        prepared,
        text,
        data,
        selection,
        blocks.born_orders,
        veto=request.particle_veto,
        fixed_parameters=fixed,
        declaration=request.colour_declaration,
        continuation=continuation,
        scope=(
            tuple(sorted(request.born_orders.items())),
            tuple(sorted(request.loop_order_bounds.items())),
            request.compatibility,
        ),
        u1_grades=u1_grades,
    )
    return program, prepared.counterterms.rules, data, selection


def _rules_in_program(program, rules, data, selection):
    """``(rules of the CT vertices and clone lines the program's DAG
    interactions use, rules of its CT-root sites)``, read from the lowered
    model: interaction vertex kind -> kernel terms -> couplings -> lowered
    structures -> rule."""

    from pyamplicol.loops.v2.ct_closure import CountertermClosure
    from pyamplicol.loops.v2.ct_recursion import rule_key

    model = program.recursion.model
    lowered = model.lowered.compiled
    name = {rule_key(rule): rule_name for rule_name, rule in rules.items()}
    of_structure = {
        structure: name[key]
        for key, structures in model.structures.items()
        for structure in structures
    }
    by_coupling: dict[str, set[str]] = {}
    for structure, coupling in model.lowered.couplings.items():
        by_coupling.setdefault(coupling, set()).add(of_structure[structure])
    by_clone: dict[str, set[str]] = {}
    for structure, clone in model.lowered.insertions.items():
        by_clone.setdefault(clone, set()).add(of_structure[structure])
    coupling = {int(term.id): term.coupling for term in lowered.ir.vertex_terms}
    by_kind: dict[int, set[str]] = {}
    for kernel in lowered.ir.oriented_kernels:
        for term_id in kernel.term_ids or (kernel.term_id,):
            by_kind.setdefault(int(kernel.kind), set()).update(
                by_coupling.get(coupling.get(int(term_id), ""), ())
            )
    particle = {int(p.pdg_code): p.name for p in lowered.ir.particles}
    dag = program.blocks.blocks_dag
    inside = set().union(
        *(by_kind.get(int(i.vertex_kind), set()) for i in dag.interactions),
        *(
            by_clone.get(particle[int(c.index.particle_id)], set())
            for c in dag.currents
        ),
    )
    root = {
        site.rule_name
        for site in CountertermClosure.root_sites(program, data, selection)
    }
    return inside, root


@pytest.mark.loop_physics
def test_insertion_reachability_is_sound_against_the_program_dag(
    loop_sm_compiled_model,
):
    """``ct_closure.InsertionReachability`` admits every rule the production
    CT program itself uses, in its DAG (CT vertices and clone lines inside
    the currents) or at its CT roots, on a charged-current request: in
    u d~ > w+ g the R2 of [u~ d W+] sits inside the current of legs 1 and 4
    and that of its conjugate [d~ u W-] at the root (a reading of every
    vertex by its incoming fields admits only the latter)."""

    from pyamplicol.loops.v2.ct_closure import InsertionReachability
    from pyamplicol.models.external import CompiledUFOModel
    from pyamplicol.processes.model import build_model_process_ir

    program, rules, data, selection = _program_and_rules(
        loop_sm_compiled_model,
        "u d~ > w+ g",
        {"QCD": 1, "QED": 1},
        {"QCD": 3, "QED": 1},
    )
    inside, root = _rules_in_program(program, rules, data, selection)
    lowered = program.recursion.model.lowered.compiled
    reachability = InsertionReachability.build(
        CompiledUFOModel(lowered),
        build_model_process_ir("u d~ > w+ g", lowered.ir, color_accuracy="full"),
        program.recursion.coupling_order_limits(),
    )
    used = inside | root
    assert used and used <= set(program.recursion.rules)
    assert not {n for n in used if not reachability.admits(rules[n])}

    def of_fields(*pdgs):
        return {n for n in used if sorted(rules[n].particle_pdgs) == sorted(pdgs)}

    assert of_fields(-2, 1, 24) and of_fields(-2, 1, 24) <= inside - root
    assert of_fields(-1, 2, -24) and of_fields(-1, 2, -24) <= root - inside


@pytest.mark.loop_physics
def test_unlowerable_charged_vertex_in_a_current_is_refused(
    loop_sm_compiled_model, monkeypatch
):
    """REQUIREMENTS_ADDENDUM J: an unlowerable CT vertex that can enter a
    tree is refused, also when it enters only inside a current. With the R2
    of [u~ d W+] made unlowerable, u d~ > w+ g is refused naming it (its
    vertex sits in the current of legs 1 and 4); its conjugate stays
    lowered."""

    from collections import OrderedDict

    from pyamplicol.loops.v2 import ct_closure, ct_recursion

    original = ct_recursion.TreeVocabulary.rule_reason

    def rule_reason(self, rule):
        if rule.kind == "R2" and sorted(rule.particle_pdgs) == [-2, 1, 24]:
            return "made unlowerable (test)"
        return original(self, rule)

    monkeypatch.setattr(ct_recursion.TreeVocabulary, "rule_reason", rule_reason)
    # A lowered model of this request without the patch must not be reused,
    # nor this one kept.
    monkeypatch.setattr(ct_closure, "_LOWERED", OrderedDict())
    with pytest.raises(
        ct_recursion.UnloweredCountertermError, match=r"\[u~ d W\+\] \(made"
    ):
        _program_and_rules(
            loop_sm_compiled_model,
            "u d~ > w+ g",
            {"QCD": 1, "QED": 1},
            {"QCD": 3, "QED": 1},
        )


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
