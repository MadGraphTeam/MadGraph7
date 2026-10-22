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

from collections import Counter
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
    """``forbidden |PDG| codes -> (CountertermModel, prepared source)``:
    loop_sm with every lowerable CT rule in its tree model, prepared once per
    veto."""

    from pyamplicol import ModelSource
    from pyamplicol.config import EvaluatorConfig
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
            lowered = lower_model_counterterms(
                compiled,
                prepared.counterterms.rules,
                fixed_parameters=prepared.structure.fixed_parameters,
                forbidden=key,
            )
            root = tmp_path_factory.mktemp("ct-lowered")
            path = lowered.lowered.compiled.write(
                root / "loop-sm-ct.pyamplicol-model.json"
            )
            bundle = root / "loop-sm-ct.pyamplicol-model"
            ModelSource.from_path(path).compile(
                cache_dir=root / "cache",
                prepared_output=bundle,
                evaluator=EvaluatorConfig(),
            )
            built[key] = (lowered, ModelSource.from_path(bundle))
        return built[key]

    return build


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
                mine = parts.get((component, power), Decimal(0))
                site = sum((_site_value(sites, key) for key in keys), Decimal(0))
                rest = sum((_site_value(residual, key) for key in keys), Decimal(0))
                assert abs(mine - site) <= tolerance * scale, (component, power, index)
                assert abs(mine + rest - reference) <= tolerance * scale, (
                    component,
                    power,
                    index,
                )


@pytest.mark.loop_fast
def test_rules_split_between_recursion_and_sites(ct_slice, lowered_ct_model):
    """Two-point insertions and the fixed-SU(3) four-gluon R2 stay sites; every
    three-gluon, quark-gluon and four-gluon UV rule is in the recursion with
    its admitted alternatives."""

    lowered, _source = lowered_ct_model()
    ct = ct_slice("gg_gg")
    recursion = _recursion(ct, lowered)
    assert set(recursion.rules) | set(recursion.residual) == set(ct.rules)
    reasons = Counter(
        "two-point" if "two-point" in reason else "multi-trace"
        for reason in recursion.residual.values()
    )
    assert set(reasons) == {"two-point", "multi-trace"}
    for name, reason in recursion.residual.items():
        rule = ct.rules[name]
        if len(rule.particle_pdgs) > 2:
            assert {e.vertex_name for e in rule.entries} == {"V_R2RGA"}, reason
    # Pure-gluon rules are admitted with all their alternatives (every quark
    # flavour, the top and the gluon run in the loops of g g > g g); rules
    # with quark legs have no witness in this process.
    for name, rule in recursion.rules.items():
        gluons = set(rule.particle_pdgs) == {21}
        expected = tuple(range(len(rule.alternatives))) if gluons else ()
        assert recursion.admission[name] == expected, name
    assert recursion.slots == (("r2", 0), ("uv_local", -1), ("uv_local", 0))
    assert recursion.coupling_order_limits() == {
        "QCD": 4,
        "QED": 0,
        "PYAMPLICOL_CT_INSERTION": 1,
    }


@pytest.mark.loop_physics
@pytest.mark.parametrize("lane", LANES)
@pytest.mark.parametrize("name", CONTROLS)
def test_recursion_equals_site_closure(
    ct_slice, lowered_ct_model, loop_native, tmp_path, name, lane
):
    """O6: Born x CT per slot equals the sites of the same rules, and with the
    remaining sites the UV_local and R2 components equal O1 (binary64). The
    Rust recurrence lane is checked per squared order, the Python eager lane
    per slot summed over squared orders (it has no order partials)."""

    ct = ct_slice(name)
    lowered, source = lowered_ct_model(ct.vetoed)
    recursion = _recursion(ct, lowered)
    runtime = _program(ct, recursion, source, tmp_path / "artifact", lane)
    if lane == "eager":
        _assert_eager_equal_sites_and_legacy(ct, recursion, runtime)
    else:
        _assert_equal_sites_and_legacy(ct, recursion, runtime, "double")


# The tree runtime keeps the U(1)-subtraction colour weight 1/N_c of
# processes with several quark lines in binary64 (``GenericDAG.color_weight``,
# wave01-open-minors), so their 200-bit Born is exact to ~1e-17 only.
_BINARY64_U1_WEIGHT = pytest.mark.xfail(
    strict=True,
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
    lowered, source = lowered_ct_model(ct.vetoed)
    recursion = _recursion(ct, lowered)
    runtime = _program(ct, recursion, source, tmp_path / "artifact", "recurrence")
    _assert_equal_sites_and_legacy(ct, recursion, runtime, "arbitrary-200")


@pytest.mark.loop_physics
def test_finite_only_binds_no_pole_slot(ct_slice, lowered_ct_model, tmp_path):
    """Finite-only evaluation of the same program: pole slots are exactly zero
    and the finite slots are unchanged."""

    lowered, source = lowered_ct_model()
    ct = ct_slice("uux_ddxg")
    recursion = _recursion(ct, lowered)
    runtime = _program(ct, recursion, source, tmp_path / "artifact", "recurrence")
    _, full = _recursion_components(ct, recursion, runtime, 0)
    _, finite = _recursion_components(ct, recursion, runtime, 0, poles=False)
    assert any(power and value for (_c, _o, power), value in full.items())
    for (component, order, power), value in finite.items():
        if power:
            assert value == 0
        else:
            assert value == full[component, order, power]
