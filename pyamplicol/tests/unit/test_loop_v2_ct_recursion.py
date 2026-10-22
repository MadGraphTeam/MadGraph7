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

import math
from collections import Counter
from decimal import Decimal
from pathlib import Path

import pytest

from pyamplicol.loops.v2.ct_sites import LAURENT_POWERS
from tools.loop_reference import legacy_capture

from .test_loop_v2_counterterms import (
    CONTROLS,
    TOLERANCE,
    _components,
    _scalars,
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
    """loop_sm with every lowerable CT rule in its tree model, prepared once."""

    from pyamplicol import ModelSource
    from pyamplicol.config import EvaluatorConfig
    from pyamplicol.loops.compiler import prepare_loop_request_model
    from pyamplicol.loops.v2.ct_recursion import lower_model_counterterms
    from pyamplicol.models.loading import load_compiled_model

    compiled = load_compiled_model(loop_sm_compiled_model)
    prepared = prepare_loop_request_model(
        compiled, fixed_parameters=legacy_capture.FIXED_PARAMETERS
    )
    lowered = lower_model_counterterms(compiled, prepared.counterterms.rules)
    root = tmp_path_factory.mktemp("ct-lowered")
    path = lowered.compiled.write(root / "loop-sm-ct.pyamplicol-model.json")
    bundle = root / "loop-sm-ct.pyamplicol-model"
    ModelSource.from_path(path).compile(
        cache_dir=root / "cache", prepared_output=bundle, evaluator=EvaluatorConfig()
    )
    return lowered, ModelSource.from_path(bundle)


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
    Generator(config).generate(ct.control.process, artifact, model=source)
    return Runtime.load(artifact)


def _model_parameters(ct, index: int) -> dict[str, str]:
    fixture = ct.fixture
    scale = Decimal(fixture["scales_squared"][index]).sqrt()
    parameters = {**legacy_capture.PARAMETERS, "MU_R": str(scale)}
    parameters.update(legacy_capture.FIXED_PARAMETERS)
    if ct.control.drell_yan:
        parameters["aEWM1"] = legacy_capture.DRELL_YAN_MODEL_VALUES[1]
    return parameters


def _recursion_components(ct, recursion, runtime, index: int, *, poles: bool = True):
    """Born and the recursion's CT partials at one fixture point (binary64)."""

    parameters = _model_parameters(ct, index)
    plan = ct.prepared.structure.scalar_bindings(recursion.scalar_outputs())
    scalars = {
        key: complex(float(re), float(im))
        for key, (re, im) in _scalars(ct, plan, parameters, "double").items()
    }
    runtime.set_model_parameters(
        {
            **{name: float(value) for name, value in parameters.items()},
            **recursion.parameter_values(scalars, poles=poles),
        }
    )
    point = [[float(x) for x in momentum] for momentum in ct.fixture["points"][index]]
    evaluation = runtime.evaluate_orders([point])
    born = {
        order: complex(evaluation.for_order(order)[0])
        for order in evaluation.orders
        if not any(name.startswith("PYAMPLICOL_CT_") for name, _ in order)
    }
    parts = {
        key: complex(values[0])
        for key, values in recursion.components(evaluation).items()
    }
    return born, parts


@pytest.mark.loop_fast
def test_rules_split_between_recursion_and_sites(ct_slice, lowered_ct_model):
    """Two-point insertions and the fixed-SU(3) four-gluon R2 stay sites; every
    three-gluon, quark-gluon and four-gluon UV rule is in the recursion with
    its admitted alternatives."""

    lowered, _source = lowered_ct_model
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
        "PYAMPLICOL_CT_INSERTION": 1,
    }


@pytest.mark.loop_physics
@pytest.mark.parametrize("lane", LANES)
@pytest.mark.parametrize("name", CONTROLS)
def test_recursion_equals_site_closure(
    ct_slice, lowered_ct_model, loop_native, tmp_path, name, lane
):
    """O6: Born x CT per slot equals the sites of the same rules, and with the
    remaining sites the UV_local and R2 components equal O1 (binary64)."""

    lowered, source = lowered_ct_model
    ct = ct_slice(name)
    recursion = _recursion(ct, lowered)
    runtime = _program(ct, recursion, source, tmp_path / "artifact", lane)
    fixture = ct.fixture
    results = fixture["results"]["double"]
    sectors = _sector_orders(fixture, "virtual")
    born_sectors = _sector_orders(fixture, "born")
    tolerance = float(TOLERANCE["double"])
    for index in range(len(fixture["points"])):
        born, parts = _recursion_components(ct, recursion, runtime, index)
        _, sites = _components(ct, "double", index, closed_rules=recursion.rules)
        _, residual = _components(ct, "double", index, closed_rules=recursion.residual)
        scale_born = max(abs(float(row[0])) for row in results["born"][index])
        for b, order in enumerate(born_sectors):
            reference = float(results["born"][index][b][0])
            assert math.isclose(born[order].real, reference, rel_tol=1e-12)
        assert {key[0] for key in parts} <= set(CT_COMPONENTS)
        for component in CT_COMPONENTS:
            for s, order in enumerate(sectors):
                for k, power in enumerate(LAURENT_POWERS):
                    key = (component, order, power)
                    mine = parts.get(key, 0j)
                    site = complex(sites.get(key, 0j))
                    reference = float(results[component][index][s][k][0])
                    scale = abs(reference) or scale_born
                    assert abs(mine.imag) <= tolerance * scale, key
                    assert abs(mine.real - site.real) <= tolerance * scale, (
                        key,
                        index,
                        mine,
                        site,
                    )
                    total = mine.real + complex(residual.get(key, 0j)).real
                    assert abs(total - reference) <= tolerance * scale, (
                        key,
                        index,
                        total,
                        reference,
                    )


@pytest.mark.loop_physics
def test_finite_only_binds_no_pole_slot(ct_slice, lowered_ct_model, tmp_path):
    """Finite-only evaluation of the same program: pole slots are exactly zero
    and the finite slots are unchanged."""

    lowered, source = lowered_ct_model
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
