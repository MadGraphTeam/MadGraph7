# SPDX-License-Identifier: 0BSD
"""Model field factors and native Born identity, independent of graph CT rules."""

import json
from dataclasses import replace
from types import ModuleType, SimpleNamespace

import pytest

from pyamplicol.loops.graphs import build_loop_process
from pyamplicol.loops.model import build_loop_model
from pyamplicol.loops.model_expressions import parse_exact_ufo
from pyamplicol.loops.wavefunction import (
    match_external_field_counterterms,
    prepare_external_field_rules,
)
from pyamplicol.models.loop_import import capture_raw_loop_model, load_raw_loop_model

pytestmark = pytest.mark.loop_fast


@pytest.fixture(scope="module")
def hep():
    symbolica = pytest.importorskip("symbolica")
    if not symbolica.is_licensed():
        pytest.skip("native Born graphs require licensed Symbolica")
    return pytest.importorskip("symbolica.community.hepkit")


def _source(hep, values=None, alternatives=None):
    native = hep.Model.qcd_qed()
    module = ModuleType("external_field_fixture")
    module.all_particles = [
        SimpleNamespace(**record)
        for record in json.loads(native.to_json())["particles"]
    ]
    module.all_orders = [
        SimpleNamespace(name=name, perturbative_expansion=1) for name in ("QCD", "QED")
    ]
    module.all_CTparameters = [
        SimpleNamespace(name="WF_ALIAS", value={-1: "a", 0: "b", 1: "c"})
    ]
    module.all_parameters = [SimpleNamespace(name="indirect", value="WF_ALIAS**2")]
    particle = next(p for p in module.all_particles if p.pdg_code == 2)
    particle.loop_particles = [
        [[2, 21], [2, 21], []] if alternatives is None else alternatives,
        [[2, 22]],
    ]
    particle.counterterm = (
        values
        if values is not None
        else {
            (1, 0, 0): {-1: "indirect", 0: "WF_ALIAS"},
            (0, 1, 1): {0: "e"},
        }
    )
    return module, native


@pytest.fixture(scope="module")
def dy(hep):
    return build_loop_process(
        hep.Model.qcd_qed().process(["u", "u~"], ["e-", "e+"]),
        selected_corrections=["QCD"],
        perturbative_orders={"QCD": 1},
    )


def test_ct_aliases_are_substituted_before_field_laurent_extraction(hep):
    module, _ = _source(hep)
    source = build_loop_model(capture_raw_loop_model(module))
    qcd, qed = prepare_external_field_rules(source)
    assert qcd.order == (("QCD", 2),) and qed.order == (("QED", 2),)
    # (a/eps+b+c*eps)^2 / eps + (a/eps+b+c*eps)
    assert qcd.coefficient.coefficient(-3) == parse_exact_ufo("a**2")
    assert qcd.coefficient.coefficient(0) == parse_exact_ufo("2*b*c+b")
    assert qcd.alternatives == ((2, 21), (2, 21), ())


def test_each_external_particle_and_antiparticle_and_alternative_counts(hep, dy):
    module, _ = _source(hep, values={(1, 0, 0): {-1: "a", 0: "b"}})
    source = build_loop_model(capture_raw_loop_model(module))
    result = match_external_field_counterterms(source, dy, selected_corrections=["QCD"])
    assert len(result.insertions) == 6
    assert {item.external_index for item in result.insertions} == {0, 1}
    assert {item.alternative_index for item in result.insertions} == {0, 1, 2}
    assert all(item.born is dy.born[0] for item in result.insertions)
    assert all(item.order == (("QCD", 2), ("QED", 2)) for item in result.insertions)
    assert sum(item.coefficient(0) for item in result.insertions) == parse_exact_ufo(
        "6*b"
    )
    assert not result.scheme_validated
    vetoed = match_external_field_counterterms(
        source, dy, selected_corrections=["QCD"], forbidden_particles=[-21]
    )
    assert len(vetoed.insertions) == 2  # explicit empty alternatives survive
    filtered = match_external_field_counterterms(
        source,
        dy,
        selected_corrections=["QCD"],
        allowed_amplitude_orders=[(("QED", 4),)],
    )
    assert not filtered.insertions
    with pytest.raises(ValueError, match="correction selection"):
        match_external_field_counterterms(source, dy, selected_corrections=["QED"])


def test_field_order_uses_model_registry_order_and_skips_higher_loops(hep):
    module, _ = _source(hep, values={(0, 1, 0): {0: "b"}, (2, 0, 1): {0: "c"}})
    module.all_orders.reverse()
    rules = prepare_external_field_rules(
        build_loop_model(capture_raw_loop_model(module))
    )
    assert len(rules) == 1 and rules[0].order == (("QCD", 2),)


@pytest.mark.parametrize("bad", [0.1, complex(1, 2), True])
def test_rounded_or_nonexpression_particle_field_values_fail(hep, bad):
    module, _ = _source(hep, values={(1, 0, 0): {0: bad}})
    with pytest.raises(ValueError, match="exact expression"):
        prepare_external_field_rules(build_loop_model(capture_raw_loop_model(module)))


def test_field_mismatched_ports_fail(hep, dy):
    module, _ = _source(hep)
    particle = next(p for p in module.all_particles if p.pdg_code == 2)
    particle.spin = 1
    source = build_loop_model(capture_raw_loop_model(module))
    with pytest.raises(ValueError, match="representations differ"):
        match_external_field_counterterms(source, dy, selected_corrections=["QCD"])


def test_real_top_field_factor_has_model_half_z_and_two_external_copies(
    loop_sm_ufo, hep
):
    path = str(loop_sm_ufo)
    source = build_loop_model(load_raw_loop_model(path))
    rules = prepare_external_field_rules(source)
    top = next(rule for rule in rules if rule.pdg == 6)
    assert top.alternatives == ((6, 21),)
    assert top.coefficient.coefficient(-1) == parse_exact_ufo(
        "cond(MT,0,-G**2*3*CF/(32*cmath.pi**2))"
    )
    assert top.coefficient.coefficient(0) == parse_exact_ufo(
        "cond(MT,0,-G**2*CF*(4-3*reglog(MT**2/MU_R**2))/(32*cmath.pi**2))"
    )
    process = hep.Model.qcd_qed().process(["u", "u~"], ["t", "t~"])
    loops = build_loop_process(
        process,
        selected_corrections=["QCD"],
        perturbative_orders={"QCD": 1},
        born_order_bounds={"QED": 0},
        loop_order_bounds={"QED": 0},
    )
    result = match_external_field_counterterms(
        source, loops, selected_corrections=["QCD"]
    )
    assert len(result.insertions) == 2
    assert {item.external_index for item in result.insertions} == {2, 3}
    assert all(item.order == (("QCD", 4),) for item in result.insertions)
    assert (
        sum(item.coefficient(0) for item in result.insertions)
        - 2 * top.coefficient.coefficient(0)
    ).expand() == 0
    # Selection does not require a matching admitted loop witness; these are
    # external factors on Borns, unlike R2/UVmass local insertions.
    no_loops = replace(loops, loops=())
    assert (
        len(
            match_external_field_counterterms(
                source, no_loops, selected_corrections=["QCD"]
            ).insertions
        )
        == 2
    )
