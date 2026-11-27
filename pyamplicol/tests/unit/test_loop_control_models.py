# SPDX-License-Identifier: 0BSD
"""The calibrated mass restriction is structural, explicit and model-local."""

import json

import pytest

pytestmark = pytest.mark.loop_fast


@pytest.fixture(scope="module")
def dependencies():
    symbolica = pytest.importorskip("symbolica")
    if not symbolica.is_licensed():
        pytest.skip("licensed native graph generation required")
    from symbolica.community import hepkit

    return symbolica, hepkit


def test_control_restriction_preserves_generic_inputs_and_heavy_mass_symbols(
    dependencies,
):
    symbolica, h = dependencies
    from pyamplicol.loops.control_models import qcd_qed_control_model

    generic = h.Model.qcd_qed()
    original = generic.to_json()
    model = qcd_qed_control_model()
    assert model.restriction == "pyamplicol-massless-uds-control"
    parameters = {p["name"]: p for p in json.loads(model.to_json())["parameters"]}
    for name in ("u", "u~", "d", "d~", "s", "s~"):
        assert model.particle(name).mass == 0
        assert model.particle(name).mass_parameter == "ZERO"
    for name in ("MU", "MD", "MS"):
        if name in parameters:
            assert parameters[name]["nature"] == "internal"
            assert parameters[name]["expression"] == "0"
    for particle, mass in (("c", "MC"), ("b", "MB"), ("t", "MT")):
        assert model.particle(particle).mass == symbolica.S(f"UFO::{mass}")
        assert parameters[mass]["nature"] == "external"
        assert parameters[mass] == next(
            p for p in json.loads(original)["parameters"] if p["name"] == mass
        )
    assert generic.to_json() == original
    assert h.Model.qcd_qed().to_json() == original


def test_control_restriction_reaches_numerators_denominators_and_families(dependencies):
    from pyamplicol.loops.control_models import qcd_qed_control_model
    from tools.loop_reference.feynkit_loop_graphs import build_loop_process

    model = qcd_qed_control_model()
    ir = build_loop_process(
        model.process(["u", "u~"], ["d", "d~"]),
        selected_corrections=["QCD"],
        perturbative_orders={"QCD": 1},
        born_order_bounds={"QCD": 2, "QED": 0},
        loop_order_bounds={"QCD": 4, "QED": 0},
    )
    assert len(ir.born) == 1 and len(ir.loops) == 14 and len(ir.families) == 8
    forbidden = {f"UFO::{name}" for name in ("MU", "MD", "MS")}
    expressions = []
    for graph in (*ir.born, *ir.loops):
        for edge in graph.native.edges:
            if abs(edge.particle.pdg_code) in (1, 2, 3):
                assert edge.particle.mass == 0
            expressions.append(edge.numerator_expression().to_expression())
    for graph in ir.loops:
        expressions.extend(graph.native.propagator_family().denominators)
    assert all(
        not forbidden.intersection(s.get_name() for s in value.get_all_symbols())
        for value in expressions
    )
